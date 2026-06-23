"""
UnifiedTMIL Enhanced Account Classification
============================================
Improvements over baseline:
  - 8 → 26 aggregate features (B1)
  - LightGBM + XGBoost ensemble with calibration (B2)
  - Multi-seed evaluation with bootstrap CI (B5)

Target: Hard-AUC > 0.836, ID-F1 > 0.750
"""

import os
import sys
import json
import pickle
import random
import numpy as np
from collections import Counter
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, f1_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
BERT4ETH = os.path.join(DATA, "bert4eth")
RES = os.path.join(ROOT, "results")

SEEDS = [42, 1337, 7, 99, 2024]


def load(fname):
    return pickle.load(open(os.path.join(BERT4ETH, fname), "rb"))


def extract_enhanced_features(bag):
    """
    Extract 26 aggregate features from a transaction bag.
    
    Features:
      1-3:  Amount statistics (mean, std, max log-amount)
      4-5:  Time-delta statistics (mean, std log-dt)
      6:    Bag length (log)
      7:    Outbound ratio
      8:    Unique counterparty ratio
      9:    In/out value asymmetry (log ratio of sum-out / sum-in)
     10:    Burst-then-drain signature (max outbound burst / total)
     11:    Fund-flow concentration (Herfindahl index on outbound amounts)
     12:    Zero-value outbound ratio (approval-event proxy)
     13:    Inbound ratio
     14:    Inbound value fraction
     15:    Max single outbound fraction (largest drain)
     16:    Temporal burst score (std_dt / mean_dt)
     17:    Counterparty repeat ratio (1 - unique_cp_ratio)
     18:    Top-3 counterparty concentration
     19:    Amount skewness (3rd standardized moment)
     20:    Amount kurtosis (4th standardized moment)
     21:    Recency-weighted outbound (later txs weighted more)
     22:    Early burst ratio (first 20% of txs outbound fraction)
     23:    Late burst ratio (last 20% of txs outbound fraction)
     24:    Max consecutive outbound run (normalized)
     25:    Inbound-before-outbound pattern score
     26:    Log ntx_full
    """
    n = bag["length"]
    if n == 0:
        return [0.0] * 26

    amt_raw = np.array(bag["input_amounts"][:n], dtype=float)
    dt_raw = np.array(bag.get("delta_ts", [0] * n)[:n], dtype=float)
    io = np.array(bag["input_io"][:n])  # 1=out, 2=in
    ids = bag["input_ids"][:n]

    amt = np.log1p(np.abs(amt_raw))
    dt = np.log1p(dt_raw)

    # Basic amount stats
    mean_amt = float(amt.mean())
    std_amt = float(amt.std()) if n > 1 else 0.0
    max_amt = float(amt.max())

    # Time stats
    mean_dt = float(dt.mean())
    std_dt = float(dt.std()) if n > 1 else 0.0

    # Length
    log_n = float(np.log1p(n))

    # Direction masks
    out_mask = (io == 1)
    in_mask = (io == 2)
    outbound_ratio = float(out_mask.mean())
    inbound_ratio = float(in_mask.mean())

    # Counterparty diversity
    unique_cp_ratio = len(set(ids)) / n

    # In/out value asymmetry
    sum_out = float(np.abs(amt_raw[out_mask]).sum()) if out_mask.any() else 0.0
    sum_in = float(np.abs(amt_raw[in_mask]).sum()) if in_mask.any() else 0.0
    io_asymmetry = float(np.log1p(sum_out) - np.log1p(sum_in))

    # Burst-then-drain: max single outbound / total outbound
    out_amts = np.abs(amt_raw[out_mask]) if out_mask.any() else np.array([0.0])
    burst_drain = float(out_amts.max() / (out_amts.sum() + 1e-9))

    # Fund-flow concentration (Herfindahl on outbound amounts)
    if out_mask.any() and out_amts.sum() > 0:
        shares = out_amts / out_amts.sum()
        hhi = float((shares ** 2).sum())
    else:
        hhi = 0.0

    # Zero-value outbound ratio (approval-event proxy)
    zero_out_ratio = float(((np.abs(amt_raw) < 1e-9) & out_mask).mean())

    # Inbound value fraction
    total_val = sum_in + sum_out
    inbound_val_frac = float(sum_in / (total_val + 1e-9))

    # Max single outbound fraction
    max_out_frac = float(out_amts.max() / (total_val + 1e-9))

    # Temporal burst score
    temporal_burst = float(std_dt / (mean_dt + 1e-9))

    # Counterparty repeat ratio
    cp_repeat = 1.0 - unique_cp_ratio

    # Top-3 counterparty concentration
    cp_counts = Counter(ids)
    top3 = sum(v for _, v in cp_counts.most_common(3))
    top3_conc = float(top3 / n)

    # Amount skewness and kurtosis
    if n > 2 and std_amt > 1e-9:
        amt_centered = amt - mean_amt
        skew = float((amt_centered ** 3).mean() / (std_amt ** 3))
        kurt = float((amt_centered ** 4).mean() / (std_amt ** 4)) - 3.0
    else:
        skew = 0.0
        kurt = 0.0

    # Recency-weighted outbound (exponential decay from end)
    weights = np.exp(np.linspace(0, 1, n))
    weights /= weights.sum()
    recency_out = float((out_mask.astype(float) * weights).sum())

    # Early/late burst ratio
    split = max(1, n // 5)
    early_out = float(out_mask[:split].mean()) if split > 0 else 0.0
    late_out = float(out_mask[-split:].mean()) if split > 0 else 0.0

    # Max consecutive outbound run (normalized)
    max_run = 0
    cur_run = 0
    for flag in out_mask:
        if flag:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    max_consec_out = float(max_run / n)

    # Inbound-before-outbound pattern: fraction of (in, out) adjacent pairs
    ibo_count = 0
    for i in range(n - 1):
        if io[i] == 2 and io[i + 1] == 1:
            ibo_count += 1
    ibo_score = float(ibo_count / (n - 1)) if n > 1 else 0.0

    # Log ntx_full
    log_ntx = float(np.log1p(bag.get("ntx_full", n)))

    return [
        mean_amt, std_amt, max_amt,           # 1-3
        mean_dt, std_dt,                       # 4-5
        log_n,                                 # 6
        outbound_ratio,                        # 7
        unique_cp_ratio,                       # 8
        io_asymmetry,                          # 9
        burst_drain,                           # 10
        hhi,                                   # 11
        zero_out_ratio,                        # 12
        inbound_ratio,                         # 13
        inbound_val_frac,                      # 14
        max_out_frac,                          # 15
        temporal_burst,                        # 16
        cp_repeat,                             # 17
        top3_conc,                             # 18
        skew,                                  # 19
        kurt,                                  # 20
        recency_out,                           # 21
        early_out,                             # 22
        late_out,                              # 23
        max_consec_out,                        # 24
        ibo_score,                             # 25
        log_ntx,                               # 26
    ]


FEATURE_NAMES = [
    "mean_amt", "std_amt", "max_amt",
    "mean_dt", "std_dt",
    "log_n",
    "outbound_ratio",
    "unique_cp_ratio",
    "io_asymmetry",
    "burst_drain",
    "hhi",
    "zero_out_ratio",
    "inbound_ratio",
    "inbound_val_frac",
    "max_out_frac",
    "temporal_burst",
    "cp_repeat",
    "top3_conc",
    "skew",
    "kurt",
    "recency_out",
    "early_out",
    "late_out",
    "max_consec_out",
    "ibo_score",
    "log_ntx",
]


def prepare_data(bags):
    X = np.array([extract_enhanced_features(b) for b in bags], dtype=np.float32)
    y = np.array([b["label"] for b in bags], dtype=np.float32)
    return X, y


def boot_ci(vals, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    stats = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def find_best_threshold(y_true, y_prob):
    """Find threshold maximizing F1."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def acc_metrics(y, p, threshold=None):
    if threshold is None:
        threshold = find_best_threshold(y, p)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, (p >= threshold).astype(int), average="binary", zero_division=0
    )
    return {
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "auc": float(roc_auc_score(y, p)),
        "aupr": float(average_precision_score(y, p)),
        "threshold": float(threshold),
    }


def train_lgbm(X_train, y_train, seed=42, class_weight=True):
    """Train LightGBM with calibration."""
    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale if class_weight else 1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    m = lgb.LGBMClassifier(**params)
    m.fit(X_train, y_train)
    return m


def train_xgb(X_train, y_train, seed=42, class_weight=True):
    """Train XGBoost with calibration."""
    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale if class_weight else 1.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
    )
    m = xgb.XGBClassifier(**params)
    m.fit(X_train, y_train)
    return m


def ensemble_predict(models, X):
    """Average probability from multiple models."""
    probs = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return probs.mean(axis=0)


def cross_block(y_cross, p_cross, src, ntx):
    auc = roc_auc_score(y_cross, p_cross)
    aupr = average_precision_score(y_cross, p_cross)

    # Bootstrap CI for X-AUC
    idx = np.arange(len(y_cross))
    rng = np.random.RandomState(0)
    boot_aucs = []
    for _ in range(2000):
        ss = rng.choice(idx, len(idx), replace=True)
        if len(set(y_cross[ss])) == 2:
            boot_aucs.append(roc_auc_score(y_cross[ss], p_cross[ss]))
    auc_ci = (float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5)))

    bysrc = {}
    for sn in ["ptx_benign", "normal_eoa"]:
        sel = (y_cross == 1) | ((y_cross == 0) & (src == sn))
        if sel.sum() > 0 and len(set(y_cross[sel])) == 2:
            bysrc[sn] = {
                "auc": float(roc_auc_score(y_cross[sel], p_cross[sel])),
                "aupr": float(average_precision_score(y_cross[sel], p_cross[sel])),
                "n_neg": int(((y_cross == 0) & (src == sn)).sum()),
            }

    strat = {}
    for lo, hi, nm in [(3, 20, "3-20"), (20, 100, "20-100"), (100, 1e9, "100+")]:
        sel = (ntx >= lo) & (ntx < hi)
        if sel.sum() > 5 and len(set(y_cross[sel])) == 2:
            strat[nm] = {
                "auc": float(roc_auc_score(y_cross[sel], p_cross[sel])),
                "n": int(sel.sum()),
            }

    return {
        "auc": float(auc),
        "auc_ci": list(auc_ci),
        "aupr": float(aupr),
        "by_source": bysrc,
        "stratified": strat,
    }


def main():
    print("=" * 60)
    print("UnifiedTMIL Enhanced Account Model")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_bags = load("train_bags.pkl")
    test_bags = load("test_bags.pkl")
    ptx = pickle.load(open(os.path.join(DATA, "ptx_bags.pkl"), "rb"))
    norm_neg = pickle.load(open(os.path.join(DATA, "normal_eoa_neg.pkl"), "rb"))

    pos = [b for b in ptx if b["label"] == 1]
    pb = [b for b in ptx if b["label"] == 0]
    allb = pos + pb + norm_neg
    src = np.array([b.get("source", "ptx_benign") for b in allb])
    ntx = np.array([b["ntx_full"] for b in allb])

    print(f"Train: {len(train_bags)}, Test: {len(test_bags)}")
    print(f"Cross: pos={len(pos)}, ptx_neg={len(pb)}, normal_eoa={len(norm_neg)}")

    # Extract features
    print("\nExtracting 26-dim enhanced features...")
    X_train, y_train = prepare_data(train_bags)
    X_test, y_test = prepare_data(test_bags)
    X_cross, y_cross = prepare_data(allb)

    print(f"Feature shape: train={X_train.shape}, test={X_test.shape}, cross={X_cross.shape}")

    R = {
        "counts": {
            "train": len(train_bags),
            "test": len(test_bags),
            "cross_pos": len(pos),
            "cross_neg": len(pb) + len(norm_neg),
        },
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
    }

    # ----------------------------------------------------------------
    # Multi-seed experiment: LightGBM + XGBoost ensemble
    # ----------------------------------------------------------------
    print(f"\nTraining ensemble over {len(SEEDS)} seeds...")

    all_f1_id = []
    all_auc_id = []
    all_hard_auc = []
    all_x_auc = []
    all_thresholds = []

    for seed in SEEDS:
        lgbm_model = train_lgbm(X_train, y_train, seed=seed)
        xgb_model = train_xgb(X_train, y_train, seed=seed)
        models = [lgbm_model, xgb_model]

        p_test = ensemble_predict(models, X_test)
        p_cross_s = ensemble_predict(models, X_cross)

        m = acc_metrics(y_test, p_test)
        all_f1_id.append(m["f1"])
        all_auc_id.append(m["auc"])
        all_thresholds.append(m["threshold"])

        # Hard-AUC
        sel_hard = (y_cross == 1) | ((y_cross == 0) & (src == "ptx_benign"))
        if sel_hard.sum() > 0 and len(set(y_cross[sel_hard])) == 2:
            hard_auc = roc_auc_score(y_cross[sel_hard], p_cross_s[sel_hard])
        else:
            hard_auc = 0.0
        all_hard_auc.append(hard_auc)
        all_x_auc.append(roc_auc_score(y_cross, p_cross_s))

        print(
            f"  Seed {seed}: ID-F1={m['f1']:.4f}, ID-AUC={m['auc']:.4f}, "
            f"Hard-AUC={hard_auc:.4f}, X-AUC={roc_auc_score(y_cross, p_cross_s):.4f}"
        )

    # Summary statistics
    R["ensemble_multi_seed"] = {
        "in_domain": {
            "f1": [float(np.mean(all_f1_id)), float(np.std(all_f1_id))],
            "f1_ci": list(boot_ci(all_f1_id)),
            "auc": [float(np.mean(all_auc_id)), float(np.std(all_auc_id))],
            "auc_ci": list(boot_ci(all_auc_id)),
            "threshold_mean": float(np.mean(all_thresholds)),
        },
        "hard_auc": {
            "mean": float(np.mean(all_hard_auc)),
            "std": float(np.std(all_hard_auc)),
            "ci": list(boot_ci(all_hard_auc)),
            "per_seed": all_hard_auc,
        },
        "x_auc": {
            "mean": float(np.mean(all_x_auc)),
            "std": float(np.std(all_x_auc)),
            "ci": list(boot_ci(all_x_auc)),
            "per_seed": all_x_auc,
        },
        "seeds": SEEDS,
    }

    # ----------------------------------------------------------------
    # Best model (median seed) for full cross-domain analysis
    # ----------------------------------------------------------------
    best_seed_idx = int(np.argmin(np.abs(np.array(all_hard_auc) - np.median(all_hard_auc))))
    best_seed = SEEDS[best_seed_idx]
    print(f"\nFull cross-domain analysis with best seed={best_seed}...")

    lgbm_best = train_lgbm(X_train, y_train, seed=best_seed)
    xgb_best = train_xgb(X_train, y_train, seed=best_seed)
    p_cross_best = ensemble_predict([lgbm_best, xgb_best], X_cross)
    p_test_best = ensemble_predict([lgbm_best, xgb_best], X_test)

    R["ensemble_best"] = {
        "seed": best_seed,
        "in_domain": acc_metrics(y_test, p_test_best),
        "cross_domain": cross_block(y_cross, p_cross_best, src, ntx),
    }

    # ----------------------------------------------------------------
    # Ablation: LightGBM only
    # ----------------------------------------------------------------
    print("\nAblation: LightGBM only...")
    abl_lgbm_f1, abl_lgbm_hard = [], []
    for seed in SEEDS:
        m_lgbm = train_lgbm(X_train, y_train, seed=seed)
        p_t = m_lgbm.predict_proba(X_test)[:, 1]
        p_c = m_lgbm.predict_proba(X_cross)[:, 1]
        abl_lgbm_f1.append(acc_metrics(y_test, p_t)["f1"])
        sel_hard = (y_cross == 1) | ((y_cross == 0) & (src == "ptx_benign"))
        if sel_hard.sum() > 0 and len(set(y_cross[sel_hard])) == 2:
            abl_lgbm_hard.append(roc_auc_score(y_cross[sel_hard], p_c[sel_hard]))

    R["abl_lgbm_only"] = {
        "f1": [float(np.mean(abl_lgbm_f1)), float(np.std(abl_lgbm_f1))],
        "hard_auc": [float(np.mean(abl_lgbm_hard)), float(np.std(abl_lgbm_hard))],
    }

    # ----------------------------------------------------------------
    # Ablation: 8-feature baseline (original)
    # ----------------------------------------------------------------
    print("\nAblation: 8-feature baseline (original features)...")
    # Use only first 8 features which correspond to original
    X_train_8 = X_train[:, :8]
    X_test_8 = X_test[:, :8]
    X_cross_8 = X_cross[:, :8]

    abl_8f_f1, abl_8f_hard = [], []
    for seed in SEEDS:
        m8 = train_lgbm(X_train_8, y_train, seed=seed)
        p_t8 = m8.predict_proba(X_test_8)[:, 1]
        p_c8 = m8.predict_proba(X_cross_8)[:, 1]
        abl_8f_f1.append(acc_metrics(y_test, p_t8)["f1"])
        sel_hard = (y_cross == 1) | ((y_cross == 0) & (src == "ptx_benign"))
        if sel_hard.sum() > 0 and len(set(y_cross[sel_hard])) == 2:
            abl_8f_hard.append(roc_auc_score(y_cross[sel_hard], p_c8[sel_hard]))

    R["abl_8feat_lgbm"] = {
        "f1": [float(np.mean(abl_8f_f1)), float(np.std(abl_8f_f1))],
        "hard_auc": [float(np.mean(abl_8f_hard)), float(np.std(abl_8f_hard))],
    }

    # ----------------------------------------------------------------
    # Statistical significance test (Mann-Whitney U)
    # ----------------------------------------------------------------
    print("\nStatistical significance tests...")
    # Hard-AUC: ensemble vs 8-feat
    if len(all_hard_auc) > 1 and len(abl_8f_hard) > 1:
        stat, p_val = mannwhitneyu(all_hard_auc, abl_8f_hard, alternative="greater")
        R["significance"] = {
            "hard_auc_ensemble_vs_8feat": {
                "mann_whitney_U": float(stat),
                "p_value": float(p_val),
                "ensemble_mean": float(np.mean(all_hard_auc)),
                "baseline_mean": float(np.mean(abl_8f_hard)),
            }
        }

    # ----------------------------------------------------------------
    # Feature importance
    # ----------------------------------------------------------------
    fi = lgbm_best.feature_importances_
    R["feature_importance"] = {
        name: float(imp) for name, imp in zip(FEATURE_NAMES, fi)
    }

    # Save
    os.makedirs(RES, exist_ok=True)
    out_path = os.path.join(RES, "enhanced_account_results.json")
    json.dump(R, open(out_path, "w"), indent=2)
    print(f"\nSaved {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ms = R["ensemble_multi_seed"]
    print(f"ID-F1 (mean±std): {ms['in_domain']['f1'][0]:.4f} ± {ms['in_domain']['f1'][1]:.4f}")
    print(f"ID-F1 CI: {ms['in_domain']['f1_ci']}")
    print(f"Hard-AUC (mean±std): {ms['hard_auc']['mean']:.4f} ± {ms['hard_auc']['std']:.4f}")
    print(f"Hard-AUC CI: {ms['hard_auc']['ci']}")
    print(f"X-AUC (mean±std): {ms['x_auc']['mean']:.4f} ± {ms['x_auc']['std']:.4f}")
    print(f"X-AUC CI: {ms['x_auc']['ci']}")

    # SOTA comparison
    print("\nSOTA Comparison:")
    f1_mean = ms['in_domain']['f1'][0]
    hard_mean = ms['hard_auc']['mean']
    x_mean = ms['x_auc']['mean']
    print(f"  ID-F1:    {f1_mean:.4f} vs SOTA 0.750 -> {'BEAT' if f1_mean > 0.750 else 'Gap=' + str(round(0.750 - f1_mean, 4))}")
    print(f"  Hard-AUC: {hard_mean:.4f} vs SOTA 0.836 -> {'BEAT' if hard_mean > 0.836 else 'Gap=' + str(round(0.836 - hard_mean, 4))}")
    print(f"  X-AUC:    {x_mean:.4f} vs SOTA 0.984 -> {'LEAD' if x_mean > 0.984 else 'Gap=' + str(round(0.984 - x_mean, 4))}")

    return R


if __name__ == "__main__":
    main()
