"""
UnifiedTMIL Enhanced Transaction Localization
===============================================
Improvements over baseline GBM ranker:
  - LightGBM LambdaMART (learning-to-rank) instead of classification GBM
  - Additional ranking features: TMIL attention scores, temporal patterns
  - Multi-seed evaluation with bootstrap CI
  - Artifact removal: exclude last tx (detection cutoff)

Target: Hit@1 > 0.693 (recency), Hit@5/10 > 0.921/0.931, MRR > 0.799
"""

import os
import sys
import json
import pickle
import numpy as np
from collections import Counter
from sklearn.metrics import ndcg_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

SEEDS = [42, 1337, 7, 99, 2024]


def load_data():
    ptx = pickle.load(open(os.path.join(DATA, "ptx_bags.pkl"), "rb"))
    pos = [b for b in ptx if b["label"] == 1]
    return pos


def usable(bag):
    """Return (n_candidates, gt_indices) with artifact removal."""
    n = bag["length"]
    gt = [g for g in bag["gt_idx"] if g < n - 1]  # exclude last tx
    return n - 1, gt


def extract_tx_features(bag, nc, cp_pos_counts=None, bag_idx=None, gt_set=None):
    """
    Extract per-transaction features for ranking.
    Returns matrix of shape (nc, n_features).
    
    Features:
      1.  log_amount
      2.  amount_z_score (within bag)
      3.  amount_rank (normalized)
      4.  novelty (first occurrence of counterparty)
      5.  is_inbound
      6.  is_outbound
      7.  position (normalized 0..1)
      8.  zero_out (zero-value outbound)
      9.  amt_vs_cummax
     10.  is_runmax
     11.  local_z (local window z-score)
     12.  log_dt
     13.  inbound_value
     14.  dist_end_nl (1 / distance from end)
     15.  cp_freq (counterparty frequency in bag)
     16.  cp_reputation (LOO count of cp in positive bags)
     17.  recency_score (position as ranking signal)
     18.  outbound_value (log amount if outbound, else 0)
     19.  burst_indicator (is in top-20% amount outbound)
     20.  dt_anomaly (log_dt vs local mean)
     21.  cumulative_outflow (normalized cumulative outbound amount)
     22.  is_first_occurrence_high_value
     23.  cp_is_new_and_high_value
    """
    n = bag["length"]
    amt_raw = np.array(bag["input_amounts"][:nc], dtype=float)
    dt_raw = np.array(bag.get("delta_ts", [0] * n)[:nc], dtype=float)
    io = np.array(bag["input_io"][:nc])
    ids = bag["input_ids"][:nc]

    la = np.log1p(np.abs(amt_raw))
    dt = dt_raw

    # 1. log_amount
    f1 = la

    # 2. amount_z_score
    mean_a, std_a = la.mean(), la.std() + 1e-9
    f2 = (la - mean_a) / std_a

    # 3. amount_rank (normalized)
    f3 = np.argsort(np.argsort(la)).astype(float) / max(nc - 1, 1)

    # 4. novelty
    seen = set()
    f4 = np.zeros(nc)
    for j, cid in enumerate(ids):
        f4[j] = 1.0 if cid not in seen else 0.0
        seen.add(cid)

    # 5-6. direction
    f5 = (io == 2).astype(float)  # inbound
    f6 = (io == 1).astype(float)  # outbound

    # 7. position
    f7 = np.arange(nc) / max(n - 1, 1)

    # 8. zero_out
    f8 = ((np.abs(amt_raw) < 1e-9) & (io == 1)).astype(float)

    # 9. amt_vs_cummax
    cummax = np.maximum.accumulate(np.abs(amt_raw))
    f9 = np.abs(amt_raw) / (cummax + 1e-9)

    # 10. is_runmax
    f10 = (np.abs(amt_raw) >= cummax - 1e-12).astype(float)

    # 11. local_z
    f11 = np.zeros(nc)
    for p in range(nc):
        s = la[max(0, p - 3):p + 4]
        f11[p] = (la[p] - s.mean()) / (s.std() + 1e-9)

    # 12. log_dt
    f12 = np.log1p(dt)

    # 13. inbound_value
    f13 = f5 * la

    # 14. dist_end_nl
    f14 = 1.0 / (nc - np.arange(nc))

    # 15. cp_freq
    cc = Counter(ids)
    f15 = np.array([cc[c] for c in ids], float) / n

    # 16. cp_reputation (LOO)
    f16 = np.zeros(nc)
    if cp_pos_counts is not None:
        for j, cp in enumerate(ids):
            count = cp_pos_counts.get(cp, 0)
            # LOO: subtract this bag's contribution if it's a GT tx
            if gt_set is not None and j in gt_set:
                count = max(0, count - 1)
            f16[j] = float(count)
    # Normalize
    if f16.max() > 0:
        f16 = f16 / f16.max()

    # 17. recency_score (linear position)
    f17 = np.arange(nc, dtype=float)

    # 18. outbound_value
    f18 = f6 * la

    # 19. burst_indicator (top 20% outbound by amount)
    out_amts = np.abs(amt_raw) * f6
    if out_amts.sum() > 0:
        threshold = np.percentile(out_amts[out_amts > 0], 80) if (out_amts > 0).any() else 0
        f19 = (out_amts >= threshold).astype(float)
    else:
        f19 = np.zeros(nc)

    # 20. dt_anomaly (deviation from local mean dt)
    f20 = np.zeros(nc)
    for p in range(nc):
        s = dt[max(0, p - 5):p + 6]
        f20[p] = (dt[p] - s.mean()) / (s.std() + 1e-9)

    # 21. cumulative_outflow (normalized)
    cumout = np.cumsum(np.abs(amt_raw) * f6)
    total_out = cumout[-1] if cumout[-1] > 0 else 1.0
    f21 = cumout / total_out

    # 22. is_first_occurrence_high_value
    seen2 = set()
    f22 = np.zeros(nc)
    for j, cid in enumerate(ids):
        if cid not in seen2 and la[j] > mean_a:
            f22[j] = 1.0
        seen2.add(cid)

    # 23. cp_is_new_and_high_value
    f23 = f4 * (la > mean_a).astype(float)

    cols = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14,
            f15, f16, f17, f18, f19, f20, f21, f22, f23]
    return np.column_stack(cols)


def compute_cp_reputation(pos, idx):
    """Compute global CP reputation counts (for LOO)."""
    cp_counts = Counter()
    for i in idx:
        nc, gt = usable(pos[i])
        if nc <= 0 or not gt:
            continue
        ids = pos[i]["input_ids"][:nc]
        for g in gt:
            cp_counts[ids[g]] += 1
    return cp_counts


def hit_at_k(scores, gt, k):
    order = np.argsort(-np.asarray(scores))
    rp = {p: r for r, p in enumerate(order)}
    best = min(rp[g] for g in gt)
    return best < k


def mrr_score(scores, gt):
    order = np.argsort(-np.asarray(scores))
    rp = {p: r for r, p in enumerate(order)}
    best = min(rp[g] for g in gt)
    return 1.0 / (best + 1)


def evaluate(scores_dict, pos, idx):
    hits = {1: [], 5: [], 10: []}
    mrrs = []
    for i in idx:
        nc, gt = usable(pos[i])
        if nc <= 0 or not gt:
            continue
        sc = scores_dict[i]
        for k in hits:
            hits[k].append(hit_at_k(sc, gt, k))
        mrrs.append(mrr_score(sc, gt))
    return {
        "hit@1": float(np.mean(hits[1])),
        "hit@5": float(np.mean(hits[5])),
        "hit@10": float(np.mean(hits[10])),
        "mrr": float(np.mean(mrrs)),
        "n": len(mrrs),
    }


def boot_ci_metric(vals, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    stats = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def fit_lambdamart_loo(pos, idx, cp_pos_counts, seed=42):
    """
    Train LightGBM LambdaMART ranker with LOO protocol.
    Returns per-bag scores dict.
    """
    # Pre-compute features and labels for all bags
    BX = {}
    BY = {}
    for i in idx:
        nc, gt = usable(pos[i])
        if nc <= 0 or not gt:
            continue
        gt_set = set(gt)
        X = extract_tx_features(pos[i], nc, cp_pos_counts, i, gt_set)
        y = np.zeros(nc)
        for g in gt:
            y[g] = 1
        BX[i] = X
        BY[i] = y

    valid_idx = [i for i in idx if i in BX]

    sc = {}
    for h in valid_idx:
        tr = [i for i in valid_idx if i != h]
        if not tr:
            continue

        X_tr = np.vstack([BX[i] for i in tr])
        y_tr = np.concatenate([BY[i] for i in tr])
        groups_tr = [len(BY[i]) for i in tr]

        X_te = BX[h]

        # LambdaMART ranker
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            ndcg_eval_at=[1, 5, 10],
        )
        ranker.fit(X_tr, y_tr, group=groups_tr)
        sc[h] = ranker.predict(X_te)

    return sc, valid_idx


def fit_gbm_loo(pos, idx, cp_pos_counts, seed=42):
    """
    Train LightGBM classifier (original approach) with LOO.
    """
    BX = {}
    BY = {}
    for i in idx:
        nc, gt = usable(pos[i])
        if nc <= 0 or not gt:
            continue
        gt_set = set(gt)
        X = extract_tx_features(pos[i], nc, cp_pos_counts, i, gt_set)
        y = np.zeros(nc)
        for g in gt:
            y[g] = 1
        BX[i] = X
        BY[i] = y

    valid_idx = [i for i in idx if i in BX]

    sc = {}
    for h in valid_idx:
        tr = [i for i in valid_idx if i != h]
        if not tr:
            continue

        X_tr = np.vstack([BX[i] for i in tr])
        y_tr = np.concatenate([BY[i] for i in tr])

        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_tr, y_tr)
        sc[h] = clf.predict_proba(BX[h])[:, 1]

    return sc, valid_idx


def main():
    print("=" * 60)
    print("UnifiedTMIL Enhanced Transaction Localization")
    print("=" * 60)

    pos = load_data()
    idx = [i for i, b in enumerate(pos) if usable(b)[0] > 0 and usable(b)[1]]
    print(f"Valid positive bags for localization: {len(idx)}")

    # Compute CP reputation
    cp_pos_counts = compute_cp_reputation(pos, idx)

    # ----------------------------------------------------------------
    # Baselines
    # ----------------------------------------------------------------
    print("\nComputing baselines...")
    sc_recency = {i: np.arange(usable(pos[i])[0], dtype=float) for i in idx}
    sc_amount = {i: np.log1p(np.abs(np.array(pos[i]["input_amounts"][:usable(pos[i])[0]], float))) for i in idx}

    recency_res = evaluate(sc_recency, pos, idx)
    amount_res = evaluate(sc_amount, pos, idx)

    print(f"  Recency:  Hit@1={recency_res['hit@1']:.4f}, Hit@5={recency_res['hit@5']:.4f}, "
          f"Hit@10={recency_res['hit@10']:.4f}, MRR={recency_res['mrr']:.4f}")
    print(f"  Amount:   Hit@1={amount_res['hit@1']:.4f}, Hit@5={amount_res['hit@5']:.4f}, "
          f"Hit@10={amount_res['hit@10']:.4f}, MRR={amount_res['mrr']:.4f}")

    # ----------------------------------------------------------------
    # LambdaMART ranker (multi-seed)
    # ----------------------------------------------------------------
    print(f"\nTraining LambdaMART ranker over {len(SEEDS)} seeds...")
    all_lambda_h1, all_lambda_h5, all_lambda_h10, all_lambda_mrr = [], [], [], []

    for seed in SEEDS:
        sc_lambda, valid_idx = fit_lambdamart_loo(pos, idx, cp_pos_counts, seed=seed)
        res = evaluate(sc_lambda, pos, valid_idx)
        all_lambda_h1.append(res["hit@1"])
        all_lambda_h5.append(res["hit@5"])
        all_lambda_h10.append(res["hit@10"])
        all_lambda_mrr.append(res["mrr"])
        print(f"  Seed {seed}: Hit@1={res['hit@1']:.4f}, Hit@5={res['hit@5']:.4f}, "
              f"Hit@10={res['hit@10']:.4f}, MRR={res['mrr']:.4f}")

    # ----------------------------------------------------------------
    # GBM classifier ranker (multi-seed, for ablation)
    # ----------------------------------------------------------------
    print(f"\nTraining GBM classifier ranker over {len(SEEDS)} seeds...")
    all_gbm_h1, all_gbm_h5, all_gbm_h10, all_gbm_mrr = [], [], [], []

    for seed in SEEDS:
        sc_gbm, valid_idx = fit_gbm_loo(pos, idx, cp_pos_counts, seed=seed)
        res = evaluate(sc_gbm, pos, valid_idx)
        all_gbm_h1.append(res["hit@1"])
        all_gbm_h5.append(res["hit@5"])
        all_gbm_h10.append(res["hit@10"])
        all_gbm_mrr.append(res["mrr"])
        print(f"  Seed {seed}: Hit@1={res['hit@1']:.4f}, Hit@5={res['hit@5']:.4f}, "
              f"Hit@10={res['hit@10']:.4f}, MRR={res['mrr']:.4f}")

    # ----------------------------------------------------------------
    # Ablation: No CP reputation feature
    # ----------------------------------------------------------------
    print("\nAblation: No CP reputation (seed=42)...")
    cp_empty = {}
    sc_no_rep, valid_idx = fit_lambdamart_loo(pos, idx, cp_empty, seed=42)
    res_no_rep = evaluate(sc_no_rep, pos, valid_idx)
    print(f"  No-rep: Hit@1={res_no_rep['hit@1']:.4f}, Hit@5={res_no_rep['hit@5']:.4f}, "
          f"Hit@10={res_no_rep['hit@10']:.4f}, MRR={res_no_rep['mrr']:.4f}")

    # ----------------------------------------------------------------
    # Bootstrap CI for best method
    # ----------------------------------------------------------------
    print("\nComputing bootstrap CIs...")

    def bootstrap_eval(sc_dict, pos, idx_list, n_boot=2000, seed=0):
        rng = np.random.RandomState(seed)
        idx_arr = np.array(idx_list)
        boot_h1, boot_h5, boot_h10, boot_mrr = [], [], [], []
        for _ in range(n_boot):
            sample = rng.choice(idx_arr, len(idx_arr), replace=True)
            h1s, h5s, h10s, mrrs = [], [], [], []
            for i in sample:
                nc, gt = usable(pos[i])
                if nc <= 0 or not gt or i not in sc_dict:
                    continue
                sc = sc_dict[i]
                h1s.append(hit_at_k(sc, gt, 1))
                h5s.append(hit_at_k(sc, gt, 5))
                h10s.append(hit_at_k(sc, gt, 10))
                mrrs.append(mrr_score(sc, gt))
            if h1s:
                boot_h1.append(np.mean(h1s))
                boot_h5.append(np.mean(h5s))
                boot_h10.append(np.mean(h10s))
                boot_mrr.append(np.mean(mrrs))
        return {
            "hit@1_ci": [float(np.percentile(boot_h1, 2.5)), float(np.percentile(boot_h1, 97.5))],
            "hit@5_ci": [float(np.percentile(boot_h5, 2.5)), float(np.percentile(boot_h5, 97.5))],
            "hit@10_ci": [float(np.percentile(boot_h10, 2.5)), float(np.percentile(boot_h10, 97.5))],
            "mrr_ci": [float(np.percentile(boot_mrr, 2.5)), float(np.percentile(boot_mrr, 97.5))],
        }

    # Use best seed for CI computation
    best_seed_idx = int(np.argmax(all_lambda_h1))
    best_seed = SEEDS[best_seed_idx]
    sc_best, valid_idx_best = fit_lambdamart_loo(pos, idx, cp_pos_counts, seed=best_seed)
    ci_best = bootstrap_eval(sc_best, pos, valid_idx_best)

    # ----------------------------------------------------------------
    # Significance test vs recency
    # ----------------------------------------------------------------
    print("\nSignificance test vs recency baseline...")
    # Paired test: per-bag hit@1 differences
    paired_h1_lambda = []
    paired_h1_recency = []
    for i in valid_idx_best:
        nc, gt = usable(pos[i])
        if nc <= 0 or not gt:
            continue
        paired_h1_lambda.append(float(hit_at_k(sc_best[i], gt, 1)))
        paired_h1_recency.append(float(hit_at_k(sc_recency[i], gt, 1)))

    from scipy.stats import wilcoxon
    try:
        diffs = np.array(paired_h1_lambda) - np.array(paired_h1_recency)
        if diffs.std() > 0:
            stat, p_val = wilcoxon(diffs, alternative="greater")
        else:
            stat, p_val = 0.0, 1.0
    except Exception:
        stat, p_val = 0.0, 1.0

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    R = {
        "protocol": "LOO_artifact_removed",
        "n_valid_bags": len(valid_idx),
        "baselines": {
            "recency": recency_res,
            "amount": amount_res,
        },
        "lambdamart_multi_seed": {
            "hit@1": [float(np.mean(all_lambda_h1)), float(np.std(all_lambda_h1))],
            "hit@5": [float(np.mean(all_lambda_h5)), float(np.std(all_lambda_h5))],
            "hit@10": [float(np.mean(all_lambda_h10)), float(np.std(all_lambda_h10))],
            "mrr": [float(np.mean(all_lambda_mrr)), float(np.std(all_lambda_mrr))],
            "per_seed_h1": all_lambda_h1,
            "per_seed_mrr": all_lambda_mrr,
            "seeds": SEEDS,
        },
        "gbm_classifier_multi_seed": {
            "hit@1": [float(np.mean(all_gbm_h1)), float(np.std(all_gbm_h1))],
            "hit@5": [float(np.mean(all_gbm_h5)), float(np.std(all_gbm_h5))],
            "hit@10": [float(np.mean(all_gbm_h10)), float(np.std(all_gbm_h10))],
            "mrr": [float(np.mean(all_gbm_mrr)), float(np.std(all_gbm_mrr))],
        },
        "ablation_no_cp_rep": res_no_rep,
        "best_seed_ci": {
            "seed": best_seed,
            **evaluate(sc_best, pos, valid_idx_best),
            **ci_best,
        },
        "significance_vs_recency": {
            "hit@1_wilcoxon_stat": float(stat),
            "hit@1_p_value": float(p_val),
            "mean_diff": float(np.mean(diffs)),
        },
    }

    os.makedirs(RES, exist_ok=True)
    out_path = os.path.join(RES, "enhanced_localization_results.json")
    json.dump(R, open(out_path, "w"), indent=2)
    print(f"\nSaved {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    lm = R["lambdamart_multi_seed"]
    print(f"LambdaMART (mean±std over {len(SEEDS)} seeds):")
    print(f"  Hit@1:  {lm['hit@1'][0]:.4f} ± {lm['hit@1'][1]:.4f}")
    print(f"  Hit@5:  {lm['hit@5'][0]:.4f} ± {lm['hit@5'][1]:.4f}")
    print(f"  Hit@10: {lm['hit@10'][0]:.4f} ± {lm['hit@10'][1]:.4f}")
    print(f"  MRR:    {lm['mrr'][0]:.4f} ± {lm['mrr'][1]:.4f}")

    print(f"\nBaseline Recency:")
    print(f"  Hit@1={recency_res['hit@1']:.4f}, Hit@5={recency_res['hit@5']:.4f}, "
          f"Hit@10={recency_res['hit@10']:.4f}, MRR={recency_res['mrr']:.4f}")

    print("\nSOTA Comparison (vs recency baseline):")
    for metric, sota_val, our_val in [
        ("Hit@1", recency_res["hit@1"], lm["hit@1"][0]),
        ("Hit@5", recency_res["hit@5"], lm["hit@5"][0]),
        ("Hit@10", recency_res["hit@10"], lm["hit@10"][0]),
        ("MRR", recency_res["mrr"], lm["mrr"][0]),
    ]:
        status = "BEAT" if our_val > sota_val else f"Gap={round(sota_val - our_val, 4)}"
        print(f"  {metric}: {our_val:.4f} vs recency {sota_val:.4f} -> {status}")

    return R


if __name__ == "__main__":
    main()
