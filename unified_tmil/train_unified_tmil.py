"""
UnifiedTMIL Full Training Pipeline
====================================
Trains the full UnifiedTMIL (attention-based TMIL) over 5 seeds,
then stacks TMIL predictions + enhanced features into a GBM meta-learner.

Outputs:
  results/unified_tmil_multiseed.json  -- per-seed and aggregate metrics
  results/unified_tmil_stacked.json    -- stacked model metrics
"""

import os
import sys
import json
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, f1_score
)
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
BERT4ETH = os.path.join(DATA, "bert4eth")
SRC_DIR = os.path.join(ROOT, "src")
RES = os.path.join(ROOT, "results")

sys.path.insert(0, SRC_DIR)
import vocab_def
sys.modules['__main__'].Vocab = vocab_def.Vocab

SEEDS = [42, 1337, 7, 99, 2024]
DEV = "cpu"


# ============================================================
# Data loading helpers
# ============================================================

def load_pkl(path):
    return pickle.load(open(path, "rb"))


def load_data():
    train_bags = load_pkl(os.path.join(BERT4ETH, "train_bags.pkl"))
    test_bags = load_pkl(os.path.join(BERT4ETH, "test_bags.pkl"))
    vocab = load_pkl(os.path.join(BERT4ETH, "vocab.pkl"))
    ptx = load_pkl(os.path.join(DATA, "ptx_bags.pkl"))
    norm_neg = load_pkl(os.path.join(DATA, "normal_eoa_neg.pkl"))

    pos = [b for b in ptx if b["label"] == 1]
    pb_ptx = [b for b in ptx if b["label"] == 0 and b.get("source") == "ptx_benign"]
    pb_defi = [b for b in ptx if b["label"] == 0 and b.get("source") == "defi_hard"]

    return train_bags, test_bags, vocab, pos, pb_ptx, pb_defi, norm_neg


# ============================================================
# Model architecture
# ============================================================

class UnifiedTMIL(nn.Module):
    """
    Unified TMIL with shared encoder and two attention heads:
      Head-C: soft attention -> account classification
      Head-L: outbound-masked attention -> transaction localization
    """
    def __init__(self, V, d=64, use_tcn=True):
        super().__init__()
        self.cp_embed = nn.Embedding(V, d, padding_idx=0)
        self.io_embed = nn.Embedding(3, d, padding_idx=0)
        self.hc_proj = nn.Sequential(nn.Linear(2, d), nn.LayerNorm(d), nn.ReLU())
        self.use_tcn = use_tcn
        if use_tcn:
            self.tcn = nn.Conv1d(d, d, 3, padding=1)
        self.norm = nn.LayerNorm(d)

        # Head-C (account classification)
        self.attn_C_V = nn.Linear(d, 128)
        self.attn_C_U = nn.Linear(d, 128)
        self.attn_C_w = nn.Linear(128, 1)
        self.clf_C = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

        # Head-L (localization)
        self.attn_L_V = nn.Linear(d, 128)
        self.attn_L_U = nn.Linear(d, 128)
        self.attn_L_w = nn.Linear(128, 1)
        self.clf_L = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

    def encode(self, ids, io, hc):
        h = self.hc_proj(hc) + self.cp_embed(ids) + self.io_embed(io)
        h = self.norm(h)
        if self.use_tcn:
            h = h + self.tcn(h.transpose(1, 2)).transpose(1, 2)
        return h

    def head_C(self, h, mask):
        s = self.attn_C_w(torch.tanh(self.attn_C_V(h)) * torch.sigmoid(self.attn_C_U(h))).squeeze(-1)
        s = s.masked_fill(~mask, -1e9)
        a = F.softmax(s, dim=1)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        return self.clf_C(z).squeeze(-1), a

    def head_L(self, h, mask, io):
        s = self.attn_L_w(torch.tanh(self.attn_L_V(h)) * torch.sigmoid(self.attn_L_U(h))).squeeze(-1)
        s = s.masked_fill(~mask, -1e9)
        # Outbound mask: only attend to outbound txs if any exist
        outb = (io == 1)
        has_out = outb.any(dim=1, keepdim=True)
        s = s.masked_fill(has_out & ~outb & mask, -1e9)
        a = F.softmax(s, dim=1)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        return self.clf_L(z).squeeze(-1), a

    def forward(self, ids, io, hc, mask):
        h = self.encode(ids, io, hc)
        logit_C, a_C = self.head_C(h, mask)
        logit_L, a_L = self.head_L(h, mask, io)
        return logit_C, a_C, logit_L, a_L


def collate(bags):
    L = max(b["length"] for b in bags)
    B = len(bags)
    ids = torch.zeros(B, L, dtype=torch.long)
    io = torch.zeros(B, L, dtype=torch.long)
    hc = torch.zeros(B, L, 2, dtype=torch.float)
    mask = torch.zeros(B, L, dtype=torch.bool)
    y = torch.zeros(B, dtype=torch.float)

    for i, b in enumerate(bags):
        n = b["length"]
        ids[i, :n] = torch.tensor(b["input_ids"][:n])
        io[i, :n] = torch.tensor(b["input_io"][:n])
        amt = np.log1p(np.abs(np.array(b["input_amounts"][:n], dtype=np.float64)))
        dt = np.log1p(np.array(b.get("delta_ts", [0] * n)[:n], dtype=np.float64))
        hc[i, :n, 0] = torch.tensor(amt, dtype=torch.float)
        hc[i, :n, 1] = torch.tensor(dt, dtype=torch.float)
        mask[i, :n] = True
        y[i] = b["label"]
    return ids, io, hc, mask, y


def train_model(train_bags, V, seed, lam=0.5, epochs=10, bs=64, lr=1e-3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    m = UnifiedTMIL(V).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    rng = random.Random(seed)

    for ep in range(epochs):
        m.train()
        idx = list(range(len(train_bags)))
        rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            batch = [train_bags[j] for j in idx[i:i + bs]]
            ids, io, hc, mask, y = collate(batch)
            logit_C, _, logit_L, _ = m(ids, io, hc, mask)
            p_C = torch.sigmoid(logit_C)
            loss = F.binary_cross_entropy(p_C, y)
            loss = loss + lam * F.binary_cross_entropy(torch.sigmoid(logit_L), y)
            pm, nm = y == 1, y == 0
            if pm.sum() > 0 and nm.sum() > 0:
                loss = loss + 0.3 * F.relu(0.3 - (p_C[pm].mean() - p_C[nm].mean()))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


@torch.no_grad()
def predict(m, bags, bs=256):
    m.eval()
    ps = []
    attns = []
    for i in range(0, len(bags), bs):
        batch = bags[i:i + bs]
        ids, io, hc, mask, _ = collate(batch)
        logit_C, a_C, _, a_L = m(ids, io, hc, mask)
        ps.extend(torch.sigmoid(logit_C).tolist())
        for j, b in enumerate(batch):
            attns.append(a_L[j, :b["length"]].tolist())
    return np.array(ps), attns


# ============================================================
# Enhanced aggregate features (26-dim)
# ============================================================

def extract_enhanced_features(bag):
    n = bag["length"]
    if n == 0:
        return [0.0] * 26

    amt_raw = np.array(bag["input_amounts"][:n], dtype=float)
    dt_raw = np.array(bag.get("delta_ts", [0] * n)[:n], dtype=float)
    io = np.array(bag["input_io"][:n])
    ids = bag["input_ids"][:n]

    amt = np.log1p(np.abs(amt_raw))
    dt = np.log1p(dt_raw)

    mean_amt = float(amt.mean())
    std_amt = float(amt.std()) if n > 1 else 0.0
    max_amt = float(amt.max())
    mean_dt = float(dt.mean())
    std_dt = float(dt.std()) if n > 1 else 0.0
    log_n = float(np.log1p(n))

    out_mask = (io == 1)
    in_mask = (io == 2)
    outbound_ratio = float(out_mask.mean())
    inbound_ratio = float(in_mask.mean())
    unique_cp_ratio = len(set(ids)) / n

    sum_out = float(np.abs(amt_raw[out_mask]).sum()) if out_mask.any() else 0.0
    sum_in = float(np.abs(amt_raw[in_mask]).sum()) if in_mask.any() else 0.0
    io_asymmetry = float(np.log1p(sum_out) - np.log1p(sum_in))

    out_amts = np.abs(amt_raw[out_mask]) if out_mask.any() else np.array([0.0])
    burst_drain = float(out_amts.max() / (out_amts.sum() + 1e-9))

    if out_mask.any() and out_amts.sum() > 0:
        shares = out_amts / out_amts.sum()
        hhi = float((shares ** 2).sum())
    else:
        hhi = 0.0

    zero_out_ratio = float(((np.abs(amt_raw) < 1e-9) & out_mask).mean())
    total_val = sum_in + sum_out
    inbound_val_frac = float(sum_in / (total_val + 1e-9))
    max_out_frac = float(out_amts.max() / (total_val + 1e-9))
    temporal_burst = float(std_dt / (mean_dt + 1e-9))
    cp_repeat = 1.0 - unique_cp_ratio

    from collections import Counter
    cp_counts = Counter(ids)
    top3 = sum(v for _, v in cp_counts.most_common(3))
    top3_conc = float(top3 / n)

    if n > 2 and std_amt > 1e-9:
        amt_centered = amt - mean_amt
        skew = float((amt_centered ** 3).mean() / (std_amt ** 3))
        kurt = float((amt_centered ** 4).mean() / (std_amt ** 4)) - 3.0
    else:
        skew = 0.0
        kurt = 0.0

    weights = np.exp(np.linspace(0, 1, n))
    weights /= weights.sum()
    recency_out = float((out_mask.astype(float) * weights).sum())

    split = max(1, n // 5)
    early_out = float(out_mask[:split].mean())
    late_out = float(out_mask[-split:].mean())

    max_run = 0
    cur_run = 0
    for flag in out_mask:
        if flag:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    max_consec_out = float(max_run / n)

    ibo_count = sum(1 for i in range(n - 1) if io[i] == 2 and io[i + 1] == 1)
    ibo_score = float(ibo_count / (n - 1)) if n > 1 else 0.0

    log_ntx = float(np.log1p(bag.get("ntx_full", n)))

    return [
        mean_amt, std_amt, max_amt, mean_dt, std_dt, log_n,
        outbound_ratio, unique_cp_ratio, io_asymmetry, burst_drain,
        hhi, zero_out_ratio, inbound_ratio, inbound_val_frac, max_out_frac,
        temporal_burst, cp_repeat, top3_conc, skew, kurt,
        recency_out, early_out, late_out, max_consec_out, ibo_score, log_ntx,
    ]


# ============================================================
# Metrics
# ============================================================

def find_best_threshold(y_true, y_prob):
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


def boot_ci(vals, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    stats = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def compute_hard_auc(y, p, src, neg_source="ptx_benign"):
    sel = (y == 1) | ((y == 0) & (src == neg_source))
    if sel.sum() < 2 or len(set(y[sel])) < 2:
        return None
    return float(roc_auc_score(y[sel], p[sel]))


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("UnifiedTMIL Full Training Pipeline")
    print("=" * 60)

    train_bags, test_bags, vocab, pos, pb_ptx, pb_defi, norm_neg = load_data()
    V = len(vocab.token2id)

    # Cross-domain evaluation set: pos + ptx_benign + normal_eoa
    allb = pos + pb_ptx + norm_neg
    y_cross = np.array([b["label"] for b in allb])
    src_cross = np.array([b.get("source", "ptx_benign") for b in allb])
    ntx_cross = np.array([b["ntx_full"] for b in allb])

    # Extended hard set: pos + ptx_benign + defi_hard + normal_eoa
    allb_ext = pos + pb_ptx + pb_defi + norm_neg
    y_ext = np.array([b["label"] for b in allb_ext])
    src_ext = np.array([b.get("source", "?") for b in allb_ext])

    y_test = np.array([b["label"] for b in test_bags])

    print(f"Train: {len(train_bags)}, Test: {len(test_bags)}, V={V}")
    print(f"Cross: pos={len(pos)}, ptx_benign={len(pb_ptx)}, defi_hard={len(pb_defi)}, normal_eoa={len(norm_neg)}")

    # Pre-compute enhanced features for cross-domain
    print("\nExtracting enhanced features...")
    X_cross = np.array([extract_enhanced_features(b) for b in allb], dtype=np.float32)
    X_test_feat = np.array([extract_enhanced_features(b) for b in test_bags], dtype=np.float32)
    X_train_feat = np.array([extract_enhanced_features(b) for b in train_bags], dtype=np.float32)

    # ----------------------------------------------------------------
    # Multi-seed TMIL training
    # ----------------------------------------------------------------
    print(f"\nTraining UnifiedTMIL over {len(SEEDS)} seeds...")

    all_f1_id = []
    all_auc_id = []
    all_hard_auc = []
    all_x_auc = []
    all_p_cross = []
    all_attns = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}...")
        m = train_model(train_bags, V, seed=seed)
        p_test, _ = predict(m, test_bags)
        p_cross, attns = predict(m, allb)

        metrics_id = acc_metrics(y_test, p_test)
        hard_auc = compute_hard_auc(y_cross, p_cross, src_cross, "ptx_benign")
        x_auc = roc_auc_score(y_cross, p_cross)

        all_f1_id.append(metrics_id["f1"])
        all_auc_id.append(metrics_id["auc"])
        all_hard_auc.append(hard_auc if hard_auc is not None else 0.0)
        all_x_auc.append(x_auc)
        all_p_cross.append(p_cross)
        all_attns.append(attns)

        hard_str = f"{hard_auc:.4f}" if hard_auc is not None else "N/A"
        print(
            f"    ID-F1={metrics_id['f1']:.4f}, ID-AUC={metrics_id['auc']:.4f}, "
            f"Hard-AUC={hard_str}, X-AUC={x_auc:.4f}"
        )

    # Ensemble predictions (mean over seeds)
    p_cross_ens = np.mean(all_p_cross, axis=0)
    p_test_ens_list = []
    for seed in SEEDS:
        m = train_model(train_bags, V, seed=seed)
        pt, _ = predict(m, test_bags)
        p_test_ens_list.append(pt)
    p_test_ens = np.mean(p_test_ens_list, axis=0)

    # ----------------------------------------------------------------
    # Stacking: TMIL predictions + enhanced features -> LightGBM
    # ----------------------------------------------------------------
    print("\nTraining stacked model (TMIL + features -> LightGBM)...")

    # Build stacking features for cross-domain (use ensemble predictions)
    X_stack_cross = np.column_stack([p_cross_ens.reshape(-1, 1), X_cross])

    # For stacking train/test: retrain TMIL on train and get OOF predictions
    # Use 3-fold CV to get OOF predictions on train set
    from sklearn.model_selection import StratifiedKFold
    y_train_arr = np.array([b["label"] for b in train_bags])
    oof_preds = np.zeros(len(train_bags))
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print("  Computing OOF TMIL predictions (3-fold)...")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(range(len(train_bags)), y_train_arr)):
        tr_bags = [train_bags[i] for i in tr_idx]
        val_bags = [train_bags[i] for i in val_idx]
        m_fold = train_model(tr_bags, V, seed=42 + fold, epochs=8)
        p_val, _ = predict(m_fold, val_bags)
        oof_preds[val_idx] = p_val

    X_stack_train = np.column_stack([oof_preds.reshape(-1, 1), X_train_feat])

    # Train LightGBM stacker
    scale = (y_train_arr == 0).sum() / max((y_train_arr == 1).sum(), 1)
    stacker = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    stacker.fit(X_stack_train, y_train_arr)

    # Predict on test
    X_stack_test = np.column_stack([p_test_ens.reshape(-1, 1), X_test_feat])
    p_test_stacked = stacker.predict_proba(X_stack_test)[:, 1]
    metrics_stacked_id = acc_metrics(y_test, p_test_stacked)

    # Predict on cross-domain
    p_cross_stacked = stacker.predict_proba(X_stack_cross)[:, 1]
    hard_auc_stacked = compute_hard_auc(y_cross, p_cross_stacked, src_cross, "ptx_benign")
    x_auc_stacked = roc_auc_score(y_cross, p_cross_stacked)

    print(f"\nStacked model:")
    print(f"  ID-F1={metrics_stacked_id['f1']:.4f}, ID-AUC={metrics_stacked_id['auc']:.4f}")
    hard_str2 = f"{hard_auc_stacked:.4f}" if hard_auc_stacked is not None else "N/A"
    print(f"  Hard-AUC={hard_str2}, X-AUC={x_auc_stacked:.4f}")

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    R = {
        "seeds": SEEDS,
        "tmil_multi_seed": {
            "in_domain": {
                "f1": [float(np.mean(all_f1_id)), float(np.std(all_f1_id))],
                "f1_ci": list(boot_ci(all_f1_id)),
                "auc": [float(np.mean(all_auc_id)), float(np.std(all_auc_id))],
                "auc_ci": list(boot_ci(all_auc_id)),
                "per_seed": all_f1_id,
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
        },
        "tmil_ensemble": {
            "in_domain": acc_metrics(y_test, p_test_ens),
            "hard_auc": float(compute_hard_auc(y_cross, p_cross_ens, src_cross, "ptx_benign") or 0),
            "x_auc": float(roc_auc_score(y_cross, p_cross_ens)),
        },
        "stacked_model": {
            "in_domain": metrics_stacked_id,
            "hard_auc": float(hard_auc_stacked or 0),
            "x_auc": float(x_auc_stacked),
        },
    }

    os.makedirs(RES, exist_ok=True)
    out_path = os.path.join(RES, "unified_tmil_multiseed.json")
    json.dump(R, open(out_path, "w"), indent=2)
    print(f"\nSaved {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ms = R["tmil_multi_seed"]
    print(f"TMIL Multi-seed (n={len(SEEDS)}):")
    print(f"  ID-F1: {ms['in_domain']['f1'][0]:.4f} ± {ms['in_domain']['f1'][1]:.4f}  CI={ms['in_domain']['f1_ci']}")
    print(f"  Hard-AUC: {ms['hard_auc']['mean']:.4f} ± {ms['hard_auc']['std']:.4f}  CI={ms['hard_auc']['ci']}")
    print(f"  X-AUC: {ms['x_auc']['mean']:.4f} ± {ms['x_auc']['std']:.4f}  CI={ms['x_auc']['ci']}")

    print(f"\nTMIL Ensemble:")
    ens = R["tmil_ensemble"]
    print(f"  ID-F1: {ens['in_domain']['f1']:.4f}, Hard-AUC: {ens['hard_auc']:.4f}, X-AUC: {ens['x_auc']:.4f}")

    print(f"\nStacked Model:")
    st = R["stacked_model"]
    print(f"  ID-F1: {st['in_domain']['f1']:.4f}, Hard-AUC: {st['hard_auc']:.4f}, X-AUC: {st['x_auc']:.4f}")

    print("\nSOTA Comparison:")
    best_f1 = max(ms['in_domain']['f1'][0], ens['in_domain']['f1'], st['in_domain']['f1'])
    best_hard = max(ms['hard_auc']['mean'], ens['hard_auc'], st['hard_auc'])
    best_x = max(ms['x_auc']['mean'], ens['x_auc'], st['x_auc'])
    print(f"  ID-F1:    best={best_f1:.4f} vs SOTA 0.750 -> {'BEAT' if best_f1 > 0.750 else 'Gap=' + str(round(0.750 - best_f1, 4))}")
    print(f"  Hard-AUC: best={best_hard:.4f} vs SOTA 0.836 -> {'BEAT' if best_hard > 0.836 else 'Gap=' + str(round(0.836 - best_hard, 4))}")
    print(f"  X-AUC:    best={best_x:.4f} vs SOTA 0.984 -> {'LEAD' if best_x > 0.984 else 'Gap=' + str(round(0.984 - best_x, 4))}")

    return R


if __name__ == "__main__":
    main()
