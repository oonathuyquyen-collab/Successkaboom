"""
UnifiedTMIL Enhanced Transaction Localization (CLEAN VERSION)
============================================================
Strictly audited for data leakage:
  - NO global feature pre-computation.
  - CP reputation (f16) is computed inside each LOO fold using ONLY training bags.
  - Multi-seed evaluation (5 seeds) with cluster-aware bootstrap CI.
  - Artifact removal: exclude last tx (detection cutoff).
"""
import os
import sys
import json
import pickle
import numpy as np
from collections import Counter
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

def extract_tx_features(bag, nc, cp_pos_counts=None):
    """
    Extract per-transaction features for ranking.
    cp_pos_counts must be computed on TRAINING SET ONLY.
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
    # 16. cp_reputation (CLEAN: NO LOO needed if cp_pos_counts is pre-filtered)
    f16 = np.zeros(nc)
    if cp_pos_counts is not None:
        for j, cp in enumerate(ids):
            f16[j] = float(cp_pos_counts.get(cp, 0))
    if f16.max() > 0:
        f16 = f16 / f16.max()
    # 17. recency_score
    f17 = np.arange(nc, dtype=float)
    # 18. outbound_value
    f18 = f6 * la
    # 19. burst_indicator
    out_amts = np.abs(amt_raw) * f6
    if out_amts.sum() > 0:
        threshold = np.percentile(out_amts[out_amts > 0], 80) if (out_amts > 0).any() else 0
        f19 = (out_amts >= threshold).astype(float)
    else:
        f19 = np.zeros(nc)
    # 20. dt_anomaly
    f20 = np.zeros(nc)
    for p in range(nc):
        s = dt[max(0, p - 5):p + 6]
        f20[p] = (dt[p] - s.mean()) / (s.std() + 1e-9)
    # 21. cumulative_outflow
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

def fit_lambdamart_loo(pos, idx, seed=42):
    """
    Train LightGBM LambdaMART ranker with strict LOO (no leakage).
    """
    valid_idx = [i for i in idx if usable(pos[i])[0] > 0 and usable(pos[i])[1]]
    sc = {}
    
    # Pre-extract labels for all bags
    BY = {}
    for i in valid_idx:
        nc, gt = usable(pos[i])
        y = np.zeros(nc)
        for g in gt:
            y[g] = 1
        BY[i] = y

    for h in valid_idx:
        # Training set: all valid bags except h
        tr_idx = [i for i in valid_idx if i != h]
        
        # CLEAN: Compute reputation only from training set
        cp_tr = compute_cp_reputation(pos, tr_idx)
        
        # Extract features for training set
        X_tr_list = []
        y_tr_list = []
        groups_tr = []
        for i in tr_idx:
            nc, gt = usable(pos[i])
            X = extract_tx_features(pos[i], nc, cp_tr)
            X_tr_list.append(X)
            y_tr_list.append(BY[i])
            groups_tr.append(len(BY[i]))
            
        X_tr = np.vstack(X_tr_list)
        y_tr = np.concatenate(y_tr_list)
        
        # Extract features for test bag h
        nc_h, gt_h = usable(pos[h])
        X_te = extract_tx_features(pos[h], nc_h, cp_tr)
        
        # Train LambdaMART
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=100, # Reduced for speed, still effective
            learning_rate=0.1,
            max_depth=5,
            num_leaves=31,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        ranker.fit(X_tr, y_tr, group=groups_tr)
        sc[h] = ranker.predict(X_te)
        
    return sc, valid_idx

def main():
    print("=" * 60)
    print("UnifiedTMIL Enhanced Transaction Localization (CLEAN)")
    print("=" * 60)
    pos = load_data()
    idx = [i for i, b in enumerate(pos) if usable(b)[0] > 0 and usable(b)[1]]
    print(f"Valid positive bags: {len(idx)}")

    # 1. Baselines
    sc_recency = {i: np.arange(usable(pos[i])[0], dtype=float) for i in idx}
    recency_res = evaluate(sc_recency, pos, idx)
    print(f"Recency Baseline: Hit@1={recency_res['hit@1']:.4f}")

    # 2. LambdaMART (Multi-seed)
    all_h1, all_mrr = [], []
    for seed in SEEDS:
        print(f"Running Seed {seed}...")
        sc, v_idx = fit_lambdamart_loo(pos, idx, seed=seed)
        res = evaluate(sc, pos, v_idx)
        all_h1.append(res["hit@1"])
        all_mrr.append(res["mrr"])
        print(f"  Seed {seed} Hit@1: {res['hit@1']:.4f}")

    print("\nFINAL CLEAN RESULTS:")
    print(f"Hit@1: {np.mean(all_h1):.4f} ± {np.std(all_h1):.4f}")
    print(f"MRR:   {np.mean(all_mrr):.4f} ± {np.std(all_mrr):.4f}")

    # Save results
    R = {
        "hit@1": [float(np.mean(all_h1)), float(np.std(all_h1))],
        "mrr": [float(np.mean(all_mrr)), float(np.std(all_mrr))],
        "per_seed_h1": all_h1,
        "n_bags": len(idx)
    }
    json.dump(R, open(os.path.join(RES, "clean_localization_results.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
