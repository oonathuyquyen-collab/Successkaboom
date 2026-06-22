"""
Lean SOTA - Transaction-Level Localization (FIXED)
Fixes data leakage in counterparty reputation by using a strict Train/Test split.
"""
import sys, os, json, pickle, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
from src import vocab_def
sys.modules['__main__'].Vocab = vocab_def.Vocab
from src.step4_final import DATA, RES

def extract_transaction_features(bag, nc):
    """Extract 15 explicit features (excluding reputation for now)."""
    amt = np.asarray(bag["input_amounts"][:nc], float)
    io = np.asarray(bag["input_io"][:nc], float)
    dt = np.asarray(bag.get("delta_ts", [0]*nc)[:nc], float)
    ids = list(bag["input_ids"][:nc])
    n = max(nc, 1)
    
    la = np.log1p(np.abs(amt))
    z = (la - la.mean()) / (la.std() + 1e-9)
    rank = np.argsort(np.argsort(amt)) / max(n-1, 1)
    
    seen = set()
    nov = []
    for c in ids:
        nov.append(1.0 if c not in seen else 0.0)
        seen.add(c)
    nov = np.array(nov)
    
    inb = (io == 2).astype(float)
    outb = (io == 1).astype(float)
    prel = np.arange(nc) / max(n-1, 1)
    zero_out = ((amt == 0) & (io == 1)).astype(float)
    
    cummax = np.maximum.accumulate(np.abs(amt))
    avc = np.abs(amt) / (cummax + 1e-9)
    runmax = (np.abs(amt) >= cummax - 1e-12).astype(float)
    
    lz = np.zeros(nc)
    for p in range(nc):
        s = la[max(0, p-3):p+4]
        lz[p] = (la[p] - s.mean()) / (s.std() + 1e-9)
        
    ldt = np.log1p(dt)
    invalue = inb * la
    dist_nl = 1.0 / (nc - np.arange(nc))
    
    cc = Counter(ids)
    cpf = np.array([cc[c] for c in ids], float) / n
    
    cols = [la, z, rank, nov, inb, outb, prel, zero_out, avc, runmax, lz, ldt, invalue, dist_nl, cpf]
    return np.column_stack(cols), ids

def usable(b):
    n = b["length"]
    gt = [g for g in b["gt_idx"] if g < n-1]
    return n-1, gt

def hit_at_k(ranks, gt, k):
    order = np.argsort(-np.asarray(ranks))
    rp = {p: r for r, p in enumerate(order)}
    best = min(rp[g] for g in gt)
    return best < k

def mrr(ranks, gt):
    order = np.argsort(-np.asarray(ranks))
    rp = {p: r for r, p in enumerate(order)}
    best = min(rp[g] for g in gt)
    return 1.0 / (best + 1)

def main():
    random.seed(42)
    np.random.seed(42)
    
    print("Loading data...")
    ptx = pickle.load(open(os.path.join(DATA, "ptx_bags.pkl"), "rb"))
    pos = [b for b in ptx if b["label"] == 1]
    
    idx = [i for i, b in enumerate(pos) if usable(b)[0] > 0 and usable(b)[1]]
    print(f"Total valid positive bags: {len(idx)}")
    
    # 1. Strict Train/Test Split (80/20)
    random.shuffle(idx)
    split = int(len(idx) * 0.8)
    train_idx = idx[:split]
    test_idx = idx[split:]
    print(f"Split: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # 2. Extract features and GT
    BX = {}; BY = {}; B_IDS = {}
    for i in idx:
        nc, gt = usable(pos[i])
        X, ids = extract_transaction_features(pos[i], nc)
        y = np.zeros(nc)
        for g in gt: y[g] = 1
        BX[i] = X; BY[i] = y; B_IDS[i] = ids
        
    # 3. Compute Reputation ONLY from Train set
    cp_pos_counts = Counter()
    for i in train_idx:
        ids = B_IDS[i]
        for j, is_gt in enumerate(BY[i]):
            if is_gt == 1:
                cp_pos_counts[ids[j]] += 1
    
    # 4. Add Reputation feature to both sets
    for i in idx:
        nc = len(BY[i])
        rep = np.zeros(nc)
        for j, cp in enumerate(B_IDS[i]):
            count = cp_pos_counts[cp]
            # LOO only for training set
            if i in train_idx and BY[i][j] == 1:
                count = max(0, count - 1)
            rep[j] = count
        BX[i] = np.column_stack([BX[i], rep])
        
    # 5. Train Model on Train set
    X_train = np.vstack([BX[i] for i in train_idx])
    Y_train = np.concatenate([BY[i] for i in train_idx])
    
    print("Training Fixed Lean GBM Ranker...")
    m = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, 
        subsample=0.9, random_state=0
    ).fit(X_train, Y_train)
    
    # 6. Evaluate on Test set
    def eval_set(indices):
        hits = {k: [] for k in [1, 5, 10]}
        mrrs = []
        for i in indices:
            sc = m.predict_proba(BX[i])[:, 1]
            _, gt = usable(pos[i])
            for k in hits:
                hits[k].append(hit_at_k(sc, gt, k))
            mrrs.append(mrr(sc, gt))
        return {
            "hit@1": float(np.mean(hits[1])),
            "hit@5": float(np.mean(hits[5])),
            "hit@10": float(np.mean(hits[10])),
            "mrr": float(np.mean(mrrs)),
            "n": len(indices)
        }
    
    res_test = eval_set(test_idx)
    res_train = eval_set(train_idx)
    
    # 7. Baselines on Test set
    # Recency
    rec_hits = {k: [] for k in [1, 5, 10]}; rec_mrrs = []
    for i in test_idx:
        sc = np.arange(len(BY[i]), dtype=float)
        _, gt = usable(pos[i])
        for k in rec_hits: rec_hits[k].append(hit_at_k(sc, gt, k))
        rec_mrrs.append(mrr(sc, gt))
    res_recency = {"hit@1": float(np.mean(rec_hits[1])), "hit@5": float(np.mean(rec_hits[5])), "mrr": float(np.mean(rec_mrrs))}

    RESULT = {
        "fixed_gbm_test": res_test,
        "fixed_gbm_train": res_train,
        "recency_test": res_recency
    }
    
    os.makedirs(RES, exist_ok=True)
    out_path = os.path.join(RES, "lean_loc_results_fixed.json")
    json.dump(RESULT, open(out_path, "w"), indent=2)
    print(f"Saved {out_path}")
    
    print("\n--- RESULTS (FIXED) ---")
    print(f"TRAIN (n={len(train_idx)}): Hit@1={res_train['hit@1']:.3f}, MRR={res_train['mrr']:.3f}")
    print(f"TEST  (n={len(test_idx)}): Hit@1={res_test['hit@1']:.3f}, MRR={res_test['mrr']:.3f}")
    print(f"RECENCY TEST       : Hit@1={res_recency['hit@1']:.3f}")

if __name__ == "__main__":
    main()
