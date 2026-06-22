"""
Lean SOTA - Transaction-Level Localization
Replaces Head-L with a robust Gradient Boosting Ranker using explicit features.
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
from src import vocab_def
sys.modules['__main__'].Vocab = vocab_def.Vocab
from src.step4_final import DATA, RES

def extract_transaction_features(bag, nc):
    """Extract 16 explicit features for each transaction in the bag (no attention)."""
    amt = np.asarray(bag["input_amounts"][:nc], float)
    io = np.asarray(bag["input_io"][:nc], float)
    dt = np.asarray(bag.get("delta_ts", [0]*nc)[:nc], float)
    ids = list(bag["input_ids"][:nc])
    n = max(nc, 1)
    
    # 1. log_amt
    la = np.log1p(np.abs(amt))
    # 2. amt_z
    z = (la - la.mean()) / (la.std() + 1e-9)
    # 3. amt_rank
    rank = np.argsort(np.argsort(amt)) / max(n-1, 1)
    
    # 4. novelty
    seen = set()
    nov = []
    for c in ids:
        nov.append(1.0 if c not in seen else 0.0)
        seen.add(c)
    nov = np.array(nov)
    
    # 5. inbound, 6. outbound
    inb = (io == 2).astype(float)
    outb = (io == 1).astype(float)
    
    # 7. position
    prel = np.arange(nc) / max(n-1, 1)
    
    # 8. zero_out
    zero_out = ((amt == 0) & (io == 1)).astype(float)
    
    # 9. amt_vs_cummax, 10. is_runmax
    cummax = np.maximum.accumulate(np.abs(amt))
    avc = np.abs(amt) / (cummax + 1e-9)
    runmax = (np.abs(amt) >= cummax - 1e-12).astype(float)
    
    # 11. local_z
    lz = np.zeros(nc)
    for p in range(nc):
        s = la[max(0, p-3):p+4]
        lz[p] = (la[p] - s.mean()) / (s.std() + 1e-9)
        
    # 12. log_dt
    ldt = np.log1p(dt)
    
    # 13. inbound_val
    invalue = inb * la
    
    # 14. dist_end_nl
    dist_nl = 1.0 / (nc - np.arange(nc))
    
    # 15. cp_freq
    cc = Counter(ids)
    cpf = np.array([cc[c] for c in ids], float) / n
    
    cols = [la, z, rank, nov, inb, outb, prel, zero_out, avc, runmax, lz, ldt, invalue, dist_nl, cpf]
    return np.column_stack(cols), ids

def usable(b):
    n = b["length"]
    # GT valid if not the last transaction (artifact removal)
    gt = [g for g in b["gt_idx"] if g < n-1]
    return n-1, gt

def compute_cp_reputation(pos_bags, idx_list):
    """Compute Leave-One-Out counterparty reputation."""
    # First, count total positive occurrences for each CP
    cp_pos_counts = Counter()
    for i in idx_list:
        nc, gt = usable(pos_bags[i])
        ids = pos_bags[i]["input_ids"][:nc]
        for g in gt:
            cp_pos_counts[ids[g]] += 1
            
    return cp_pos_counts

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
    print("Loading data...")
    ptx = pickle.load(open(os.path.join(DATA, "ptx_bags.pkl"), "rb"))
    pos = [b for b in ptx if b["label"] == 1]
    
    idx = [i for i, b in enumerate(pos) if usable(b)[0] > 0 and usable(b)[1]]
    print(f"Valid positive bags for localization: {len(idx)}")
    
    BX = {}
    BY = {}
    B_IDS = {}
    
    for i in idx:
        nc, gt = usable(pos[i])
        X, ids = extract_transaction_features(pos[i], nc)
        y = np.zeros(nc)
        for g in gt:
            y[g] = 1
        BX[i] = X
        BY[i] = y
        B_IDS[i] = ids
        
    cp_pos_counts = compute_cp_reputation(pos, idx)
    
    # 16. Add cp_reputation (LOO)
    for i in idx:
        nc = len(BY[i])
        rep = np.zeros(nc)
        for j, cp in enumerate(B_IDS[i]):
            # LOO: subtract this bag's contribution
            count = cp_pos_counts[cp]
            if BY[i][j] == 1:
                count -= 1
            rep[j] = count
        BX[i] = np.column_stack([BX[i], rep])
        
    def fit_loo(feature_indices=None):
        sc = {}
        for h in idx:
            tr = [i for i in idx if i != h]
            
            if feature_indices is not None:
                Xtr = np.vstack([BX[i][:, feature_indices] for i in tr])
                Xte = BX[h][:, feature_indices]
            else:
                Xtr = np.vstack([BX[i] for i in tr])
                Xte = BX[h]
                
            Ytr = np.concatenate([BY[i] for i in tr])
            
            m = GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05, 
                subsample=0.9, random_state=0
            ).fit(Xtr, Ytr)
            
            sc[h] = m.predict_proba(Xte)[:, 1]
        return sc

    print("Training Full Lean SOTA Ranker (LOO)...")
    sc_full = fit_loo()
    
    print("Training Ablation: No CP Reputation...")
    sc_no_rep = fit_loo(feature_indices=list(range(15))) # exclude 16th feature
    
    print("Evaluating...")
    RESULT = {}
    
    def eval_scores(sc_dict):
        hits = {k: [] for k in [1, 5, 10]}
        mrrs = []
        for i in idx:
            _, gt = usable(pos[i])
            for k in hits:
                hits[k].append(hit_at_k(sc_dict[i], gt, k))
            mrrs.append(mrr(sc_dict[i], gt))
        
        return {
            "hit@1": float(np.mean(hits[1])),
            "hit@5": float(np.mean(hits[5])),
            "hit@10": float(np.mean(hits[10])),
            "mrr": float(np.mean(mrrs)),
            "n": len(idx)
        }
        
    RESULT["lean_gbm_full"] = eval_scores(sc_full)
    RESULT["lean_gbm_no_rep"] = eval_scores(sc_no_rep)
    
    # Recency baseline
    sc_recency = {i: np.arange(len(BY[i]), dtype=float) for i in idx}
    RESULT["recency"] = eval_scores(sc_recency)
    
    os.makedirs(RES, exist_ok=True)
    out_path = os.path.join(RES, "lean_loc_results.json")
    json.dump(RESULT, open(out_path, "w"), indent=2)
    print(f"Saved {out_path}")
    
    for k, v in RESULT.items():
        print(f"{k:20s}: Hit@1={v['hit@1']:.3f}, Hit@5={v['hit@5']:.3f}, MRR={v['mrr']:.3f}")

if __name__ == "__main__":
    main()
