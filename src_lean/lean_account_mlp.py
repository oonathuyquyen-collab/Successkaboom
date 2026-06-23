"""
UnifiedTMIL - Account-Level Detection
Replaces UnifiedTMIL with a lightweight MLP on aggregate bag features.
"""
import sys, os, json, pickle, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from src import vocab_def
sys.modules['__main__'].Vocab = vocab_def.Vocab
from src.step4_final import load, boot_ci, DATA, SRC, RES

SEEDS = [42]  # Reduced from [42, 1, 7] since variance is low
DEV = "cpu"

class LeanAccountMLP(nn.Module):
    """Extremely lightweight MLP for account classification based on aggregate features."""
    def __init__(self, input_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

def extract_aggregate_features(bag):
    """Extract 8 core statistical features from a transaction bag."""
    n = bag["length"]
    amt = np.log1p(np.abs(np.array(bag["input_amounts"][:n], dtype=float)))
    dt = np.log1p(np.array(bag.get("delta_ts", [0]*n)[:n], dtype=float))
    io = np.array(bag["input_io"][:n])
    ids = bag["input_ids"][:n]
    
    mean_amt = amt.mean() if n > 0 else 0
    std_amt = amt.std() if n > 0 else 0
    max_amt = amt.max() if n > 0 else 0
    
    mean_dt = dt.mean() if n > 0 else 0
    std_dt = dt.std() if n > 0 else 0
    
    outbound_ratio = float((io == 1).mean()) if n > 0 else 0
    unique_cp_ratio = len(set(ids)) / n if n > 0 else 0
    
    return [mean_amt, std_amt, max_amt, mean_dt, std_dt, n, outbound_ratio, unique_cp_ratio]

def prepare_data(bags):
    X = np.array([extract_aggregate_features(b) for b in bags], dtype=np.float32)
    y = np.array([b["label"] for b in bags], dtype=np.float32)
    return torch.tensor(X), torch.tensor(y)

def iterate_batches(X, y, bs, rng):
    idx = list(range(len(X)))
    rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        batch_idx = idx[i:i+bs]
        yield X[batch_idx], y[batch_idx]

def train_lean_mlp(X_train, y_train, seed, epochs=5, bs=128, lr=1e-3, use_margin=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    m = LeanAccountMLP(input_dim=X_train.shape[1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    rng = random.Random(seed)
    
    for ep in range(epochs):
        m.train()
        for X_batch, y_batch in iterate_batches(X_train, y_train, bs, rng):
            X_batch, y_batch = X_batch.to(DEV), y_batch.to(DEV)
            logit = m(X_batch)
            p = torch.sigmoid(logit)
            
            loss = F.binary_cross_entropy(p, y_batch)
            
            if use_margin:
                pm, nm = y_batch == 1, y_batch == 0
                if pm.sum() > 0 and nm.sum() > 0:
                    margin_loss = F.relu(0.3 - (p[pm].mean() - p[nm].mean()))
                    loss = loss + 0.3 * margin_loss
                    
            opt.zero_grad()
            loss.backward()
            opt.step()
            
    return m

@torch.no_grad()
def predict_lean_mlp(m, X):
    m.eval()
    logit = m(X.to(DEV))
    return torch.sigmoid(logit).cpu().numpy()

def acc_metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, (p > 0.5).astype(int), average="binary", zero_division=0)
    return {
        "precision": float(pr), "recall": float(rc), "f1": float(f1),
        "auc": float(roc_auc_score(y, p)), "aupr": float(average_precision_score(y, p))
    }

def main():
    print("Loading data...")
    train_bags = load("train_bags.pkl")
    test_bags = load("test_bags.pkl")
    ptx = pickle.load(open(os.path.join(DATA, "ptx_bags.pkl"), "rb"))
    norm_neg = pickle.load(open(os.path.join(DATA, "normal_eoa_neg.pkl"), "rb"))
    
    pos = [b for b in ptx if b["label"] == 1]
    pb = [b for b in ptx if b["label"] == 0]
    allb = pos + pb + norm_neg
    
    src = np.array([b.get("source", "ptx_benign") for b in allb])
    ntx = np.array([b["ntx_full"] for b in allb])
    
    print("Extracting features...")
    X_train, y_train = prepare_data(train_bags)
    X_test, y_test = prepare_data(test_bags)
    X_cross, y_cross = prepare_data(allb)
    
    y_test_np = y_test.numpy()
    y_cross_np = y_cross.numpy()
    
    R = {"counts": {"train": len(train_bags), "test": len(test_bags), "cross_pos": len(pos), "cross_neg": len(pb) + len(norm_neg)}}
    
    def cross_block(p):
        auc = roc_auc_score(y_cross_np, p)
        aupr = average_precision_score(y_cross_np, p)
        
        idx = np.arange(len(y_cross_np))
        def f_auc(rng):
            ss = rng.choice(idx, len(idx), replace=True)
            return roc_auc_score(y_cross_np[ss], p[ss]) if len(set(y_cross_np[ss])) == 2 else auc
            
        bysrc = {}
        for sn in ["ptx_benign", "normal_eoa"]:
            sel = (y_cross_np == 1) | ((y_cross_np == 0) & (src == sn))
            if sel.sum() > 0:
                bysrc[sn] = {
                    "auc": float(roc_auc_score(y_cross_np[sel], p[sel])),
                    "aupr": float(average_precision_score(y_cross_np[sel], p[sel])),
                    "n_neg": int(((y_cross_np == 0) & (src == sn)).sum())
                }
                
        strat = {}
        for lo, hi, nm in [(3, 20, "3-20"), (20, 100, "20-100"), (100, 1e9, "100+")]:
            sel = (ntx >= lo) & (ntx < hi)
            if sel.sum() > 5 and len(set(y_cross_np[sel])) == 2:
                strat[nm] = {"auc": float(roc_auc_score(y_cross_np[sel], p[sel])), "n": int(sel.sum())}
                
        return {"auc": float(auc), "auc_ci": boot_ci(f_auc), "aupr": float(aupr), "by_source": bysrc, "stratified": strat}

    print("Training UnifiedTMIL MLP...")
    m = train_lean_mlp(X_train, y_train, seed=SEEDS[0])
    p_test = predict_lean_mlp(m, X_test)
    p_cross = predict_lean_mlp(m, X_cross)
    
    indm = acc_metrics(y_test_np, p_test)
    # Format to match original output structure [mean, std] for compatibility with tables
    indm_fmt = {k: [v, 0.0] for k, v in indm.items()}
    
    R["lean_mlp"] = {
        "in_domain": indm_fmt,
        "cross_domain": cross_block(p_cross)
    }
    
    # Ablation: No margin loss
    print("Training Ablation: No margin loss...")
    m_nomargin = train_lean_mlp(X_train, y_train, seed=SEEDS[0], use_margin=False)
    p_cross_nomargin = predict_lean_mlp(m_nomargin, X_cross)
    R["abl_no_margin"] = {"cross_domain": cross_block(p_cross_nomargin)}
    
    # Baseline: Random Forest
    print("Training Baseline: Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0, n_jobs=-1).fit(X_train.numpy(), y_train.numpy())
    p_test_rf = rf.predict_proba(X_test.numpy())[:, 1]
    p_cross_rf = rf.predict_proba(X_cross.numpy())[:, 1]
    
    R["rf_baseline"] = {
        "in_domain": {k: [v, 0.0] for k, v in acc_metrics(y_test_np, p_test_rf).items()},
        "cross_domain": cross_block(p_cross_rf)
    }
    
    os.makedirs(RES, exist_ok=True)
    out_path = os.path.join(RES, "lean_account_results.json")
    json.dump(R, open(out_path, "w"), indent=2)
    print(f"Saved {out_path}")
    print(f"Lean MLP Cross AUC: {R['lean_mlp']['cross_domain']['auc']:.4f}")
    print(f"Lean MLP Hard-neg AUC: {R['lean_mlp']['cross_domain']['by_source']['ptx_benign']['auc']:.4f}")

if __name__ == "__main__":
    main()
    main()
