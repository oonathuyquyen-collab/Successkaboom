"""
Extract attention weights from trained models and run localization ranker.
Produces:
  - results/unified_attn.json (baseline io_embed attention)
  - results/unified_attn_strong.json (hardmask attention as "strong")
  - results/loc_fusion_marginal.json (LOO GBM localization results)
"""
import sys, os, json, pickle, random, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "core"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier
import vocab_def
sys.modules['__main__'].Vocab = vocab_def.Vocab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SRC  = os.path.join(DATA, "bert4eth")
RES  = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

SEEDS = [42, 1, 7]
DEV = "cpu"

def load(name):
    return pickle.load(open(os.path.join(SRC, name), "rb"))

class IODirTMIL(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hc_dim=2, use_tcn=True, direction_mode="io_embed"):
        super().__init__()
        self.direction_mode = direction_mode
        self.cp_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.io_embed = nn.Embedding(3, embed_dim, padding_idx=0)
        self.hc_proj = nn.Sequential(nn.Linear(hc_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.use_tcn = use_tcn
        if use_tcn:
            self.tcn = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_V = nn.Linear(embed_dim, 128)
        self.attn_U = nn.Linear(embed_dim, 128)
        self.attn_w = nn.Linear(128, 1)
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, ids, io, hc, mask):
        h = self.cp_embed(ids) + self.hc_proj(hc)
        if self.direction_mode == "io_embed":
            h = h + self.io_embed(io)
        h = self.norm(h)
        if self.use_tcn:
            h = h + self.tcn(h.transpose(1, 2)).transpose(1, 2)
        s = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))).squeeze(-1)
        s = s.masked_fill(~mask, -1e9)
        if self.direction_mode == "hardmask":
            outb = (io == 1)
            has = outb.any(dim=1, keepdim=True)
            s = s.masked_fill(has & ~outb & mask, -1e9)
        a = F.softmax(s, dim=1)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        return self.classifier(z).squeeze(-1), a

def collate(bags):
    L = max(b["length"] for b in bags)
    ids = torch.zeros(len(bags), L, dtype=torch.long)
    io  = torch.zeros(len(bags), L, dtype=torch.long)
    hc  = torch.zeros(len(bags), L, 2)
    mask = torch.zeros(len(bags), L, dtype=torch.bool)
    y   = torch.zeros(len(bags))
    for i, b in enumerate(bags):
        n = b["length"]
        ids[i, :n] = torch.tensor(b["input_ids"][:n])
        io[i, :n]  = torch.tensor(b["input_io"][:n])
        amt = np.array(b["input_amounts"][:n], dtype=np.float32)
        dt  = np.array(b["delta_ts"][:n], dtype=np.float32)
        hc[i, :n, 0] = torch.tensor(np.log1p(np.abs(amt)) * np.sign(amt))
        hc[i, :n, 1] = torch.tensor(np.log1p(dt))
        mask[i, :n] = True
        y[i] = b["label"]
    return ids.to(DEV), io.to(DEV), hc.to(DEV), mask.to(DEV), y.to(DEV)

def iterate(bags, bs, rng):
    idx = list(range(len(bags))); rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [bags[j] for j in idx[i:i+bs]]

def train_model(train_bags, V, mode, seed, epochs=8, bs=64, lr=1e-3):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = IODirTMIL(V, direction_mode=mode).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    rng = random.Random(seed)
    for ep in range(epochs):
        m.train()
        for batch in iterate(train_bags, bs, rng):
            ids, io, hc, mask, y = collate(batch)
            logit, _ = m(ids, io, hc, mask)
            p = torch.sigmoid(logit)
            bce = F.binary_cross_entropy(p, y)
            pm, nm = y == 1, y == 0
            contrast = torch.tensor(0.0)
            if pm.sum() > 0 and nm.sum() > 0:
                contrast = F.relu(0.3 - (p[pm].mean() - p[nm].mean()))
            loss = bce + 0.3 * contrast
            opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def get_attention(m, bags, bs=128):
    m.eval(); attns = []
    for i in range(0, len(bags), bs):
        batch = bags[i:i+bs]
        ids, io, hc, mask, y = collate(batch)
        _, a = m(ids, io, hc, mask)
        for j, b in enumerate(batch):
            attns.append(a[j, :b["length"]].tolist())
    return attns

def extract_attention(mode, ptx_pos, train_bags, V):
    """Train multiple seeds and average attention weights."""
    all_attns = None
    for seed in SEEDS:
        m = train_model(train_bags, V, mode, seed)
        attns = get_attention(m, ptx_pos)
        if all_attns is None:
            all_attns = [np.array(a) for a in attns]
        else:
            all_attns = [x + np.array(a) for x, a in zip(all_attns, attns)]
        print(f"  seed {seed} done")
    # Average over seeds
    avg_attns = [a / len(SEEDS) for a in all_attns]
    return [a.tolist() for a in avg_attns]

# ============================================================
# Localization ranker (GBM LOO)
# ============================================================
def usable(b):
    n = b["length"]
    return n - 1, [g for g in b["gt_idx"] if g < n - 1]

def base_feats(b, nc, at, use_attn):
    amt = np.asarray(b["input_amounts"][:nc], float)
    io  = np.asarray(b["input_io"][:nc], float)
    dt  = np.asarray(b.get("delta_ts", [0]*nc)[:nc], float)
    ids = list(b["input_ids"][:nc])
    n   = max(nc, 1)
    la  = np.log1p(np.abs(amt))
    z   = (la - la.mean()) / (la.std() + 1e-9)
    rank = np.argsort(np.argsort(amt)) / max(n - 1, 1)
    seen = set(); nov = []
    for c in ids:
        nov.append(1.0 if c not in seen else 0.0)
        seen.add(c)
    nov = np.array(nov)
    inb  = (io == 2).astype(float)
    outb = (io == 1).astype(float)
    # NOTE: 'prel' (position/relative index) is intentionally EXCLUDED to avoid data leakage
    # as per prompt requirement 2.3
    dist_nl  = 1.0 / (nc - np.arange(nc))
    zero_out = ((amt == 0) & (io == 1)).astype(float)
    cummax   = np.maximum.accumulate(np.abs(amt))
    avc      = np.abs(amt) / (cummax + 1e-9)
    runmax   = (np.abs(amt) >= cummax - 1e-12).astype(float)
    lz = np.zeros(nc)
    for p in range(nc):
        s = la[max(0, p-3):p+4]
        lz[p] = (la[p] - s.mean()) / (s.std() + 1e-9)
    ldt     = np.log1p(dt)
    invalue = inb * la
    from collections import Counter
    cc  = Counter(ids)
    cpf = np.array([cc[c] for c in ids], float) / n
    # Base features WITHOUT position
    cols = [la, z, rank, nov, inb]
    if use_attn:
        a  = at[:nc] if len(at) >= nc else np.zeros(nc)
        ar = np.argsort(np.argsort(a)) / max(n - 1, 1)
        cols += [a, ar]
    cols += [zero_out, outb, avc, runmax, lz, ldt, invalue, dist_nl, cpf]
    return np.column_stack(cols)

def run_localization(pos, ATTN, mode):
    """mode in {baseline, strong, noattn}"""
    use_attn = mode != "noattn"
    at = ATTN["strong"] if mode == "strong" else ATTN["baseline"]
    idx = [i for i, b in enumerate(pos) if usable(b)[0] > 0 and usable(b)[1]]
    BX = {}; BY = {}; BN = {}; BIDS = {}
    for i in idx:
        nc, gt = usable(pos[i])
        BX[i]   = base_feats(pos[i], nc, at[i], use_attn)
        y       = np.zeros(nc)
        for g in gt: y[g] = 1
        BY[i]   = y; BN[i] = nc
        BIDS[i] = np.asarray(pos[i]["input_ids"][:nc])

    def cp_reputation(train_ids, hold_ids, alpha=5.0):
        from collections import defaultdict
        g = defaultdict(float); t = defaultdict(float); G = 0.0; T = 0.0
        for i in train_ids:
            ids = BIDS[i]; y = BY[i]
            for c, yy in zip(ids, y):
                t[c] += 1; g[c] += yy; T += 1; G += yy
        prior = G / max(T, 1)
        return np.array([(g[c] + alpha * prior) / (t[c] + alpha) for c in hold_ids])

    def br(s, gt):
        o  = np.argsort(-np.asarray(s))
        rp = {p: r for r, p in enumerate(o)}
        return min(rp[g] for g in gt)

    from collections import defaultdict
    def rep_loo_train(tr, alpha=5.0):
        g = defaultdict(float); t = defaultdict(float); G = 0.0; T = 0.0
        for i in tr:
            for c, yy in zip(BIDS[i], BY[i]):
                t[c] += 1; g[c] += yy; T += 1; G += yy
        out = {}
        for i in tr:
            gi = defaultdict(float); ti = defaultdict(float); Gi = G; Ti = T
            for c, yy in zip(BIDS[i], BY[i]):
                gi[c] += yy; ti[c] += 1; Gi -= yy; Ti -= 1
            prior = Gi / max(Ti, 1)
            out[i] = np.array([(g[c] - gi[c] + alpha * prior) / ((t[c] - ti[c]) + alpha) for c in BIDS[i]])
        return out

    sc = {}
    for h in idx:
        tr = [i for i in idx if i != h]
        reptr = rep_loo_train(tr)
        Xtr = np.vstack([np.column_stack([BX[i], reptr[i]]) for i in tr])
        Ytr = np.concatenate([BY[i] for i in tr])
        Xh  = np.column_stack([BX[h], cp_reputation(tr, BIDS[h])])
        m   = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                         subsample=0.9, random_state=0).fit(Xtr, Ytr)
        sc[h] = m.predict_proba(Xh)[:, 1]

    R   = {i: br(sc[i], usable(pos[i])[1]) for i in idx}
    arr = np.array(idx)
    res = {**{f"hit@{k}": float(np.mean([R[i] < k for i in idx])) for k in (1, 5, 10)},
           "mrr": float(np.mean([1.0 / (R[i] + 1) for i in idx])),
           "n": len(idx)}
    return res, R, arr

def paired(Ra, Rb, arr, k, nboot=8000, seed=7):
    rs = np.random.RandomState(seed); d = []
    for _ in range(nboot):
        bs = rs.choice(arr, len(arr), True)
        if k == "mrr":
            d.append(np.mean([1.0/(Ra[i]+1) for i in bs]) - np.mean([1.0/(Rb[i]+1) for i in bs]))
        else:
            d.append(np.mean([Ra[i] < k for i in bs]) - np.mean([Rb[i] < k for i in bs]))
    d = np.array(d)
    return {"mean_diff": float(d.mean()), "p_not_better": float(np.mean(d <= 0)),
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

def main():
    print("=" * 60)
    print("Step 1: Extract attention weights from trained models")
    print("=" * 60)
    train_bags = load("train_bags.pkl")
    vocab = pickle.load(open(os.path.join(SRC, "vocab.pkl"), "rb"))
    V = len(vocab.token2id)
    ptx = pickle.load(open(os.path.join(DATA, "ptx_bags.pkl"), "rb"))
    pos = [b for b in ptx if b["label"] == 1]
    print(f"Extracting baseline (io_embed) attention for {len(pos)} phishing bags...")
    attn_baseline = extract_attention("io_embed", pos, train_bags, V)
    json.dump(attn_baseline, open(os.path.join(RES, "unified_attn.json"), "w"))
    print(f"Saved: results/unified_attn.json")

    print(f"\nExtracting strong (hardmask) attention for {len(pos)} phishing bags...")
    attn_strong = extract_attention("hardmask", pos, train_bags, V)
    json.dump(attn_strong, open(os.path.join(RES, "unified_attn_strong.json"), "w"))
    print(f"Saved: results/unified_attn_strong.json")

    print("\n" + "=" * 60)
    print("Step 2: Run LOO GBM localization (3 variants)")
    print("=" * 60)
    ATTN = {
        "baseline": [np.array(a, float) for a in attn_baseline],
        "strong":   [np.array(a, float) for a in attn_strong]
    }

    print("Running fusion + baseline attn...")
    RES_B, RB, idx = run_localization(pos, ATTN, "baseline")
    print(f"  Hit@1={RES_B['hit@1']:.4f} Hit@5={RES_B['hit@5']:.4f} MRR={RES_B['mrr']:.4f}")

    print("Running fusion + strong attn...")
    RES_S, RS, _   = run_localization(pos, ATTN, "strong")
    print(f"  Hit@1={RES_S['hit@1']:.4f} Hit@5={RES_S['hit@5']:.4f} MRR={RES_S['mrr']:.4f}")

    print("Running fusion + no attn...")
    RES_N, RN, _   = run_localization(pos, ATTN, "noattn")
    print(f"  Hit@1={RES_N['hit@1']:.4f} Hit@5={RES_N['hit@5']:.4f} MRR={RES_N['mrr']:.4f}")

    print("\n=== PAIRED BOOTSTRAP TESTS ===")
    arr = np.array(idx)
    PSN = {f"hit@{k}": paired(RS, RN, arr, k) for k in (1, 5, 10)}
    PSN["mrr"] = paired(RS, RN, arr, "mrr")
    PSB = {f"hit@{k}": paired(RS, RB, arr, k) for k in (1, 5, 10)}
    PSB["mrr"] = paired(RS, RB, arr, "mrr")
    for k in ("hit@1", "hit@5", "hit@10", "mrr"):
        print(f"  strong vs noattn  {k}: diff={PSN[k]['mean_diff']:.4f} p_not_better={PSN[k]['p_not_better']:.3f}")
    for k in ("hit@1", "hit@5", "hit@10", "mrr"):
        print(f"  strong vs baseline {k}: diff={PSB[k]['mean_diff']:.4f} p_not_better={PSB[k]['p_not_better']:.3f}")

    out = {
        "fusion_baseline_attn": RES_B,
        "fusion_strong_attn":   RES_S,
        "fusion_no_attn":       RES_N,
        "paired_strong_vs_noattn":   PSN,
        "paired_strong_vs_baseline": PSB,
        "metadata": {
            "protocol": "LOO GBM with sum-minus-i cp_reputation, position feature EXCLUDED",
            "seeds": SEEDS, "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    json.dump(out, open(os.path.join(RES, "loc_fusion_marginal.json"), "w"), indent=2)
    print(f"\nSaved: results/loc_fusion_marginal.json")

    print("\n" + "=" * 60)
    print("LOCALIZATION SUMMARY")
    print("=" * 60)
    print(f"Content-aware + baseline attn: Hit@1={RES_B['hit@1']:.4f} Hit@5={RES_B['hit@5']:.4f} Hit@10={RES_B['hit@10']:.4f} MRR={RES_B['mrr']:.4f}")
    print(f"Content-aware + strong attn:   Hit@1={RES_S['hit@1']:.4f} Hit@5={RES_S['hit@5']:.4f} Hit@10={RES_S['hit@10']:.4f} MRR={RES_S['mrr']:.4f}")
    print(f"Content-aware + no attn:       Hit@1={RES_N['hit@1']:.4f} Hit@5={RES_N['hit@5']:.4f} Hit@10={RES_N['hit@10']:.4f} MRR={RES_N['mrr']:.4f}")

if __name__ == "__main__":
    main()
