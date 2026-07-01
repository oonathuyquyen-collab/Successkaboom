"""
Ablation study for UnifiedTMIL:
  A. Contrastive loss weight: 0.0, 0.1, 0.3 (baseline), 0.5
  B. Backbone variants: full, no-TCN, no-counterparty-emb, no-IO-emb
  C. Loss lambda (entropy regularization): 0.0, 0.1, 0.25, 0.5

Saves results to results/ablation_results.json
"""
import sys, os, json, pickle, random, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "core"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
import vocab_def
sys.modules['__main__'].Vocab = vocab_def.Vocab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SRC  = os.path.join(DATA, "bert4eth")
RES  = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

# Use 2 seeds for ablation (faster)
SEEDS = [42, 7]
DEV = "cpu"

def load(name):
    return pickle.load(open(os.path.join(SRC, name), "rb"))

class IODirTMIL(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hc_dim=2, use_tcn=True,
                 direction_mode="io_embed", use_cp_embed=True):
        super().__init__()
        self.direction_mode = direction_mode
        self.use_cp_embed = use_cp_embed
        if use_cp_embed:
            self.cp_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        else:
            self.cp_embed = None
            self.cp_proj = nn.Linear(1, embed_dim)
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
        if self.use_cp_embed:
            h = self.cp_embed(ids) + self.hc_proj(hc)
        else:
            h = self.hc_proj(hc)
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

def train_and_eval(train_bags, test_bags, V, config, seed):
    """Train one configuration and return metrics."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = IODirTMIL(
        V,
        direction_mode=config.get("direction_mode", "io_embed"),
        use_tcn=config.get("use_tcn", True),
        use_cp_embed=config.get("use_cp_embed", True)
    ).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    rng = random.Random(seed)
    contrast_w = config.get("contrast_weight", 0.3)
    entropy_lambda = config.get("entropy_lambda", 0.0)
    epochs = config.get("epochs", 8)

    for ep in range(epochs):
        m.train()
        for batch in iterate(train_bags, 64, rng):
            ids, io, hc, mask, y = collate(batch)
            logit, a = m(ids, io, hc, mask)
            p = torch.sigmoid(logit)
            bce = F.binary_cross_entropy(p, y)
            contrast = torch.tensor(0.0)
            if contrast_w > 0:
                pm, nm = y == 1, y == 0
                if pm.sum() > 0 and nm.sum() > 0:
                    contrast = F.relu(0.3 - (p[pm].mean() - p[nm].mean()))
            # Entropy regularization (attention sparsity)
            entropy_loss = torch.tensor(0.0)
            if entropy_lambda > 0:
                eps = 1e-8
                entropy_loss = -torch.sum(a * torch.log(a + eps), dim=1).mean()
            loss = bce + contrast_w * contrast + entropy_lambda * entropy_loss
            opt.zero_grad(); loss.backward(); opt.step()

    m.eval()
    with torch.no_grad():
        y_test = np.array([b["label"] for b in test_bags])
        ps = []
        for i in range(0, len(test_bags), 128):
            batch = test_bags[i:i+128]
            ids, io, hc, mask, y = collate(batch)
            logit, _ = m(ids, io, hc, mask)
            ps.extend(torch.sigmoid(logit).tolist())
        ps = np.array(ps)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_test, (ps > 0.5).astype(int), average="binary", zero_division=0)
        auc  = roc_auc_score(y_test, ps)
        aupr = average_precision_score(y_test, ps)
    return {"f1": float(f1), "auc": float(auc), "aupr": float(aupr),
            "precision": float(pr), "recall": float(rc)}

def run_ablation(train_bags, test_bags, V, configs):
    """Run all ablation configs, return results dict."""
    results = {}
    for name, config in configs.items():
        print(f"  [{name}]", end="", flush=True)
        seed_results = []
        for seed in SEEDS:
            r = train_and_eval(train_bags, test_bags, V, config, seed)
            seed_results.append(r)
            print(f" seed{seed}:f1={r['f1']:.4f}", end="", flush=True)
        print()
        # Aggregate
        results[name] = {
            "f1":   {"mean": float(np.mean([r["f1"]   for r in seed_results])),
                     "std":  float(np.std( [r["f1"]   for r in seed_results])),
                     "per_seed": [r["f1"] for r in seed_results]},
            "auc":  {"mean": float(np.mean([r["auc"]  for r in seed_results])),
                     "std":  float(np.std( [r["auc"]  for r in seed_results])),
                     "per_seed": [r["auc"] for r in seed_results]},
            "aupr": {"mean": float(np.mean([r["aupr"] for r in seed_results])),
                     "std":  float(np.std( [r["aupr"] for r in seed_results])),
                     "per_seed": [r["aupr"] for r in seed_results]},
            "config": config
        }
    return results

def main():
    print("=" * 60)
    print("UnifiedTMIL Ablation Study")
    print("=" * 60)
    t0 = time.time()

    train_bags = load("train_bags.pkl")
    test_bags  = load("test_bags.pkl")
    vocab = pickle.load(open(os.path.join(SRC, "vocab.pkl"), "rb"))
    V = len(vocab.token2id)
    print(f"train={len(train_bags)} test={len(test_bags)} V={V} seeds={SEEDS}")

    # ============================================================
    # A. Contrastive loss weight ablation
    # ============================================================
    print("\n--- A. Contrastive Loss Weight Ablation ---")
    contrast_configs = {
        "contrast_0.0":  {"contrast_weight": 0.0, "direction_mode": "io_embed"},
        "contrast_0.1":  {"contrast_weight": 0.1, "direction_mode": "io_embed"},
        "contrast_0.3":  {"contrast_weight": 0.3, "direction_mode": "io_embed"},  # baseline
        "contrast_0.5":  {"contrast_weight": 0.5, "direction_mode": "io_embed"},
    }
    contrast_results = run_ablation(train_bags, test_bags, V, contrast_configs)

    # ============================================================
    # B. Backbone component ablation
    # ============================================================
    print("\n--- B. Backbone Component Ablation ---")
    backbone_configs = {
        "full_model":      {"contrast_weight": 0.3, "direction_mode": "io_embed", "use_tcn": True,  "use_cp_embed": True},
        "no_tcn":          {"contrast_weight": 0.3, "direction_mode": "io_embed", "use_tcn": False, "use_cp_embed": True},
        "no_cp_embed":     {"contrast_weight": 0.3, "direction_mode": "io_embed", "use_tcn": True,  "use_cp_embed": False},
        "no_io_embed":     {"contrast_weight": 0.3, "direction_mode": "none",     "use_tcn": True,  "use_cp_embed": True},
        "hardmask_io":     {"contrast_weight": 0.3, "direction_mode": "hardmask", "use_tcn": True,  "use_cp_embed": True},
    }
    backbone_results = run_ablation(train_bags, test_bags, V, backbone_configs)

    # ============================================================
    # C. Entropy regularization lambda ablation
    # ============================================================
    print("\n--- C. Entropy Regularization Lambda Ablation ---")
    entropy_configs = {
        "lambda_0.00":  {"contrast_weight": 0.3, "direction_mode": "io_embed", "entropy_lambda": 0.00},
        "lambda_0.10":  {"contrast_weight": 0.3, "direction_mode": "io_embed", "entropy_lambda": 0.10},
        "lambda_0.25":  {"contrast_weight": 0.3, "direction_mode": "io_embed", "entropy_lambda": 0.25},
        "lambda_0.50":  {"contrast_weight": 0.3, "direction_mode": "io_embed", "entropy_lambda": 0.50},
    }
    entropy_results = run_ablation(train_bags, test_bags, V, entropy_configs)

    # ============================================================
    # Save all results
    # ============================================================
    all_results = {
        "metadata": {
            "seeds": SEEDS, "epochs": 8, "device": DEV,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "position feature excluded from all configs (anti-leakage)"
        },
        "A_contrastive_loss": contrast_results,
        "B_backbone_ablation": backbone_results,
        "C_entropy_lambda": entropy_results
    }
    out_path = os.path.join(RES, "ablation_results.json")
    json.dump(all_results, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s")

    # ============================================================
    # Print summary tables
    # ============================================================
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)

    print("\nA. Contrastive Loss Weight (in-domain F1):")
    print(f"  {'Config':<20} {'F1 mean':>10} {'F1 std':>10} {'AUC mean':>10}")
    for name, r in contrast_results.items():
        print(f"  {name:<20} {r['f1']['mean']:>10.4f} {r['f1']['std']:>10.4f} {r['auc']['mean']:>10.4f}")

    print("\nB. Backbone Ablation (in-domain F1):")
    print(f"  {'Config':<20} {'F1 mean':>10} {'F1 std':>10} {'AUC mean':>10}")
    for name, r in backbone_results.items():
        print(f"  {name:<20} {r['f1']['mean']:>10.4f} {r['f1']['std']:>10.4f} {r['auc']['mean']:>10.4f}")

    print("\nC. Entropy Lambda (in-domain F1):")
    print(f"  {'Config':<20} {'F1 mean':>10} {'F1 std':>10} {'AUC mean':>10}")
    for name, r in entropy_results.items():
        print(f"  {name:<20} {r['f1']['mean']:>10.4f} {r['f1']['std']:>10.4f} {r['auc']['mean']:>10.4f}")

if __name__ == "__main__":
    main()
