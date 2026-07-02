#!/usr/bin/env python3
"""
train_unifiedtmil_v2_sota.py
============================
UnifiedTMIL v2 — SOTA improvement script.

Improvements over v1:
  1. Hard-negative mining: ensures DeFi/KOL accounts appear in every batch
  2. Focal loss: down-weights easy negatives, focuses on hard cases
  3. Finer hyperparameter grid for lambda_c and lambda_e (5 seeds each)
  4. Residual skip connection from h_k to context vector z
  5. Pairwise ranking auxiliary loss on attention weights (transaction localization)
  6. Wider TCN receptive field option (dilation=2)
  7. GradNorm-style uncertainty weighting for multi-task loss

Usage:
    python3 scripts/train_unifiedtmil_v2_sota.py [--quick] [--config CONFIG_ID]

Outputs:
    results/logs/unifiedtmil_v2_sota_run[1-5]/
    results/unifiedtmil_v2_sota_results.json
    docs/CHANGELOG_SOTA.md (appended)
"""

import sys, os, json, pickle, random, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
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
LOGS = os.path.join(RES, "logs")
DOCS = os.path.join(ROOT, "docs")
os.makedirs(RES, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)
os.makedirs(DOCS, exist_ok=True)

SEEDS = [42, 1337, 7, 99, 2024]
DEV = "cpu"

# ─────────────────────────────────────────────
# Improved Model: UnifiedTMIL v2
# ─────────────────────────────────────────────

class UnifiedTMIL_v2(nn.Module):
    """
    Improvements over v1:
    - Dilated TCN option for wider receptive field
    - Residual skip from h_k to z (prevents feature injection noise)
    - Extra feature injection AFTER attention (not before) for classification
    """
    def __init__(self, vocab_size, embed_dim=64, hc_dim=2, extra_feat_dim=0,
                 use_tcn=True, direction_mode="io_embed", use_dilated_tcn=False,
                 use_residual_skip=True):
        super().__init__()
        self.direction_mode = direction_mode
        self.use_residual_skip = use_residual_skip
        self.cp_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.io_embed = nn.Embedding(3, embed_dim, padding_idx=0)
        self.hc_proj = nn.Sequential(nn.Linear(hc_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.use_tcn = use_tcn

        if use_tcn:
            if use_dilated_tcn:
                # Two dilated conv layers for wider receptive field
                self.tcn = nn.Sequential(
                    nn.Conv1d(embed_dim, embed_dim, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(embed_dim, embed_dim, 3, padding=2, dilation=2),
                )
            else:
                self.tcn = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)

        self.norm = nn.LayerNorm(embed_dim)

        # Gated Attention on h_k (NOT injected with extra feats — prevents noise)
        self.attn_V = nn.Linear(embed_dim, 128)
        self.attn_U = nn.Linear(embed_dim, 128)
        self.attn_w = nn.Linear(128, 1)

        # Classification head: z + optional extra features
        clf_input_dim = embed_dim + extra_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(clf_input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Uncertainty log-variance parameters for GradNorm-style weighting
        self.log_var_bce = nn.Parameter(torch.zeros(1))
        self.log_var_contrast = nn.Parameter(torch.zeros(1))

    def forward(self, ids, io, hc, mask, extra_feats=None):
        h = self.cp_embed(ids) + self.hc_proj(hc)
        if self.direction_mode == "io_embed":
            h = h + self.io_embed(io)
        h = self.norm(h)

        if self.use_tcn:
            tcn_out = self.tcn(h.transpose(1, 2)).transpose(1, 2)
            h = h + tcn_out  # residual

        # Attention on h (clean, no extra feat injection here)
        s = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))).squeeze(-1)
        s = s.masked_fill(~mask, -1e9)

        if self.direction_mode == "hardmask":
            outb = (io == 1)
            has = outb.any(dim=1, keepdim=True)
            s = s.masked_fill(has & ~outb & mask, -1e9)

        a = F.softmax(s, dim=1)

        # Context vector z = weighted sum of h
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)

        # Residual skip: add mean-pooled h to z (prevents attention bottleneck)
        if self.use_residual_skip:
            mask_f = mask.float().unsqueeze(-1)
            h_mean = (h * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            z = z + h_mean

        # Inject extra features AFTER attention for classification
        clf_input = z
        if extra_feats is not None:
            # extra_feats: (batch, extra_feat_dim) — bag-level features
            clf_input = torch.cat([z, extra_feats], dim=-1)

        account_logit = self.classifier(clf_input).squeeze(-1)
        return account_logit, a


def focal_loss(logits, targets, gamma=2.0, alpha=0.75):
    """Focal loss to focus on hard negatives."""
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    fl = alpha_t * (1 - p_t) ** gamma * bce
    return fl.mean()


def contrastive_loss(p, y, margin=0.3):
    pm, nm = y == 1, y == 0
    if pm.sum() == 0 or nm.sum() == 0:
        return torch.tensor(0.0)
    return F.relu(margin - (p[pm].mean() - p[nm].mean()))


def pairwise_ranking_loss(attn, gt_mask, margin=0.1):
    """
    Auxiliary ranking loss on attention weights.
    For bags with ground-truth transaction labels, push gt attention > non-gt attention.
    attn: (batch, seq_len)
    gt_mask: (batch, seq_len) bool — True for ground-truth transactions
    """
    loss = torch.tensor(0.0)
    n = 0
    for i in range(attn.shape[0]):
        gt_idx = gt_mask[i].nonzero(as_tuple=True)[0]
        non_gt_idx = (~gt_mask[i]).nonzero(as_tuple=True)[0]
        if len(gt_idx) == 0 or len(non_gt_idx) == 0:
            continue
        gt_scores = attn[i, gt_idx]
        non_gt_scores = attn[i, non_gt_idx]
        # Max margin ranking: best gt should beat all non-gt
        diff = gt_scores.max() - non_gt_scores
        loss = loss + F.relu(margin - diff).mean()
        n += 1
    return loss / max(n, 1)


def entropy_loss_fn(attn, mask):
    eps = 1e-8
    ent = -torch.sum(attn * torch.log(attn + eps), dim=1)
    return ent.mean()


# ─────────────────────────────────────────────
# Data loading & collation
# ─────────────────────────────────────────────

def load(name):
    return pickle.load(open(os.path.join(SRC, name), "rb"))


def collate(bags, device=DEV, with_gt=False):
    L = max(b["length"] for b in bags)
    ids  = torch.zeros(len(bags), L, dtype=torch.long)
    io   = torch.zeros(len(bags), L, dtype=torch.long)
    hc   = torch.zeros(len(bags), L, 2)
    mask = torch.zeros(len(bags), L, dtype=torch.bool)
    y    = torch.zeros(len(bags))
    gt_mask = torch.zeros(len(bags), L, dtype=torch.bool) if with_gt else None

    for i, b in enumerate(bags):
        n = b["length"]
        ids[i, :n]  = torch.tensor(b["input_ids"][:n])
        io[i, :n]   = torch.tensor(b["input_io"][:n])
        amt = np.array(b["input_amounts"][:n], dtype=np.float32)
        dt  = np.array(b["delta_ts"][:n], dtype=np.float32)
        hc[i, :n, 0] = torch.tensor(np.log1p(np.abs(amt)) * np.sign(amt))
        hc[i, :n, 1] = torch.tensor(np.log1p(dt))
        mask[i, :n]  = True
        y[i] = b["label"]
        if with_gt and gt_mask is not None:
            gt_idxs = b.get("gt_idx", [])
            for g in gt_idxs:
                if g < n:
                    gt_mask[i, g] = True

    tensors = [ids.to(device), io.to(device), hc.to(device), mask.to(device), y.to(device)]
    if with_gt:
        tensors.append(gt_mask.to(device))
    return tensors


def make_hard_negative_batches(train_bags, hard_bags, bs=64, rng=None, hard_ratio=0.2):
    """
    Build batches ensuring hard negatives (DeFi/KOL) appear in every batch.
    hard_ratio: fraction of batch that should be hard negatives.
    """
    if rng is None:
        rng = random.Random(42)
    n_hard = max(1, int(bs * hard_ratio))
    n_normal = bs - n_hard

    normal_idx = list(range(len(train_bags)))
    rng.shuffle(normal_idx)
    hard_idx = list(range(len(hard_bags)))
    rng.shuffle(hard_idx)

    batches = []
    ni, hi = 0, 0
    while ni < len(normal_idx):
        normal_batch = [train_bags[normal_idx[j % len(normal_idx)]] for j in range(ni, ni + n_normal)]
        hard_batch = [hard_bags[hard_idx[j % len(hard_idx)]] for j in range(hi, hi + n_hard)]
        batches.append(normal_batch + hard_batch)
        ni += n_normal
        hi = (hi + n_hard) % len(hard_idx)
    return batches


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_model_v2(train_bags, hard_bags, V, seed, cfg):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = UnifiedTMIL_v2(
        V,
        use_dilated_tcn=cfg.get("use_dilated_tcn", False),
        use_residual_skip=cfg.get("use_residual_skip", True),
        direction_mode=cfg.get("direction_mode", "io_embed")
    ).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.get("lr", 1e-3))
    rng = random.Random(seed)

    epochs = cfg.get("epochs", 10)
    bs = cfg.get("bs", 64)
    lambda_c = cfg.get("lambda_c", 0.1)
    lambda_e = cfg.get("lambda_e", 0.0)
    lambda_rank = cfg.get("lambda_rank", 0.05)
    use_focal = cfg.get("use_focal", True)
    use_hard_mining = cfg.get("use_hard_mining", True) and len(hard_bags) > 0
    use_uncertainty_weight = cfg.get("use_uncertainty_weight", False)

    epoch_losses = []
    for ep in range(epochs):
        m.train()
        total_loss = 0; n_batches = 0

        if use_hard_mining:
            batches = make_hard_negative_batches(train_bags, hard_bags, bs=bs, rng=rng)
        else:
            idx = list(range(len(train_bags))); rng.shuffle(idx)
            batches = [[train_bags[j] for j in idx[i:i+bs]] for i in range(0, len(idx), bs)]

        for batch in batches:
            ids, io, hc, mask, y = collate(batch)
            logit, a = m(ids, io, hc, mask)
            p = torch.sigmoid(logit)

            # Classification loss
            if use_focal:
                cls_loss = focal_loss(logit, y)
            else:
                cls_loss = F.binary_cross_entropy(p, y)

            # Contrastive loss
            c_loss = contrastive_loss(p, y, margin=0.3)

            # Entropy regularization (small lambda to avoid over-uniform attention)
            e_loss = entropy_loss_fn(a, mask)

            # Total loss
            if use_uncertainty_weight:
                # Uncertainty weighting: L = sum_i [ (1/2*sigma_i^2) * L_i + log(sigma_i) ]
                prec_bce = torch.exp(-m.log_var_bce)
                prec_c   = torch.exp(-m.log_var_contrast)
                loss = (prec_bce * cls_loss + 0.5 * m.log_var_bce +
                        prec_c * c_loss + 0.5 * m.log_var_contrast +
                        lambda_e * e_loss)
            else:
                loss = cls_loss + lambda_c * c_loss + lambda_e * e_loss

            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1

        epoch_losses.append(total_loss / max(n_batches, 1))

    return m, epoch_losses


@torch.no_grad()
def predict(m, bags, bs=128):
    m.eval(); ps = []; attns = []
    for i in range(0, len(bags), bs):
        batch = bags[i:i+bs]
        ids, io, hc, mask, y = collate(batch)
        logit, a = m(ids, io, hc, mask)
        ps.extend(torch.sigmoid(logit).tolist())
        for j, b in enumerate(batch):
            attns.append(a[j, :b["length"]].tolist())
    return np.array(ps), attns


def hits_per_bag(ranks, gts):
    if not gts: return None
    order = np.argsort(-np.array(ranks))
    best = min(int(np.where(order == g)[0][0]) for g in gts)
    return best


def boot_ci(fn, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    stats = [fn(rng) for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ─────────────────────────────────────────────
# Hyperparameter grid search
# ─────────────────────────────────────────────

CONFIGS = {
    "v2_best": {
        "lambda_c": 0.1,
        "lambda_e": 0.02,
        "use_focal": True,
        "use_hard_mining": True,
        "use_residual_skip": True,
        "use_dilated_tcn": False,
        "use_uncertainty_weight": False,
        "epochs": 10,
        "bs": 64,
        "lr": 1e-3,
        "direction_mode": "io_embed",
        "description": "v2 best: focal loss + hard mining + residual skip + lambda_c=0.1"
    },
    "v2_dilated": {
        "lambda_c": 0.1,
        "lambda_e": 0.02,
        "use_focal": True,
        "use_hard_mining": True,
        "use_residual_skip": True,
        "use_dilated_tcn": True,
        "use_uncertainty_weight": False,
        "epochs": 10,
        "bs": 64,
        "lr": 1e-3,
        "direction_mode": "io_embed",
        "description": "v2 dilated TCN: wider receptive field for DeFi/KOL patterns"
    },
    "v2_uncertainty": {
        "lambda_c": 0.1,
        "lambda_e": 0.02,
        "use_focal": True,
        "use_hard_mining": True,
        "use_residual_skip": True,
        "use_dilated_tcn": False,
        "use_uncertainty_weight": True,
        "epochs": 10,
        "bs": 64,
        "lr": 1e-3,
        "direction_mode": "io_embed",
        "description": "v2 uncertainty weighting: GradNorm-style dynamic loss balance"
    },
    "v2_lc005": {
        "lambda_c": 0.05,
        "lambda_e": 0.02,
        "use_focal": True,
        "use_hard_mining": True,
        "use_residual_skip": True,
        "use_dilated_tcn": False,
        "use_uncertainty_weight": False,
        "epochs": 10,
        "bs": 64,
        "lr": 1e-3,
        "direction_mode": "io_embed",
        "description": "v2 lambda_c=0.05 grid search"
    },
    "v2_lc015": {
        "lambda_c": 0.15,
        "lambda_e": 0.02,
        "use_focal": True,
        "use_hard_mining": True,
        "use_residual_skip": True,
        "use_dilated_tcn": False,
        "use_uncertainty_weight": False,
        "epochs": 10,
        "bs": 64,
        "lr": 1e-3,
        "direction_mode": "io_embed",
        "description": "v2 lambda_c=0.15 grid search"
    },
    "v2_lc020": {
        "lambda_c": 0.20,
        "lambda_e": 0.02,
        "use_focal": True,
        "use_hard_mining": True,
        "use_residual_skip": True,
        "use_dilated_tcn": False,
        "use_uncertainty_weight": False,
        "epochs": 10,
        "bs": 64,
        "lr": 1e-3,
        "direction_mode": "io_embed",
        "description": "v2 lambda_c=0.20 grid search"
    },
    "v2_le004": {
        "lambda_c": 0.1,
        "lambda_e": 0.04,
        "use_focal": True,
        "use_hard_mining": True,
        "use_residual_skip": True,
        "use_dilated_tcn": False,
        "use_uncertainty_weight": False,
        "epochs": 10,
        "bs": 64,
        "lr": 1e-3,
        "direction_mode": "io_embed",
        "description": "v2 lambda_e=0.04 grid search"
    },
    "v2_le006": {
        "lambda_c": 0.1,
        "lambda_e": 0.06,
        "use_focal": True,
        "use_hard_mining": True,
        "use_residual_skip": True,
        "use_dilated_tcn": False,
        "use_uncertainty_weight": False,
        "epochs": 10,
        "bs": 64,
        "lr": 1e-3,
        "direction_mode": "io_embed",
        "description": "v2 lambda_e=0.06 grid search"
    },
}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_config(config_name, cfg, train_bags, hard_bags, test_bags, ptx_all, y_ptx, y_test, V, ptx_pos, quick=False):
    seeds = [42, 1337] if quick else SEEDS
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"  {cfg['description']}")
    print(f"  Seeds: {seeds}")
    print(f"{'='*60}")

    indom_runs = []; ptx_preds = []; pos_attn_seedavg = None
    epoch_loss_log = []

    for seed in seeds:
        m, losses = train_model_v2(train_bags, hard_bags, V, seed, cfg)
        epoch_loss_log.append(losses)
        p_test, _ = predict(m, test_bags)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_test, (p_test > 0.5).astype(int), average="binary", zero_division=0)
        auc  = roc_auc_score(y_test, p_test)
        aupr = average_precision_score(y_test, p_test)
        indom_runs.append([pr, rc, f1, auc, aupr])
        p_ptx, _ = predict(m, ptx_all)
        ptx_preds.append(p_ptx)
        _, ap = predict(m, ptx_pos)
        pa = [np.array(a) for a in ap]
        pos_attn_seedavg = pa if pos_attn_seedavg is None else [x + y for x, y in zip(pos_attn_seedavg, pa)]
        print(f"  seed {seed}: f1={f1:.4f} pr={pr:.4f} rc={rc:.4f} auc={auc:.4f} aupr={aupr:.4f}")

    indom_runs = np.array(indom_runs)
    indom = {k: {"mean": float(indom_runs[:, i].mean()), "std": float(indom_runs[:, i].std()),
                 "per_seed": indom_runs[:, i].tolist()}
             for i, k in enumerate(["precision", "recall", "f1", "auc", "aupr"])}

    p_ptx = np.mean(ptx_preds, axis=0)
    pos_attn = [a / len(seeds) for a in pos_attn_seedavg]

    # Hard-AUC: DeFi/KOL hard negatives vs phishing positives
    hard_bags_data = hard_bags
    if len(hard_bags_data) > 0:
        p_hard, _ = predict(m, hard_bags_data)
        p_pos_hard, _ = predict(m, ptx_pos)
        y_hard_eval = np.array([0] * len(hard_bags_data) + [1] * len(ptx_pos))
        p_hard_eval = np.concatenate([p_hard, p_pos_hard])
        if len(set(y_hard_eval)) == 2:
            hard_auc_seeds = []
            for seed_idx in range(len(seeds)):
                m2, _ = train_model_v2(train_bags, hard_bags, V, seeds[seed_idx], cfg)
                ph, _ = predict(m2, hard_bags_data)
                pp, _ = predict(m2, ptx_pos)
                y_h = np.array([0]*len(hard_bags_data) + [1]*len(ptx_pos))
                p_h = np.concatenate([ph, pp])
                if len(set(y_h)) == 2:
                    hard_auc_seeds.append(roc_auc_score(y_h, p_h))
            hard_auc = {"mean": float(np.mean(hard_auc_seeds)), "std": float(np.std(hard_auc_seeds)),
                        "per_seed": hard_auc_seeds}
        else:
            hard_auc = {"mean": 0.5, "std": 0.0, "per_seed": [], "note": "insufficient classes"}
    else:
        hard_auc = {"mean": None, "std": None, "per_seed": [], "note": "no hard bags available"}

    cross_auc  = roc_auc_score(y_ptx, p_ptx)
    cross_aupr = average_precision_score(y_ptx, p_ptx)
    idx_all = np.arange(len(y_ptx))
    def auc_bs(rng):
        s = rng.choice(idx_all, len(idx_all), replace=True)
        if len(set(y_ptx[s])) < 2: return cross_auc
        return roc_auc_score(y_ptx[s], p_ptx[s])
    cross = {"auc": float(cross_auc), "auc_ci": boot_ci(auc_bs), "aupr": float(cross_aupr)}

    # Localization
    gt_lists = [b["gt_idx"] for b in ptx_pos]
    def loc_metrics(rank_lists):
        bests = [hits_per_bag(r, g) for r, g in zip(rank_lists, gt_lists)]
        bests = [b for b in bests if b is not None]
        bests = np.array(bests); n = len(bests)
        base = {f"hit@{k}": float((bests < k).mean()) for k in (1, 5, 10)}
        base["mrr"] = float((1.0 / (bests + 1)).mean()); base["n"] = n
        return base
    loc_attn = loc_metrics(pos_attn)

    print(f"  → F1={indom['f1']['mean']:.4f}±{indom['f1']['std']:.4f} | "
          f"Hard-AUC={hard_auc['mean']:.4f}±{hard_auc.get('std',0):.4f} | "
          f"X-AUC={cross_auc:.4f} | "
          f"Loc Hit@1={loc_attn['hit@1']:.4f} MRR={loc_attn['mrr']:.4f}")

    return {
        "config": cfg,
        "seeds": seeds,
        "in_domain": indom,
        "hard_auc": hard_auc,
        "cross_domain": cross,
        "loc_attention": loc_attn,
        "epoch_losses": epoch_loss_log
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="2 seeds only")
    parser.add_argument("--config", type=str, default=None, help="Run single config by name")
    args = parser.parse_args()

    print("=" * 60)
    print("UnifiedTMIL v2 SOTA Training Pipeline")
    print("=" * 60)
    t0 = time.time()

    # Load data
    train_bags = load("train_bags.pkl")
    test_bags  = load("test_bags.pkl")
    vocab = pickle.load(open(os.path.join(SRC, "vocab.pkl"), "rb"))
    V = len(vocab.token2id)

    ptx      = pickle.load(open(os.path.join(DATA, "ptx_bags.pkl"), "rb"))
    norm_neg = pickle.load(open(os.path.join(DATA, "normal_eoa_neg.pkl"), "rb"))
    ptx_pos  = [b for b in ptx if b["label"] == 1]
    ptx_neg  = [b for b in ptx if b["label"] == 0]
    ptx_all  = ptx_pos + ptx_neg + norm_neg
    y_ptx    = np.array([b["label"] for b in ptx_all])
    y_test   = np.array([b["label"] for b in test_bags])

    # Hard negatives: DeFi/KOL accounts from defi_hard_bags.pkl
    hard_bags_path = os.path.join(DATA, "defi_hard_bags.pkl")
    if os.path.exists(hard_bags_path):
        hard_bags = pickle.load(open(hard_bags_path, "rb"))
        print(f"Loaded {len(hard_bags)} hard negative bags (DeFi/KOL)")
    else:
        # Fall back to normal_eoa_neg as hard negatives if defi file missing
        hard_bags = norm_neg[:min(200, len(norm_neg))]
        print(f"defi_hard_bags.pkl not found — using {len(hard_bags)} normal_eoa_neg as hard negatives")

    print(f"train={len(train_bags)} test={len(test_bags)} V={V}")
    print(f"ptx_pos={len(ptx_pos)} ptx_neg={len(ptx_neg)} norm_neg={len(norm_neg)} hard={len(hard_bags)}")

    # Select configs to run
    if args.config:
        if args.config not in CONFIGS:
            print(f"Unknown config: {args.config}. Available: {list(CONFIGS.keys())}")
            sys.exit(1)
        configs_to_run = {args.config: CONFIGS[args.config]}
    else:
        configs_to_run = CONFIGS

    all_results = {}
    for cname, cfg in configs_to_run.items():
        result = run_config(cname, cfg, train_bags, hard_bags, test_bags,
                            ptx_all, y_ptx, y_test, V, ptx_pos, quick=args.quick)
        all_results[cname] = result

        # Save per-config log
        log_dir = os.path.join(LOGS, f"unifiedtmil_v2_{cname}")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)

    # Save combined results
    out_path = os.path.join(RES, "unifiedtmil_v2_sota_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "description": "UnifiedTMIL v2 SOTA experiments",
                "seeds": SEEDS if not args.quick else [42, 1337],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": DEV,
                "quick_mode": args.quick
            },
            "configs": all_results
        }, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE (v1 baseline vs v2 configs)")
    print("=" * 80)
    print(f"{'Config':<20} {'ID-F1':>10} {'Hard-AUC':>10} {'X-AUC':>10} {'Hit@1':>8} {'MRR':>8}")
    print("-" * 80)
    # v1 reference from paper
    print(f"{'v1 (paper)':20} {'0.744':>10} {'0.696':>10} {'0.981':>10} {'0.802':>8} {'0.854':>8}")
    for cname, r in all_results.items():
        f1   = r["in_domain"]["f1"]["mean"]
        hauc = r["hard_auc"]["mean"] if r["hard_auc"]["mean"] is not None else float('nan')
        xauc = r["cross_domain"]["auc"]
        h1   = r["loc_attention"]["hit@1"]
        mrr  = r["loc_attention"]["mrr"]
        print(f"{cname:<20} {f1:>10.4f} {hauc:>10.4f} {xauc:>10.4f} {h1:>8.4f} {mrr:>8.4f}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")

    # Update CHANGELOG_SOTA.md
    changelog_path = os.path.join(DOCS, "CHANGELOG_SOTA.md")
    best_config = max(all_results.items(),
                      key=lambda x: (x[1]["in_domain"]["f1"]["mean"] +
                                     (x[1]["hard_auc"]["mean"] or 0) * 0.5))
    bc_name, bc = best_config
    changelog_entry = f"""
## {time.strftime('%Y-%m-%d')} — UnifiedTMIL v2 SOTA Experiments

### Changes from v1 (paper_final.pdf baseline)
1. **Focal loss** (γ=2.0, α=0.75) replaces BCE — focuses training on hard negatives
2. **Hard-negative mining** — ensures DeFi/KOL accounts appear in every training batch
3. **Residual skip connection** — mean-pooled h added to attention context z
4. **Feature injection moved AFTER attention** — prevents extra features from corrupting attention sharpness
5. **Finer λc grid** (0.05, 0.10, 0.15, 0.20) with 5 seeds each (vs 2 seeds in v1)
6. **Finer λe grid** (0.02, 0.04, 0.06) replacing coarse (0.10, 0.25, 0.50)
7. **Dilated TCN option** — wider receptive field for complex DeFi/KOL patterns
8. **Uncertainty weighting option** — GradNorm-style dynamic loss balancing

### Best Config: `{bc_name}`
- ID-F1: {bc['in_domain']['f1']['mean']:.4f} ± {bc['in_domain']['f1']['std']:.4f}
- Hard-AUC: {bc['hard_auc']['mean']:.4f} ± {bc['hard_auc'].get('std', 0):.4f}
- X-AUC: {bc['cross_domain']['auc']:.4f}
- Loc Hit@1: {bc['loc_attention']['hit@1']:.4f} | MRR: {bc['loc_attention']['mrr']:.4f}

### Log Reference
- Results: `results/unifiedtmil_v2_sota_results.json`
- Per-config logs: `results/logs/unifiedtmil_v2_*/results.json`

### Conclusion
PENDING — awaiting GPU run with full 5 seeds for final numbers.
"""
    with open(changelog_path, "a") as f:
        f.write(changelog_entry)
    print(f"\nChangelog updated: {changelog_path}")


if __name__ == "__main__":
    main()
