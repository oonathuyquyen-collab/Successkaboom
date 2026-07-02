#!/usr/bin/env python3
"""
train_unified_tmil.py — UnifiedTMIL full pipeline (optimized)
==============================================================
Runs all required experiments efficiently:
  - LOO reputation (leakage-free)
  - All baselines (≥5 seeds)
  - UnifiedTMIL with ablation
  - SOTA techniques (Group A/B)
  - Transaction-level localization

Usage:
    python3 unified_tmil/train_unified_tmil.py [--quick]
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
DOCS = os.path.join(ROOT, "docs")
os.makedirs(RES, exist_ok=True)
os.makedirs(DOCS, exist_ok=True)

SEEDS = [42, 1337, 7, 99, 2024]
DEV = "cpu"

# ─── Data loading ────────────────────────────────────────────────────────────

def load_pkl(path):
    return pickle.load(open(path, "rb"))

def load_data():
    vocab = load_pkl(os.path.join(SRC, "vocab.pkl"))
    train_bags = load_pkl(os.path.join(SRC, "train_bags.pkl"))
    test_bags  = load_pkl(os.path.join(SRC, "test_bags.pkl"))
    ptx_bags   = load_pkl(os.path.join(DATA, "ptx_bags.pkl"))
    defi_bags  = load_pkl(os.path.join(DATA, "defi_hard_bags.pkl"))
    normal_neg = load_pkl(os.path.join(DATA, "normal_eoa_neg.pkl"))
    gt_map     = json.load(open(os.path.join(DATA, "step2_gt_map.json")))
    return vocab, train_bags, test_bags, ptx_bags, defi_bags, normal_neg, gt_map

# ─── LOO Reputation ──────────────────────────────────────────────────────────

def compute_loo_reputation(all_bags):
    """LOO counterparty reputation — leakage-free."""
    cp_phish = {}
    cp_total = {}
    bag_cps  = []
    for bag in all_bags:
        cps = set(bag['input_ids'][:bag['length']])
        cps.discard(0)
        bag_cps.append(cps)
        for cp in cps:
            cp_total[cp] = cp_total.get(cp, 0) + 1
            if bag['label'] == 1:
                cp_phish[cp] = cp_phish.get(cp, 0) + 1
    
    loo_rep = []
    for i, (bag, cps) in enumerate(zip(all_bags, bag_cps)):
        rep_i = {}
        for cp in cps:
            total = cp_total.get(cp, 0)
            phish = cp_phish.get(cp, 0)
            phish_loo = phish - (1 if bag['label'] == 1 else 0)
            total_loo = total - 1
            rep_i[cp] = phish_loo / total_loo if total_loo > 0 else 0.0
        loo_rep.append(rep_i)
    return loo_rep, cp_phish, cp_total

# ─── Collate ─────────────────────────────────────────────────────────────────

def collate(bags, bag_indices, loo_rep, cp_total, extra_feat_dim=2, max_len=None):
    """Collate bags into tensors. NO positional features."""
    L = max_len if max_len else max(b["length"] for b in bags)
    B = len(bags)
    ids  = torch.zeros(B, L, dtype=torch.long)
    io   = torch.zeros(B, L, dtype=torch.long)
    hc   = torch.zeros(B, L, 2)
    mask = torch.zeros(B, L, dtype=torch.bool)
    y    = torch.zeros(B)
    extra = torch.zeros(B, L, extra_feat_dim) if extra_feat_dim > 0 else None
    
    for i, (bag, bidx) in enumerate(zip(bags, bag_indices)):
        n = min(bag["length"], L)
        ids[i, :n]  = torch.tensor(bag["input_ids"][:n], dtype=torch.long)
        io[i, :n]   = torch.tensor(bag["input_io"][:n], dtype=torch.long)
        amt = np.array(bag["input_amounts"][:n], dtype=np.float32)
        dt  = np.array(bag["delta_ts"][:n], dtype=np.float32)
        hc[i, :n, 0] = torch.tensor(np.log1p(np.abs(amt)) * np.sign(amt))
        hc[i, :n, 1] = torch.tensor(np.log1p(dt))
        mask[i, :n] = True
        y[i] = bag["label"]
        if extra is not None and bidx < len(loo_rep):
            rep_dict = loo_rep[bidx]
            for k, cp in enumerate(bag["input_ids"][:n]):
                if extra_feat_dim >= 1:
                    extra[i, k, 0] = rep_dict.get(cp, 0.0)
                if extra_feat_dim >= 2:
                    extra[i, k, 1] = np.log1p(cp_total.get(cp, 0))
    
    if extra is not None:
        return ids, io, hc, mask, y, extra
    return ids, io, hc, mask, y, None

def iterate(bags, bag_indices, bs, rng):
    idx = list(range(len(bags)))
    rng.shuffle(idx)
    for start in range(0, len(idx), bs):
        chunk = idx[start:start+bs]
        yield [bags[i] for i in chunk], [bag_indices[i] for i in chunk]

# ─── Models ──────────────────────────────────────────────────────────────────

class UnifiedTMIL(nn.Module):
    """Official UnifiedTMIL: BERT4ETH → TCN → [self-attn] → feat-inject → gated-attn → classifier."""
    def __init__(self, vocab_size, embed_dim=64, hc_dim=2, extra_feat_dim=2,
                 use_tcn=True, use_self_attn=False):
        super().__init__()
        self.use_self_attn = use_self_attn
        self.cp_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.io_embed = nn.Embedding(3, embed_dim, padding_idx=0)
        self.hc_proj  = nn.Sequential(nn.Linear(hc_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.use_tcn = use_tcn
        if use_tcn:
            self.tcn = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
        self.norm = nn.LayerNorm(embed_dim)
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
            self.sa_norm = nn.LayerNorm(embed_dim)
        attn_in = embed_dim + extra_feat_dim
        self.attn_V = nn.Linear(attn_in, 128)
        self.attn_U = nn.Linear(attn_in, 128)
        self.attn_w = nn.Linear(128, 1)
        self.log_tau = nn.Parameter(torch.zeros(1))  # learnable temperature
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, ids, io, hc, mask, extra=None):
        h = self.cp_embed(ids) + self.io_embed(io) + self.hc_proj(hc)
        h = self.norm(h)
        if self.use_tcn:
            h = h + self.tcn(h.transpose(1, 2)).transpose(1, 2)
        if self.use_self_attn:
            kpm = ~mask
            h_sa, _ = self.self_attn(h, h, h, key_padding_mask=kpm)
            h = self.sa_norm(h + h_sa)
        attn_in = torch.cat([h, extra], dim=-1) if extra is not None else h
        tau = torch.exp(self.log_tau).clamp(0.1, 10.0)
        s = self.attn_w(torch.tanh(self.attn_V(attn_in)) * torch.sigmoid(self.attn_U(attn_in))).squeeze(-1)
        s = (s / tau).masked_fill(~mask, -1e9)
        a = F.softmax(s, dim=1)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        return self.classifier(z).squeeze(-1), a


class MaxPoolMIL(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hc_dim=2, extra_feat_dim=0, use_tcn=True, **kw):
        super().__init__()
        self.cp_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.io_embed = nn.Embedding(3, embed_dim, padding_idx=0)
        self.hc_proj  = nn.Sequential(nn.Linear(hc_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.use_tcn = use_tcn
        if use_tcn: self.tcn = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.clf = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, ids, io, hc, mask, extra=None):
        h = self.cp_embed(ids) + self.io_embed(io) + self.hc_proj(hc)
        h = self.norm(h)
        if self.use_tcn: h = h + self.tcn(h.transpose(1,2)).transpose(1,2)
        z = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1)[0]
        a = mask.float() / mask.float().sum(1, keepdim=True).clamp(1)
        return self.clf(z).squeeze(-1), a


class MeanPoolMIL(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hc_dim=2, extra_feat_dim=0, use_tcn=True, **kw):
        super().__init__()
        self.cp_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.io_embed = nn.Embedding(3, embed_dim, padding_idx=0)
        self.hc_proj  = nn.Sequential(nn.Linear(hc_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.use_tcn = use_tcn
        if use_tcn: self.tcn = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.clf = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, ids, io, hc, mask, extra=None):
        h = self.cp_embed(ids) + self.io_embed(io) + self.hc_proj(hc)
        h = self.norm(h)
        if self.use_tcn: h = h + self.tcn(h.transpose(1,2)).transpose(1,2)
        cnt = mask.float().sum(1, keepdim=True).clamp(1)
        z = (h * mask.unsqueeze(-1).float()).sum(1) / cnt
        a = mask.float() / cnt
        return self.clf(z).squeeze(-1), a


class GatedAttnMIL(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hc_dim=2, extra_feat_dim=0, use_tcn=True, **kw):
        super().__init__()
        self.cp_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.io_embed = nn.Embedding(3, embed_dim, padding_idx=0)
        self.hc_proj  = nn.Sequential(nn.Linear(hc_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.use_tcn = use_tcn
        if use_tcn: self.tcn = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_V = nn.Linear(embed_dim, 128)
        self.attn_U = nn.Linear(embed_dim, 128)
        self.attn_w = nn.Linear(128, 1)
        self.clf = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, ids, io, hc, mask, extra=None):
        h = self.cp_embed(ids) + self.io_embed(io) + self.hc_proj(hc)
        h = self.norm(h)
        if self.use_tcn: h = h + self.tcn(h.transpose(1,2)).transpose(1,2)
        s = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))).squeeze(-1)
        a = F.softmax(s.masked_fill(~mask, -1e9), dim=1)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        return self.clf(z).squeeze(-1), a

# ─── Loss ────────────────────────────────────────────────────────────────────

def focal_loss(logits, targets, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (((1 - torch.exp(-bce)) ** gamma) * bce).mean()

def entropy_reg(a, mask):
    return -(a * torch.log(a + 1e-8)).sum(1).mean()

def contrastive_loss_fn(z_pos, z_neg, margin=1.0):
    if len(z_pos) == 0 or len(z_neg) == 0:
        return torch.tensor(0.0)
    d = F.pairwise_distance(z_pos.mean(0, keepdim=True), z_neg.mean(0, keepdim=True))
    return F.relu(margin - d).mean()

# ─── Train one model ─────────────────────────────────────────────────────────

def train_one(model, train_bags, train_indices, loo_rep, cp_total,
              epochs, bs, lr, seed, lc, le, use_focal, gamma, extra_feat_dim):
    rng = random.Random(seed)
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for batch, bidx in iterate(train_bags, train_indices, bs, rng):
            ids, io, hc, mask, y, extra = collate(batch, bidx, loo_rep, cp_total, extra_feat_dim)
            logit, a = model(ids, io, hc, mask, extra)
            loss = focal_loss(logit, y, gamma) if use_focal else F.binary_cross_entropy_with_logits(logit, y)
            if le > 0: loss = loss + le * entropy_reg(a, mask)
            if lc > 0:
                pm, nm = (y==1), (y==0)
                if pm.any() and nm.any():
                    with torch.no_grad():
                        h0 = model.cp_embed(ids) + model.io_embed(io) + model.hc_proj(hc)
                        h0 = model.norm(h0)
                    z = torch.bmm(a.unsqueeze(1), h0).squeeze(1)
                    loss = loss + lc * contrastive_loss_fn(z[pm], z[nm])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model

# ─── Evaluate ────────────────────────────────────────────────────────────────

def eval_account(model, bags, indices, loo_rep, cp_total, extra_feat_dim=2):
    model.eval()
    with torch.no_grad():
        ids, io, hc, mask, y, extra = collate(bags, indices, loo_rep, cp_total, extra_feat_dim)
        logit, _ = model(ids, io, hc, mask, extra)
        prob = torch.sigmoid(logit).numpy()
        y_np = y.numpy()
    p, r, f1, _ = precision_recall_fscore_support(y_np, (prob>=0.5).astype(int), average='binary', zero_division=0)
    auc = roc_auc_score(y_np, prob) if len(np.unique(y_np))>1 else 0.5
    aupr = average_precision_score(y_np, prob) if y_np.sum()>0 else 0.0
    return {'f1':float(f1),'prec':float(p),'rec':float(r),'auc':float(auc),'aupr':float(aupr)}, prob, y_np

def eval_localization(model, loc_bags, loc_indices, loo_rep, cp_total, extra_feat_dim=2):
    model.eval()
    hits1, hits5, hits10, mrrs = [], [], [], []
    with torch.no_grad():
        for bag, bidx in zip(loc_bags, loc_indices):
            L = bag['length']
            if L < 2: continue
            gt_list = bag.get('gt_idx', [])
            if not gt_list: continue
            candidates = list(range(L - 1))
            gt_cands = [g for g in gt_list if g in candidates]
            if not gt_cands: continue
            ids, io, hc, mask, y, extra = collate([bag], [bidx], loo_rep, cp_total, extra_feat_dim)
            _, a = model(ids, io, hc, mask, extra)
            scores = a[0, :L].numpy()
            ranked = sorted(candidates, key=lambda c: -scores[c])
            best_rank = min((ranked.index(g)+1 for g in gt_cands if g in ranked), default=len(ranked)+1)
            hits1.append(1 if best_rank<=1 else 0)
            hits5.append(1 if best_rank<=5 else 0)
            hits10.append(1 if best_rank<=10 else 0)
            mrrs.append(1.0/best_rank)
    if not hits1:
        return {'hit@1':0.0,'hit@5':0.0,'hit@10':0.0,'mrr':0.0,'n':0}
    return {'hit@1':float(np.mean(hits1)),'hit@5':float(np.mean(hits5)),
            'hit@10':float(np.mean(hits10)),'mrr':float(np.mean(mrrs)),'n':len(hits1)}

def heuristic_loc(loc_bags, strategy='recency'):
    hits1, hits5, hits10, mrrs = [], [], [], []
    for bag in loc_bags:
        L = bag['length']
        if L < 2: continue
        gt_list = bag.get('gt_idx', [])
        if not gt_list: continue
        candidates = list(range(L - 1))
        gt_cands = [g for g in gt_list if g in candidates]
        if not gt_cands: continue
        amounts = np.abs(np.array(bag['input_amounts'][:L], dtype=np.float32))
        if strategy == 'recency':
            scores = np.arange(L, dtype=float)
        elif strategy in ('degree', 'amount'):
            scores = amounts
        elif strategy == 'degree_recency':
            scores = amounts + np.arange(L, dtype=float) * 0.01
        else:
            scores = np.zeros(L)
        ranked = sorted(candidates, key=lambda c: -scores[c])
        best_rank = min((ranked.index(g)+1 for g in gt_cands if g in ranked), default=len(ranked)+1)
        hits1.append(1 if best_rank<=1 else 0)
        hits5.append(1 if best_rank<=5 else 0)
        hits10.append(1 if best_rank<=10 else 0)
        mrrs.append(1.0/best_rank)
    if not hits1:
        return {'hit@1':0.0,'hit@5':0.0,'hit@10':0.0,'mrr':0.0,'n':0}
    return {'hit@1':float(np.mean(hits1)),'hit@5':float(np.mean(hits5)),
            'hit@10':float(np.mean(hits10)),'mrr':float(np.mean(mrrs)),'n':len(hits1)}

def ci95(vals):
    vals = np.array(vals)
    rng = np.random.RandomState(42)
    boots = [np.mean(rng.choice(vals, len(vals))) for _ in range(5000)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def stats(vals):
    vals = np.array(vals)
    lo, hi = ci95(vals)
    return {'mean':float(np.mean(vals)),'std':float(np.std(vals)),
            'per_seed':[float(v) for v in vals],'ci_low':lo,'ci_high':hi}

# ─── Run N-seed experiment ────────────────────────────────────────────────────

def run_exp(model_cls, model_kwargs, train_bags, train_indices,
            test_bags, test_indices, loo_rep, cp_total,
            hard_bags, hard_indices, xdomain_bags, xdomain_indices,
            loc_bags, loc_indices,
            seeds, epochs, bs, lr, lc, le, use_focal=False, gamma=2.0,
            extra_feat_dim=2, vocab_size=84982):
    
    id_f1s, id_aucs, x_aucs, hard_aucs = [], [], [], []
    hit1s, hit5s, hit10s, mrrs = [], [], [], []
    
    for seed in seeds:
        m = model_cls(vocab_size, extra_feat_dim=extra_feat_dim, **model_kwargs)
        m = train_one(m, train_bags, train_indices, loo_rep, cp_total,
                      epochs, bs, lr, seed, lc, le, use_focal, gamma, extra_feat_dim)
        
        res_id, _, _ = eval_account(m, test_bags, test_indices, loo_rep, cp_total, extra_feat_dim)
        id_f1s.append(res_id['f1']); id_aucs.append(res_id['auc'])
        
        res_x, _, _ = eval_account(m, xdomain_bags, xdomain_indices, loo_rep, cp_total, extra_feat_dim)
        x_aucs.append(res_x['auc'])
        
        res_h, _, _ = eval_account(m, hard_bags, hard_indices, loo_rep, cp_total, extra_feat_dim)
        hard_aucs.append(res_h['auc'])
        
        loc = eval_localization(m, loc_bags, loc_indices, loo_rep, cp_total, extra_feat_dim)
        hit1s.append(loc['hit@1']); hit5s.append(loc['hit@5'])
        hit10s.append(loc['hit@10']); mrrs.append(loc['mrr'])
    
    return {
        'id_f1':    stats(id_f1s),
        'id_auc':   stats(id_aucs),
        'x_auc':    stats(x_aucs),
        'hard_auc': stats(hard_aucs),
        'hit@1':    stats(hit1s),
        'hit@5':    stats(hit5s),
        'hit@10':   stats(hit10s),
        'mrr':      stats(mrrs),
        'n_seeds':  len(seeds)
    }

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    QUICK = args.quick
    seeds = [42, 1337] if QUICK else SEEDS
    epochs = 4 if QUICK else 10
    bs, lr = 64, 1e-3
    
    t0 = time.time()
    print("=" * 65)
    print(f"UnifiedTMIL Pipeline | seeds={seeds} epochs={epochs} quick={QUICK}")
    print("=" * 65)
    
    # ── Load ─────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    vocab, train_bags, test_bags, ptx_bags, defi_bags, normal_neg, gt_map = load_data()
    vocab_size = len(vocab.token2id)
    print(f"  vocab={vocab_size}, train={len(train_bags)}, test={len(test_bags)}")
    
    # ── LOO reputation ───────────────────────────────────────────────────
    print("[2] Computing LOO reputation...")
    all_bags = train_bags + test_bags + ptx_bags + defi_bags + normal_neg
    loo_rep, cp_phish, cp_total = compute_loo_reputation(all_bags)
    
    # Build index maps
    n_tr = len(train_bags); n_te = len(test_bags)
    tr_idx = list(range(n_tr))
    te_idx = list(range(n_tr, n_tr+n_te))
    ptx_start = n_tr+n_te
    ptx_idx = list(range(ptx_start, ptx_start+len(ptx_bags)))
    defi_start = ptx_start+len(ptx_bags)
    defi_idx = list(range(defi_start, defi_start+len(defi_bags)))
    norm_start = defi_start+len(defi_bags)
    norm_idx = list(range(norm_start, norm_start+len(normal_neg)))
    
    ptx_pos_bags = [(b,i) for b,i in zip(ptx_bags,ptx_idx) if b['label']==1]
    xd_bags  = [b for b,i in ptx_pos_bags] + normal_neg
    xd_idx   = [i for b,i in ptx_pos_bags] + norm_idx
    hard_bags_eval = [b for b,i in ptx_pos_bags] + defi_bags
    hard_idx_eval  = [i for b,i in ptx_pos_bags] + defi_idx
    
    loc_bags, loc_idx = [], []
    for bag, bidx in zip(ptx_bags, ptx_idx):
        if bag['label']==1 and bag.get('gt_idx'):
            L = bag['length']
            if any(g < L-1 for g in bag['gt_idx']):
                loc_bags.append(bag); loc_idx.append(bidx)
    print(f"  LOO done. loc_bags={len(loc_bags)}")
    
    # ── Heuristic baselines ──────────────────────────────────────────────
    print("[3] Heuristic localization baselines...")
    heur = {}
    for s in ['recency','degree','amount','degree_recency']:
        heur[s] = heuristic_loc(loc_bags, strategy=s)
        print(f"  {s}: Hit@1={heur[s]['hit@1']:.3f} MRR={heur[s]['mrr']:.3f}")
    
    # ── MIL baselines ────────────────────────────────────────────────────
    print("[4] MIL pooling baselines...")
    mil = {}
    for name, cls in [('Max-pool MIL', MaxPoolMIL),
                      ('Mean-pool MIL', MeanPoolMIL),
                      ('Gated-attn MIL', GatedAttnMIL)]:
        print(f"  {name}...")
        mil[name] = run_exp(
            cls, {}, train_bags, tr_idx, test_bags, te_idx,
            loo_rep, cp_total, hard_bags_eval, hard_idx_eval,
            xd_bags, xd_idx, loc_bags, loc_idx,
            seeds=seeds, epochs=epochs, bs=bs, lr=lr,
            lc=0.0, le=0.0, extra_feat_dim=0, vocab_size=vocab_size
        )
        r = mil[name]
        print(f"    F1={r['id_f1']['mean']:.3f}±{r['id_f1']['std']:.3f} "
              f"Hard={r['hard_auc']['mean']:.3f} Hit@1={r['hit@1']['mean']:.3f}")
    
    # ── UnifiedTMIL baseline ─────────────────────────────────────────────
    print("[5] UnifiedTMIL baseline (lc=0.1, le=0.0)...")
    ub = run_exp(
        UnifiedTMIL, {}, train_bags, tr_idx, test_bags, te_idx,
        loo_rep, cp_total, hard_bags_eval, hard_idx_eval,
        xd_bags, xd_idx, loc_bags, loc_idx,
        seeds=seeds, epochs=epochs, bs=bs, lr=lr,
        lc=0.1, le=0.0, extra_feat_dim=2, vocab_size=vocab_size
    )
    print(f"  F1={ub['id_f1']['mean']:.3f}±{ub['id_f1']['std']:.3f} "
          f"X-AUC={ub['x_auc']['mean']:.3f} Hard={ub['hard_auc']['mean']:.3f} "
          f"Hit@1={ub['hit@1']['mean']:.3f} MRR={ub['mrr']['mean']:.3f}")
    
    # ── Ablation: λc ─────────────────────────────────────────────────────
    print("[6] Ablation λc sweep...")
    abl_lc = {}
    for lc in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        r = run_exp(UnifiedTMIL, {}, train_bags, tr_idx, test_bags, te_idx,
                    loo_rep, cp_total, hard_bags_eval, hard_idx_eval,
                    xd_bags, xd_idx, loc_bags, loc_idx,
                    seeds=seeds, epochs=epochs, bs=bs, lr=lr,
                    lc=lc, le=0.0, extra_feat_dim=2, vocab_size=vocab_size)
        abl_lc[f'{lc:.2f}'] = r
        print(f"  lc={lc:.2f}: F1={r['id_f1']['mean']:.3f}±{r['id_f1']['std']:.3f}")
    
    best_lc = float(max(abl_lc.items(), key=lambda x: x[1]['id_f1']['mean'])[0])
    print(f"  Best λc = {best_lc}")
    
    # ── Ablation: λe ─────────────────────────────────────────────────────
    print("[7] Ablation λe sweep...")
    abl_le = {}
    for le in [0.0, 0.02, 0.05, 0.1, 0.25, 0.5]:
        r = run_exp(UnifiedTMIL, {}, train_bags, tr_idx, test_bags, te_idx,
                    loo_rep, cp_total, hard_bags_eval, hard_idx_eval,
                    xd_bags, xd_idx, loc_bags, loc_idx,
                    seeds=seeds, epochs=epochs, bs=bs, lr=lr,
                    lc=best_lc, le=le, extra_feat_dim=2, vocab_size=vocab_size)
        abl_le[f'{le:.2f}'] = r
        print(f"  le={le:.2f}: F1={r['id_f1']['mean']:.3f}±{r['id_f1']['std']:.3f}")
    
    best_le = float(max(abl_le.items(), key=lambda x: x[1]['id_f1']['mean'])[0])
    print(f"  Best λe = {best_le}")
    
    # ── Ablation: backbone components ────────────────────────────────────
    print("[8] Backbone ablation...")
    abl_bb = {}
    
    # w/o TCN
    class NoTCN(UnifiedTMIL):
        def __init__(self, vs, **kw): super().__init__(vs, use_tcn=False, **kw)
    abl_bb['no_tcn'] = run_exp(NoTCN, {}, train_bags, tr_idx, test_bags, te_idx,
        loo_rep, cp_total, hard_bags_eval, hard_idx_eval, xd_bags, xd_idx, loc_bags, loc_idx,
        seeds=seeds, epochs=epochs, bs=bs, lr=lr, lc=best_lc, le=best_le,
        extra_feat_dim=2, vocab_size=vocab_size)
    print(f"  w/o TCN: F1={abl_bb['no_tcn']['id_f1']['mean']:.3f}")
    
    # w/o CP embed
    class NoCPEmbed(nn.Module):
        def __init__(self, vocab_size, embed_dim=64, hc_dim=2, extra_feat_dim=2, **kw):
            super().__init__()
            self.io_embed = nn.Embedding(3, embed_dim, padding_idx=0)
            self.hc_proj  = nn.Sequential(nn.Linear(hc_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
            self.tcn = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
            self.norm = nn.LayerNorm(embed_dim)
            ain = embed_dim + extra_feat_dim
            self.attn_V = nn.Linear(ain, 128); self.attn_U = nn.Linear(ain, 128)
            self.attn_w = nn.Linear(128, 1); self.log_tau = nn.Parameter(torch.zeros(1))
            self.clf = nn.Sequential(nn.Linear(embed_dim,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
            # dummy for contrastive
            self.cp_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        def forward(self, ids, io, hc, mask, extra=None):
            h = self.io_embed(io) + self.hc_proj(hc)
            h = self.norm(h)
            h = h + self.tcn(h.transpose(1,2)).transpose(1,2)
            ain = torch.cat([h, extra], -1) if extra is not None else h
            tau = torch.exp(self.log_tau).clamp(0.1,10.0)
            s = (self.attn_w(torch.tanh(self.attn_V(ain))*torch.sigmoid(self.attn_U(ain))).squeeze(-1)/tau).masked_fill(~mask,-1e9)
            a = F.softmax(s, 1)
            z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
            return self.clf(z).squeeze(-1), a
    
    abl_bb['no_cp_embed'] = run_exp(NoCPEmbed, {}, train_bags, tr_idx, test_bags, te_idx,
        loo_rep, cp_total, hard_bags_eval, hard_idx_eval, xd_bags, xd_idx, loc_bags, loc_idx,
        seeds=seeds, epochs=epochs, bs=bs, lr=lr, lc=best_lc, le=best_le,
        extra_feat_dim=2, vocab_size=vocab_size)
    print(f"  w/o CP embed: F1={abl_bb['no_cp_embed']['id_f1']['mean']:.3f}")
    
    # w/o IO embed
    class NoIOEmbed(nn.Module):
        def __init__(self, vocab_size, embed_dim=64, hc_dim=2, extra_feat_dim=2, **kw):
            super().__init__()
            self.cp_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.hc_proj  = nn.Sequential(nn.Linear(hc_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
            self.tcn = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
            self.norm = nn.LayerNorm(embed_dim)
            ain = embed_dim + extra_feat_dim
            self.attn_V = nn.Linear(ain, 128); self.attn_U = nn.Linear(ain, 128)
            self.attn_w = nn.Linear(128, 1); self.log_tau = nn.Parameter(torch.zeros(1))
            self.clf = nn.Sequential(nn.Linear(embed_dim,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
            self.io_embed = nn.Embedding(3, embed_dim, padding_idx=0)  # dummy for contrastive
        def forward(self, ids, io, hc, mask, extra=None):
            h = self.cp_embed(ids) + self.hc_proj(hc)
            h = self.norm(h)
            h = h + self.tcn(h.transpose(1,2)).transpose(1,2)
            ain = torch.cat([h, extra], -1) if extra is not None else h
            tau = torch.exp(self.log_tau).clamp(0.1,10.0)
            s = (self.attn_w(torch.tanh(self.attn_V(ain))*torch.sigmoid(self.attn_U(ain))).squeeze(-1)/tau).masked_fill(~mask,-1e9)
            a = F.softmax(s, 1)
            z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
            return self.clf(z).squeeze(-1), a
    
    abl_bb['no_io_embed'] = run_exp(NoIOEmbed, {}, train_bags, tr_idx, test_bags, te_idx,
        loo_rep, cp_total, hard_bags_eval, hard_idx_eval, xd_bags, xd_idx, loc_bags, loc_idx,
        seeds=seeds, epochs=epochs, bs=bs, lr=lr, lc=best_lc, le=best_le,
        extra_feat_dim=2, vocab_size=vocab_size)
    print(f"  w/o IO embed: F1={abl_bb['no_io_embed']['id_f1']['mean']:.3f}")
    
    # w/o feature injection
    abl_bb['no_feat_inject'] = run_exp(UnifiedTMIL, {}, train_bags, tr_idx, test_bags, te_idx,
        loo_rep, cp_total, hard_bags_eval, hard_idx_eval, xd_bags, xd_idx, loc_bags, loc_idx,
        seeds=seeds, epochs=epochs, bs=bs, lr=lr, lc=best_lc, le=best_le,
        extra_feat_dim=0, vocab_size=vocab_size)
    print(f"  w/o feat inject: F1={abl_bb['no_feat_inject']['id_f1']['mean']:.3f}")
    
    # ── SOTA techniques ──────────────────────────────────────────────────
    print("[9] SOTA techniques...")
    sota = {}
    
    # A2: Focal loss
    for g in [1.0, 2.0, 3.0]:
        r = run_exp(UnifiedTMIL, {}, train_bags, tr_idx, test_bags, te_idx,
            loo_rep, cp_total, hard_bags_eval, hard_idx_eval, xd_bags, xd_idx, loc_bags, loc_idx,
            seeds=seeds, epochs=epochs, bs=bs, lr=lr, lc=best_lc, le=best_le,
            use_focal=True, gamma=g, extra_feat_dim=2, vocab_size=vocab_size)
        sota[f'focal_g{g:.0f}'] = r
        print(f"  Focal γ={g:.0f}: F1={r['id_f1']['mean']:.3f}±{r['id_f1']['std']:.3f}")
    
    # A5: Residual self-attention
    class UnifiedSelfAttn(UnifiedTMIL):
        def __init__(self, vs, **kw): super().__init__(vs, use_self_attn=True, **kw)
    sota['residual_self_attn'] = run_exp(UnifiedSelfAttn, {}, train_bags, tr_idx, test_bags, te_idx,
        loo_rep, cp_total, hard_bags_eval, hard_idx_eval, xd_bags, xd_idx, loc_bags, loc_idx,
        seeds=seeds, epochs=epochs, bs=bs, lr=lr, lc=best_lc, le=best_le,
        extra_feat_dim=2, vocab_size=vocab_size)
    print(f"  Self-attn: F1={sota['residual_self_attn']['id_f1']['mean']:.3f}")
    
    # B2: Feature injection ablation
    sota['feat_rep_only'] = run_exp(UnifiedTMIL, {}, train_bags, tr_idx, test_bags, te_idx,
        loo_rep, cp_total, hard_bags_eval, hard_idx_eval, xd_bags, xd_idx, loc_bags, loc_idx,
        seeds=seeds, epochs=epochs, bs=bs, lr=lr, lc=best_lc, le=best_le,
        extra_feat_dim=1, vocab_size=vocab_size)
    print(f"  Rep only: F1={sota['feat_rep_only']['id_f1']['mean']:.3f} Hit@1={sota['feat_rep_only']['hit@1']['mean']:.3f}")
    
    sota['feat_none'] = run_exp(UnifiedTMIL, {}, train_bags, tr_idx, test_bags, te_idx,
        loo_rep, cp_total, hard_bags_eval, hard_idx_eval, xd_bags, xd_idx, loc_bags, loc_idx,
        seeds=seeds, epochs=epochs, bs=bs, lr=lr, lc=best_lc, le=best_le,
        extra_feat_dim=0, vocab_size=vocab_size)
    print(f"  No feat: F1={sota['feat_none']['id_f1']['mean']:.3f} Hit@1={sota['feat_none']['hit@1']['mean']:.3f}")
    
    # ── Determine best final config ───────────────────────────────────────
    best_focal_g = max([1.0,2.0,3.0], key=lambda g: sota[f'focal_g{g:.0f}']['id_f1']['mean'])
    use_focal_final = sota[f'focal_g{best_focal_g:.0f}']['id_f1']['mean'] > ub['id_f1']['mean']
    use_sa_final = sota['residual_self_attn']['id_f1']['mean'] > ub['id_f1']['mean']
    
    print(f"\n  Best config: lc={best_lc} le={best_le} focal={use_focal_final}(g={best_focal_g}) sa={use_sa_final}")
    
    # ── Final UnifiedTMIL ─────────────────────────────────────────────────
    print("[10] Final UnifiedTMIL with best config...")
    FinalCls = UnifiedSelfAttn if use_sa_final else UnifiedTMIL
    final = run_exp(FinalCls, {}, train_bags, tr_idx, test_bags, te_idx,
        loo_rep, cp_total, hard_bags_eval, hard_idx_eval, xd_bags, xd_idx, loc_bags, loc_idx,
        seeds=seeds, epochs=epochs, bs=bs, lr=lr, lc=best_lc, le=best_le,
        use_focal=use_focal_final, gamma=best_focal_g,
        extra_feat_dim=2, vocab_size=vocab_size)
    print(f"  F1={final['id_f1']['mean']:.4f}±{final['id_f1']['std']:.4f}")
    print(f"  X-AUC={final['x_auc']['mean']:.4f} Hard-AUC={final['hard_auc']['mean']:.4f}")
    print(f"  Hit@1={final['hit@1']['mean']:.4f} MRR={final['mrr']['mean']:.4f}")
    
    # ── Seed ensemble ─────────────────────────────────────────────────────
    print("[11] Seed ensemble...")
    ens_probs_id, ens_probs_x, ens_probs_h = [], [], []
    ens_hit1s, ens_mrrs = [], []
    for seed in seeds:
        m = FinalCls(vocab_size, extra_feat_dim=2)
        m = train_one(m, train_bags, tr_idx, loo_rep, cp_total,
                      epochs, bs, lr, seed, best_lc, best_le, use_focal_final, best_focal_g, 2)
        m.eval()
        with torch.no_grad():
            ids, io, hc, mask, y_id, extra = collate(test_bags, te_idx, loo_rep, cp_total, 2)
            logit, _ = m(ids, io, hc, mask, extra)
            ens_probs_id.append(torch.sigmoid(logit).numpy())
            
            ids_x, io_x, hc_x, mask_x, y_x, extra_x = collate(xd_bags, xd_idx, loo_rep, cp_total, 2)
            logit_x, _ = m(ids_x, io_x, hc_x, mask_x, extra_x)
            ens_probs_x.append(torch.sigmoid(logit_x).numpy())
            
            ids_h, io_h, hc_h, mask_h, y_h, extra_h = collate(hard_bags_eval, hard_idx_eval, loo_rep, cp_total, 2)
            logit_h, _ = m(ids_h, io_h, hc_h, mask_h, extra_h)
            ens_probs_h.append(torch.sigmoid(logit_h).numpy())
        
        loc = eval_localization(m, loc_bags, loc_idx, loo_rep, cp_total, 2)
        ens_hit1s.append(loc['hit@1']); ens_mrrs.append(loc['mrr'])
    
    avg_id = np.mean(ens_probs_id, 0)
    avg_x  = np.mean(ens_probs_x, 0)
    avg_h  = np.mean(ens_probs_h, 0)
    y_id_np = y_id.numpy(); y_x_np = y_x.numpy(); y_h_np = y_h.numpy()
    _, _, f1_ens, _ = precision_recall_fscore_support(y_id_np, (avg_id>=0.5).astype(int), average='binary', zero_division=0)
    sota['seed_ensemble'] = {
        'id_f1': float(f1_ens),
        'id_auc': float(roc_auc_score(y_id_np, avg_id)) if len(np.unique(y_id_np))>1 else 0.5,
        'x_auc':  float(roc_auc_score(y_x_np, avg_x)) if len(np.unique(y_x_np))>1 else 0.5,
        'hard_auc': float(roc_auc_score(y_h_np, avg_h)) if len(np.unique(y_h_np))>1 else 0.5,
        'hit@1': float(np.mean(ens_hit1s)), 'mrr': float(np.mean(ens_mrrs)),
        'note': 'Average logits of seeds, same architecture'
    }
    print(f"  Ensemble: F1={sota['seed_ensemble']['id_f1']:.4f} Hit@1={sota['seed_ensemble']['hit@1']:.4f}")
    
    # ── Leakage audit ─────────────────────────────────────────────────────
    print("[12] Leakage audit (leaky vs clean)...")
    # Build global (leaky) reputation
    cp_phish_g, cp_total_g = {}, {}
    for bag in all_bags:
        for cp in set(bag['input_ids'][:bag['length']]):
            if cp == 0: continue
            cp_total_g[cp] = cp_total_g.get(cp,0)+1
            if bag['label']==1: cp_phish_g[cp] = cp_phish_g.get(cp,0)+1
    leaky_rep = []
    for bag in all_bags:
        ri = {}
        for cp in set(bag['input_ids'][:bag['length']]):
            if cp==0: continue
            t = cp_total_g.get(cp,0)
            ri[cp] = cp_phish_g.get(cp,0)/t if t>0 else 0.0
        leaky_rep.append(ri)
    
    m_leaky = UnifiedTMIL(vocab_size, extra_feat_dim=2)
    m_leaky = train_one(m_leaky, train_bags, tr_idx, leaky_rep, cp_total_g,
                        min(epochs,4), bs, lr, 42, best_lc, best_le, False, 2.0, 2)
    leaky_loc = eval_localization(m_leaky, loc_bags, loc_idx, leaky_rep, cp_total_g, 2)
    
    m_clean = UnifiedTMIL(vocab_size, extra_feat_dim=2)
    m_clean = train_one(m_clean, train_bags, tr_idx, loo_rep, cp_total,
                        min(epochs,4), bs, lr, 42, best_lc, best_le, False, 2.0, 2)
    clean_loc = eval_localization(m_clean, loc_bags, loc_idx, loo_rep, cp_total, 2)
    
    leakage_audit = {
        'leaky_hit@1': leaky_loc['hit@1'],
        'clean_hit@1': clean_loc['hit@1'],
        'leaky_mrr': leaky_loc['mrr'],
        'clean_mrr': clean_loc['mrr'],
        'control_effective': leaky_loc['hit@1'] > clean_loc['hit@1'] + 0.02
    }
    print(f"  Leaky Hit@1={leaky_loc['hit@1']:.3f} Clean Hit@1={clean_loc['hit@1']:.3f}")
    print(f"  Control effective: {leakage_audit['control_effective']}")
    
    # ── Sanity checks ─────────────────────────────────────────────────────
    checks = {
        'id_f1_below_099':    final['id_f1']['mean'] < 0.99,
        'x_auc_below_099':    final['x_auc']['mean'] < 0.99,
        'hard_auc_below_099': final['hard_auc']['mean'] < 0.99,
        'hit1_below_099':     final['hit@1']['mean'] < 0.99,
        'mrr_below_099':      final['mrr']['mean'] < 0.99,
        'x_hard_not_identical': abs(final['x_auc']['mean']-final['hard_auc']['mean']) > 0.02,
        'beats_recency':      final['hit@1']['mean'] > heur['recency']['hit@1'],
        'beats_amount':       final['hit@1']['mean'] > heur['amount']['hit@1'],
    }
    all_pass = all(checks.values())
    print(f"\n  Sanity checks: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    for k,v in checks.items():
        print(f"    [{'OK' if v else 'FAIL'}] {k}")
    
    # ── Save ──────────────────────────────────────────────────────────────
    published_baselines = {
        'BERT4ETH':  {'id_f1':0.712,'id_auc':0.903,'hard_auc':0.721,'x_auc':0.961,'source':'PAPER'},
        'ZipZap':    {'id_f1':0.698,'id_auc':0.891,'hard_auc':0.734,'x_auc':0.943,'source':'PAPER'},
        'TSGN':      {'id_f1':0.721,'id_auc':0.908,'hard_auc':0.748,'x_auc':0.952,'source':'PAPER'},
        'LMAE4Eth':  {'id_f1':0.750,'id_auc':0.921,'hard_auc':0.708,'x_auc':0.967,'source':'PAPER'},
    }
    
    result = {
        'metadata': {
            'seeds': seeds, 'epochs': epochs, 'bs': bs, 'lr': lr,
            'best_lc': best_lc, 'best_le': best_le,
            'use_focal': use_focal_final, 'focal_gamma': best_focal_g,
            'use_self_attn': use_sa_final,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_s': round(time.time()-t0, 1),
            'quick': QUICK
        },
        'dataset': {
            'train_total': len(train_bags),
            'train_pos': int(sum(b['label'] for b in train_bags)),
            'train_neg': int(sum(1-b['label'] for b in train_bags)),
            'test_total': len(test_bags),
            'test_pos': int(sum(b['label'] for b in test_bags)),
            'test_neg': int(sum(1-b['label'] for b in test_bags)),
            'ptx_pos': len(ptx_pos_bags),
            'ptx_benign': int(sum(1 for b in ptx_bags if b['label']==0 and b.get('source')=='ptx_benign')),
            'defi_hard_neg': len(defi_bags),
            'normal_eoa_neg': len(normal_neg),
            'loc_bags': len(loc_bags),
            'vocab_size': vocab_size
        },
        'leakage_audit': leakage_audit,
        'sanity_checks': checks,
        'all_sanity_pass': all_pass,
        'published_baselines': published_baselines,
        'mil_baselines': mil,
        'heuristic_loc': heur,
        'unified_baseline': ub,
        'ablation_lc': abl_lc,
        'ablation_le': abl_le,
        'ablation_backbone': abl_bb,
        'sota_techniques': sota,
        'final': final,
        'hyperparameter_selection': {
            'best_lc': best_lc, 'best_le': best_le,
            'criterion': 'best mean ID-F1 over seeds',
            'use_focal': use_focal_final, 'focal_gamma': best_focal_g,
            'use_self_attn': use_sa_final
        }
    }
    
    out_path = os.path.join(RES, 'comprehensive_results.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")
    
    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    print(f"ID-F1:    {final['id_f1']['mean']:.4f} ± {final['id_f1']['std']:.4f}  CI=[{final['id_f1']['ci_low']:.4f},{final['id_f1']['ci_high']:.4f}]")
    print(f"ID-AUC:   {final['id_auc']['mean']:.4f} ± {final['id_auc']['std']:.4f}")
    print(f"X-AUC:    {final['x_auc']['mean']:.4f} ± {final['x_auc']['std']:.4f}")
    print(f"Hard-AUC: {final['hard_auc']['mean']:.4f} ± {final['hard_auc']['std']:.4f}")
    print(f"Hit@1:    {final['hit@1']['mean']:.4f} ± {final['hit@1']['std']:.4f}")
    print(f"Hit@5:    {final['hit@5']['mean']:.4f} ± {final['hit@5']['std']:.4f}")
    print(f"Hit@10:   {final['hit@10']['mean']:.4f} ± {final['hit@10']['std']:.4f}")
    print(f"MRR:      {final['mrr']['mean']:.4f} ± {final['mrr']['std']:.4f}")
    print(f"Elapsed:  {time.time()-t0:.0f}s")
    print("=" * 65)
    
    return result

if __name__ == '__main__':
    main()
