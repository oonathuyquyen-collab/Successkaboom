#!/usr/bin/env python3
"""Quick verification: train 1 seed, 2 epochs, evaluate everything."""
import sys, os, json, pickle, random, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
import vocab_def
sys.modules['__main__'].Vocab = vocab_def.Vocab

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "data/bert4eth")
DATA = os.path.join(ROOT, "data")

vocab = pickle.load(open(os.path.join(SRC, "vocab.pkl"), "rb"))
train_bags = pickle.load(open(os.path.join(SRC, "train_bags.pkl"), "rb"))
test_bags = pickle.load(open(os.path.join(SRC, "test_bags.pkl"), "rb"))
ptx_bags = pickle.load(open(os.path.join(DATA, "ptx_bags.pkl"), "rb"))
defi_bags = pickle.load(open(os.path.join(DATA, "defi_hard_bags.pkl"), "rb"))
normal_neg = pickle.load(open(os.path.join(DATA, "normal_eoa_neg.pkl"), "rb"))
vocab_size = len(vocab.token2id)

all_bags = train_bags + test_bags + ptx_bags + defi_bags + normal_neg
n_tr = len(train_bags); n_te = len(test_bags)
tr_idx = list(range(n_tr))
te_idx = list(range(n_tr, n_tr + n_te))
ptx_start = n_tr + n_te
ptx_idx = list(range(ptx_start, ptx_start + len(ptx_bags)))
defi_start = ptx_start + len(ptx_bags)
defi_idx = list(range(defi_start, defi_start + len(defi_bags)))
norm_start = defi_start + len(defi_bags)
norm_idx = list(range(norm_start, norm_start + len(normal_neg)))

# LOO Reputation
cp_phish = {}; cp_total = {}; bag_cps = []
for bag in all_bags:
    cps = set(bag['input_ids'][:bag['length']]); cps.discard(0)
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

# Cross-domain & Hard
ptx_pos = [(b, i) for b, i in zip(ptx_bags, ptx_idx) if b['label'] == 1]
xd_bags = [b for b, _ in ptx_pos] + normal_neg
xd_idx = [i for _, i in ptx_pos] + norm_idx
hard_be = [b for b, _ in ptx_pos] + defi_bags
hard_ie = [i for _, i in ptx_pos] + defi_idx

# Loc bags
loc_bags = []; loc_idx = []
for bag, bidx in zip(ptx_bags, ptx_idx):
    if bag['label'] == 1 and bag.get('gt_idx'):
        L = bag['length']
        if any(g < L - 1 for g in bag['gt_idx']):
            loc_bags.append(bag); loc_idx.append(bidx)
print(f"loc_bags={len(loc_bags)}")

# Collate
def collate(bags, indices, extra_feat_dim=1):
    L = max(b["length"] for b in bags)
    B = len(bags)
    ids = torch.zeros(B, L, dtype=torch.long)
    io = torch.zeros(B, L, dtype=torch.long)
    hc = torch.zeros(B, L, 2)
    mask = torch.zeros(B, L, dtype=torch.bool)
    y = torch.zeros(B)
    extra = torch.zeros(B, L, extra_feat_dim) if extra_feat_dim > 0 else None
    for i, (bag, bidx) in enumerate(zip(bags, indices)):
        n = min(bag["length"], L)
        ids[i, :n] = torch.tensor(bag["input_ids"][:n], dtype=torch.long)
        io[i, :n] = torch.tensor(bag["input_io"][:n], dtype=torch.long)
        amt = np.array(bag["input_amounts"][:n], dtype=np.float32)
        dt = np.array(bag["delta_ts"][:n], dtype=np.float32)
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
    return ids, io, hc, mask, y, extra

# V3 Model
class UnifiedTMILv3(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, extra_feat_dim=1):
        super().__init__()
        self.cp_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.io_embed = nn.Embedding(3, embed_dim, padding_idx=0)
        self.hc_proj = nn.Sequential(nn.Linear(2, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.tcn = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
        self.norm = nn.LayerNorm(embed_dim)
        attn_in = embed_dim + extra_feat_dim
        self.attn_V = nn.Linear(attn_in, 128)
        self.attn_U = nn.Linear(attn_in, 128)
        self.attn_w = nn.Linear(128, 1)
        self.log_tau = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, ids, io, hc, mask, extra=None):
        h = self.cp_embed(ids) + self.io_embed(io) + self.hc_proj(hc)
        h = self.norm(h)
        h = h + self.tcn(h.transpose(1, 2)).transpose(1, 2)
        attn_in = torch.cat([h, extra], dim=-1) if extra is not None else h
        tau = torch.exp(self.log_tau).clamp(0.1, 10.0)
        s = self.attn_w(torch.tanh(self.attn_V(attn_in)) * torch.sigmoid(self.attn_U(attn_in))).squeeze(-1)
        s = (s / tau).masked_fill(~mask, -1e9)
        a = F.softmax(s, dim=1)
        z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        return self.classifier(z).squeeze(-1), a

def contrastive_loss(z_pos, z_neg, margin=1.0):
    if len(z_pos) == 0 or len(z_neg) == 0:
        return torch.tensor(0.0)
    d = F.pairwise_distance(z_pos.mean(0, keepdim=True), z_neg.mean(0, keepdim=True))
    return F.relu(margin - d).mean()

def entropy_reg(a, mask):
    return -(a * torch.log(a + 1e-8)).sum(1).mean()

# Train
torch.manual_seed(42); np.random.seed(42)
model = UnifiedTMILv3(vocab_size, extra_feat_dim=1)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
bs = 64
lc = 0.1; le = 0.02

t0 = time.time()
model.train()
for epoch in range(3):
    idx = list(range(len(train_bags)))
    random.Random(42 + epoch).shuffle(idx)
    losses = []
    for start in range(0, len(idx), bs):
        chunk = idx[start:start + bs]
        batch = [train_bags[i] for i in chunk]
        bidx = [tr_idx[i] for i in chunk]
        ids, io, hc, mask, y, extra = collate(batch, bidx, extra_feat_dim=1)
        logit, a = model(ids, io, hc, mask, extra)
        loss = F.binary_cross_entropy_with_logits(logit, y)
        if le > 0:
            loss = loss + le * entropy_reg(a, mask)
        if lc > 0:
            pm, nm = (y == 1), (y == 0)
            if pm.any() and nm.any():
                with torch.no_grad():
                    h0 = model.cp_embed(ids) + model.io_embed(io) + model.hc_proj(hc)
                    h0 = model.norm(h0)
                z = torch.bmm(a.unsqueeze(1), h0).squeeze(1)
                loss = loss + lc * contrastive_loss(z[pm], z[nm])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    print(f"  Epoch {epoch}: loss={np.mean(losses):.4f}")

print(f"Train done in {time.time() - t0:.0f}s")

# Evaluate
model.eval()
with torch.no_grad():
    # ID test
    ids, io, hc, mask, y, extra = collate(test_bags, te_idx, 1)
    logit, a = model(ids, io, hc, mask, extra)
    prob_id = torch.sigmoid(logit).numpy()
    y_id = y.numpy()
    p, r, f1, _ = precision_recall_fscore_support(y_id, (prob_id >= 0.5).astype(int), average='binary', zero_division=0)
    auc_id = roc_auc_score(y_id, prob_id) if len(np.unique(y_id)) > 1 else 0.5
    print(f"ID: F1={f1:.4f} AUC={auc_id:.4f}")

    # X-domain
    ids, io, hc, mask, y_x, extra = collate(xd_bags, xd_idx, 1)
    logit_x, _ = model(ids, io, hc, mask, extra)
    prob_x = torch.sigmoid(logit_x).numpy()
    y_x_np = y_x.numpy()
    auc_x = roc_auc_score(y_x_np, prob_x) if len(np.unique(y_x_np)) > 1 else 0.5
    print(f"X-AUC: {auc_x:.4f}")

    # Hard
    ids, io, hc, mask, y_h, extra = collate(hard_be, hard_ie, 1)
    logit_h, _ = model(ids, io, hc, mask, extra)
    prob_h = torch.sigmoid(logit_h).numpy()
    y_h_np = y_h.numpy()
    auc_h = roc_auc_score(y_h_np, prob_h) if len(np.unique(y_h_np)) > 1 else 0.5
    print(f"Hard-AUC: {auc_h:.4f}")

    # Localization
    hit1s, hit5s, hit10s, mrrs = [], [], [], []
    for bag, bidx in zip(loc_bags, loc_idx):
        L = bag['length']
        gt_cands = [g for g in bag.get('gt_idx', []) if g < L - 1]
        if not gt_cands:
            continue
        ids, io, hc, mask, _, extra = collate([bag], [bidx], 1)
        _, a = model(ids, io, hc, mask, extra)
        scores = a[0, :L].numpy()
        candidates = list(range(L - 1))
        ranked = sorted(candidates, key=lambda c: -scores[c])
        best_rank = min((ranked.index(g) + 1 for g in gt_cands if g in ranked), default=len(ranked) + 1)
        hit1s.append(1 if best_rank <= 1 else 0)
        hit5s.append(1 if best_rank <= 5 else 0)
        hit10s.append(1 if best_rank <= 10 else 0)
        mrrs.append(1.0 / best_rank)

    print(f"Loc: Hit@1={np.mean(hit1s):.4f} Hit@5={np.mean(hit5s):.4f} MRR={np.mean(mrrs):.4f} (n={len(hit1s)})")

print(f"Total time: {time.time() - t0:.0f}s")
print("VERIFY PASSED" if f1 > 0.3 else "VERIFY FAILED")
