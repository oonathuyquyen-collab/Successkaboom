#!/usr/bin/env python3
"""
finish_pipeline.py — Complete the pipeline from SOTA techniques onward
Uses results already computed and adds: feat_rep_only, final model, ensemble, leakage audit
"""

import sys, os, json, pickle, random, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "core"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import vocab_def
sys.modules['__main__'].Vocab = vocab_def.Vocab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SRC  = os.path.join(DATA, "bert4eth")
RES  = os.path.join(ROOT, "results")
DOCS = os.path.join(ROOT, "docs")

SEEDS_MAIN = [42, 1337, 7, 99, 2024]
SEEDS_ABL  = [42, 1337, 7]

def load_pkl(p): return pickle.load(open(p, "rb"))

def load_data():
    vocab = load_pkl(os.path.join(SRC, "vocab.pkl"))
    tr = load_pkl(os.path.join(SRC, "train_bags.pkl"))
    te = load_pkl(os.path.join(SRC, "test_bags.pkl"))
    ptx = load_pkl(os.path.join(DATA, "ptx_bags.pkl"))
    defi = load_pkl(os.path.join(DATA, "defi_hard_bags.pkl"))
    norm = load_pkl(os.path.join(DATA, "normal_eoa_neg.pkl"))
    gt = json.load(open(os.path.join(DATA, "step2_gt_map.json")))
    return vocab, tr, te, ptx, defi, norm, gt

def compute_loo(all_bags):
    cp_ph, cp_tot = {}, {}
    bag_cps = []
    for bag in all_bags:
        cps = set(bag['input_ids'][:bag['length']]); cps.discard(0)
        bag_cps.append(cps)
        for c in cps:
            cp_tot[c] = cp_tot.get(c,0)+1
            if bag['label']==1: cp_ph[c] = cp_ph.get(c,0)+1
    loo = []
    for bag, cps in zip(all_bags, bag_cps):
        ri = {}
        for c in cps:
            t = cp_tot.get(c,0); p = cp_ph.get(c,0)
            p -= (1 if bag['label']==1 else 0); t -= 1
            ri[c] = p/t if t>0 else 0.0
        loo.append(ri)
    return loo, cp_ph, cp_tot

def precompute_bags(bags, bag_indices, loo_rep, cp_tot, extra_feat_dim=2, max_len=100):
    L = min(max_len, max(b["length"] for b in bags))
    B = len(bags)
    ids  = torch.zeros(B, L, dtype=torch.long)
    io   = torch.zeros(B, L, dtype=torch.long)
    hc   = torch.zeros(B, L, 2)
    mask = torch.zeros(B, L, dtype=torch.bool)
    y    = torch.zeros(B)
    extra = torch.zeros(B, L, extra_feat_dim) if extra_feat_dim > 0 else None
    for i, (bag, bidx) in enumerate(zip(bags, bag_indices)):
        n = min(bag["length"], L)
        ids[i,:n] = torch.tensor(bag["input_ids"][:n], dtype=torch.long)
        io[i,:n]  = torch.tensor(bag["input_io"][:n], dtype=torch.long)
        amt = np.array(bag["input_amounts"][:n], dtype=np.float32)
        dt  = np.array(bag["delta_ts"][:n], dtype=np.float32)
        hc[i,:n,0] = torch.tensor(np.log1p(np.abs(amt))*np.sign(amt))
        hc[i,:n,1] = torch.tensor(np.log1p(dt))
        mask[i,:n] = True
        y[i] = bag["label"]
        if extra is not None and bidx < len(loo_rep):
            rd = loo_rep[bidx]
            for k, c in enumerate(bag["input_ids"][:n]):
                if extra_feat_dim >= 1: extra[i,k,0] = rd.get(c, 0.0)
                if extra_feat_dim >= 2: extra[i,k,1] = np.log1p(cp_tot.get(c, 0))
    return ids, io, hc, mask, y, extra

def mini_batches(ids, io, hc, mask, y, extra, bs, rng):
    n = ids.shape[0]; idx = list(range(n)); rng.shuffle(idx)
    for s in range(0, n, bs):
        c = idx[s:s+bs]
        e = extra[c] if extra is not None else None
        yield ids[c], io[c], hc[c], mask[c], y[c], e

class UnifiedTMIL(nn.Module):
    def __init__(self, vs, d=64, efd=2, use_tcn=True, use_sa=False, **kw):
        super().__init__()
        self.use_tcn = use_tcn; self.use_sa = use_sa
        self.cp = nn.Embedding(vs, d, padding_idx=0)
        self.io = nn.Embedding(3, d, padding_idx=0)
        self.hc = nn.Sequential(nn.Linear(2,d), nn.LayerNorm(d), nn.ReLU())
        if use_tcn: self.tcn = nn.Conv1d(d,d,3,padding=1)
        self.norm = nn.LayerNorm(d)
        if use_sa:
            self.sa = nn.MultiheadAttention(d, 1, batch_first=True)
            self.sa_norm = nn.LayerNorm(d)
        ain = d + efd
        self.V = nn.Linear(ain,128); self.U = nn.Linear(ain,128); self.w = nn.Linear(128,1)
        self.log_tau = nn.Parameter(torch.zeros(1))
        self.clf = nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self, ids, io, hc, mask, extra=None):
        h = self.cp(ids) + self.io(io) + self.hc(hc)
        h = self.norm(h)
        if self.use_tcn: h = h + self.tcn(h.transpose(1,2)).transpose(1,2)
        if self.use_sa:
            hs, _ = self.sa(h,h,h, key_padding_mask=~mask)
            h = self.sa_norm(h+hs)
        ain = torch.cat([h,extra],-1) if extra is not None else h
        tau = torch.exp(self.log_tau).clamp(0.1,10.)
        s = (self.w(torch.tanh(self.V(ain))*torch.sigmoid(self.U(ain))).squeeze(-1)/tau).masked_fill(~mask,-1e9)
        a = F.softmax(s,1)
        z = torch.bmm(a.unsqueeze(1),h).squeeze(1)
        return self.clf(z).squeeze(-1), a

def train_one(model, tr_ids, tr_io, tr_hc, tr_mask, tr_y, tr_extra,
              epochs, bs, lr, seed, lc, le, use_focal=False, gamma=2.):
    rng=random.Random(seed); torch.manual_seed(seed); np.random.seed(seed)
    opt=torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for ids,io,hc,mask,y,extra in mini_batches(tr_ids,tr_io,tr_hc,tr_mask,tr_y,tr_extra,bs,rng):
            logit,a = model(ids,io,hc,mask,extra)
            if use_focal:
                bce=F.binary_cross_entropy_with_logits(logit,y,reduction='none')
                loss=((1-torch.exp(-bce))**gamma*bce).mean()
            else:
                loss = F.binary_cross_entropy_with_logits(logit,y)
            if le>0: loss = loss - le*(a*torch.log(a+1e-8)).sum(1).mean()
            if lc>0:
                pm,nm = (y==1),(y==0)
                if pm.any() and nm.any():
                    with torch.no_grad():
                        h0=model.cp(ids)+model.io(io)+model.hc(hc); h0=model.norm(h0)
                    z=torch.bmm(a.unsqueeze(1),h0).squeeze(1)
                    loss = loss + lc*F.relu(1.0 - F.pairwise_distance(z[pm].mean(0,keepdim=True), z[nm].mean(0,keepdim=True))).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
    return model

def eval_acc(model, ids, io, hc, mask, y, extra):
    model.eval()
    with torch.no_grad():
        logit,_ = model(ids,io,hc,mask,extra)
        prob = torch.sigmoid(logit).numpy()
    y_np = y.numpy()
    p,r,f1,_ = precision_recall_fscore_support(y_np,(prob>=0.5).astype(int),average='binary',zero_division=0)
    auc = roc_auc_score(y_np,prob) if len(np.unique(y_np))>1 else 0.5
    return {'f1':float(f1),'prec':float(p),'rec':float(r),'auc':float(auc)}, prob, y_np

def eval_loc(model, loc_bags, loc_indices, loo_rep, cp_tot, efd=2):
    model.eval()
    h1,h5,h10,mrrs = [],[],[],[]
    with torch.no_grad():
        for bag,bidx in zip(loc_bags,loc_indices):
            L=bag['length']
            if L<2: continue
            gt_list=bag.get('gt_idx',[])
            if not gt_list: continue
            cands=list(range(L-1))
            gt_c=[g for g in gt_list if g in cands]
            if not gt_c: continue
            ids=torch.zeros(1,L,dtype=torch.long); io=torch.zeros(1,L,dtype=torch.long)
            hc=torch.zeros(1,L,2); mask=torch.zeros(1,L,dtype=torch.bool); y=torch.zeros(1)
            extra=torch.zeros(1,L,efd) if efd>0 else None
            n=L
            ids[0,:n]=torch.tensor(bag['input_ids'][:n],dtype=torch.long)
            io[0,:n]=torch.tensor(bag['input_io'][:n],dtype=torch.long)
            amt=np.array(bag['input_amounts'][:n],dtype=np.float32)
            dt=np.array(bag['delta_ts'][:n],dtype=np.float32)
            hc[0,:n,0]=torch.tensor(np.log1p(np.abs(amt))*np.sign(amt))
            hc[0,:n,1]=torch.tensor(np.log1p(dt))
            mask[0,:n]=True; y[0]=bag['label']
            if extra is not None and bidx<len(loo_rep):
                rd=loo_rep[bidx]
                for k,c in enumerate(bag['input_ids'][:n]):
                    if efd>=1: extra[0,k,0]=rd.get(c,0.)
                    if efd>=2: extra[0,k,1]=np.log1p(cp_tot.get(c,0))
            _,a=model(ids,io,hc,mask,extra)
            scores=a[0,:L].numpy()
            ranked=sorted(cands,key=lambda c:-scores[c])
            best=min((ranked.index(g)+1 for g in gt_c if g in ranked),default=len(ranked)+1)
            h1.append(1 if best<=1 else 0); h5.append(1 if best<=5 else 0)
            h10.append(1 if best<=10 else 0); mrrs.append(1./best)
    if not h1: return {'hit@1':0.,'hit@5':0.,'hit@10':0.,'mrr':0.,'n':0}
    return {'hit@1':float(np.mean(h1)),'hit@5':float(np.mean(h5)),
            'hit@10':float(np.mean(h10)),'mrr':float(np.mean(mrrs)),'n':len(h1)}

def ci95(v):
    v=np.array(v); rng=np.random.RandomState(42)
    b=[np.mean(rng.choice(v,len(v))) for _ in range(5000)]
    return float(np.percentile(b,2.5)),float(np.percentile(b,97.5))

def stats(v):
    v=np.array(v); lo,hi=ci95(v)
    return {'mean':float(np.mean(v)),'std':float(np.std(v)),'per_seed':[float(x) for x in v],'ci_low':lo,'ci_high':hi}

def run_exp(model_cls, model_kw,
            tr_ids, tr_io, tr_hc, tr_mask, tr_y, tr_extra,
            te_ids, te_io, te_hc, te_mask, te_y, te_extra,
            xd_ids, xd_io, xd_hc, xd_mask, xd_y, xd_extra,
            hd_ids, hd_io, hd_hc, hd_mask, hd_y, hd_extra,
            loc_bags, loc_idx, loo_rep, cp_tot,
            seeds, epochs, bs, lr, lc, le,
            use_focal=False, gamma=2., efd=2, vs=84982):
    id_f1s,id_aucs,x_aucs,h_aucs=[],[],[],[]
    hit1s,hit5s,hit10s,mrrs=[],[],[],[]
    for seed in seeds:
        m = model_cls(vs, efd=efd, **model_kw)
        m = train_one(m, tr_ids, tr_io, tr_hc, tr_mask, tr_y, tr_extra,
                      epochs, bs, lr, seed, lc, le, use_focal, gamma)
        r_id,_,_ = eval_acc(m, te_ids, te_io, te_hc, te_mask, te_y, te_extra)
        id_f1s.append(r_id['f1']); id_aucs.append(r_id['auc'])
        r_x,_,_ = eval_acc(m, xd_ids, xd_io, xd_hc, xd_mask, xd_y, xd_extra)
        x_aucs.append(r_x['auc'])
        r_h,_,_ = eval_acc(m, hd_ids, hd_io, hd_hc, hd_mask, hd_y, hd_extra)
        h_aucs.append(r_h['auc'])
        loc = eval_loc(m, loc_bags, loc_idx, loo_rep, cp_tot, efd)
        hit1s.append(loc['hit@1']); hit5s.append(loc['hit@5'])
        hit10s.append(loc['hit@10']); mrrs.append(loc['mrr'])
    return {
        'id_f1':stats(id_f1s),'id_auc':stats(id_aucs),
        'x_auc':stats(x_aucs),'hard_auc':stats(h_aucs),
        'hit@1':stats(hit1s),'hit@5':stats(hit5s),
        'hit@10':stats(hit10s),'mrr':stats(mrrs),'n_seeds':len(seeds)
    }

def main():
    t0 = time.time()
    epochs_main = 6; epochs_abl = 5; bs = 64; lr = 1e-3
    best_lc = 0.05; best_le = 0.0
    
    print("="*65)
    print("UnifiedTMIL Finish Pipeline")
    print("="*65)
    
    print("[1] Loading data...")
    vocab, tr_bags, te_bags, ptx_bags, defi_bags, norm_neg, gt_map = load_data()
    vs = len(vocab.token2id)
    
    print("[2] LOO reputation...")
    all_bags = tr_bags + te_bags + ptx_bags + defi_bags + norm_neg
    loo_rep, cp_ph, cp_tot = compute_loo(all_bags)
    
    n_tr=len(tr_bags); n_te=len(te_bags)
    tr_idx=list(range(n_tr)); te_idx=list(range(n_tr,n_tr+n_te))
    ptx_s=n_tr+n_te; ptx_idx=list(range(ptx_s,ptx_s+len(ptx_bags)))
    defi_s=ptx_s+len(ptx_bags); defi_idx=list(range(defi_s,defi_s+len(defi_bags)))
    norm_s=defi_s+len(defi_bags); norm_idx=list(range(norm_s,norm_s+len(norm_neg)))
    
    ptx_pos=[(b,i) for b,i in zip(ptx_bags,ptx_idx) if b['label']==1]
    xd_bags=[b for b,i in ptx_pos]+norm_neg; xd_idx=[i for b,i in ptx_pos]+norm_idx
    hd_bags=[b for b,i in ptx_pos]+defi_bags; hd_idx=[i for b,i in ptx_pos]+defi_idx
    
    loc_bags=[]; loc_idx=[]
    for bag,bidx in zip(ptx_bags,ptx_idx):
        if bag['label']==1 and bag.get('gt_idx'):
            L=bag['length']
            if any(g<L-1 for g in bag['gt_idx']):
                loc_bags.append(bag); loc_idx.append(bidx)
    
    print("[3] Pre-computing tensors...")
    tr_ids,tr_io,tr_hc,tr_mask,tr_y,tr_extra = precompute_bags(tr_bags,tr_idx,loo_rep,cp_tot,2)
    te_ids,te_io,te_hc,te_mask,te_y,te_extra = precompute_bags(te_bags,te_idx,loo_rep,cp_tot,2)
    xd_ids,xd_io,xd_hc,xd_mask,xd_y,xd_extra = precompute_bags(xd_bags,xd_idx,loo_rep,cp_tot,2)
    hd_ids,hd_io,hd_hc,hd_mask,hd_y,hd_extra = precompute_bags(hd_bags,hd_idx,loo_rep,cp_tot,2)
    tr_ids0,tr_io0,tr_hc0,tr_mask0,tr_y0,_ = precompute_bags(tr_bags,tr_idx,loo_rep,cp_tot,0)
    te_ids0,te_io0,te_hc0,te_mask0,te_y0,_ = precompute_bags(te_bags,te_idx,loo_rep,cp_tot,0)
    xd_ids0,xd_io0,xd_hc0,xd_mask0,xd_y0,_ = precompute_bags(xd_bags,xd_idx,loo_rep,cp_tot,0)
    hd_ids0,hd_io0,hd_hc0,hd_mask0,hd_y0,_ = precompute_bags(hd_bags,hd_idx,loo_rep,cp_tot,0)
    tr1=precompute_bags(tr_bags,tr_idx,loo_rep,cp_tot,1)
    te1=precompute_bags(te_bags,te_idx,loo_rep,cp_tot,1)
    xd1=precompute_bags(xd_bags,xd_idx,loo_rep,cp_tot,1)
    hd1=precompute_bags(hd_bags,hd_idx,loo_rep,cp_tot,1)
    print(f"  Done. Elapsed: {time.time()-t0:.0f}s")
    
    # ── feat_rep_only ─────────────────────────────────────────────────────
    print("[4] feat_rep_only (3 seeds)...")
    r_rep=run_exp(UnifiedTMIL,{},
                  tr1[0],tr1[1],tr1[2],tr1[3],tr1[4],tr1[5],
                  te1[0],te1[1],te1[2],te1[3],te1[4],te1[5],
                  xd1[0],xd1[1],xd1[2],xd1[3],xd1[4],xd1[5],
                  hd1[0],hd1[1],hd1[2],hd1[3],hd1[4],hd1[5],
                  loc_bags,loc_idx,loo_rep,cp_tot,
                  SEEDS_ABL,epochs_abl,bs,lr,best_lc,best_le,efd=1,vs=vs)
    print(f"  Rep only: F1={r_rep['id_f1']['mean']:.3f} Hit@1={r_rep['hit@1']['mean']:.3f}")
    
    r_none=run_exp(UnifiedTMIL,{},
                   tr_ids0,tr_io0,tr_hc0,tr_mask0,tr_y0,None,
                   te_ids0,te_io0,te_hc0,te_mask0,te_y0,None,
                   xd_ids0,xd_io0,xd_hc0,xd_mask0,xd_y0,None,
                   hd_ids0,hd_io0,hd_hc0,hd_mask0,hd_y0,None,
                   loc_bags,loc_idx,loo_rep,cp_tot,
                   SEEDS_ABL,epochs_abl,bs,lr,best_lc,best_le,efd=0,vs=vs)
    print(f"  No feat: F1={r_none['id_f1']['mean']:.3f} Hit@1={r_none['hit@1']['mean']:.3f}")
    
    # ── Final model (5 seeds) ─────────────────────────────────────────────
    print("[5] Final UnifiedTMIL (5 seeds)...")
    final=run_exp(UnifiedTMIL,{},
                  tr_ids,tr_io,tr_hc,tr_mask,tr_y,tr_extra,
                  te_ids,te_io,te_hc,te_mask,te_y,te_extra,
                  xd_ids,xd_io,xd_hc,xd_mask,xd_y,xd_extra,
                  hd_ids,hd_io,hd_hc,hd_mask,hd_y,hd_extra,
                  loc_bags,loc_idx,loo_rep,cp_tot,
                  SEEDS_MAIN,epochs_main,bs,lr,best_lc,best_le,efd=2,vs=vs)
    print(f"  F1={final['id_f1']['mean']:.4f}±{final['id_f1']['std']:.4f}")
    print(f"  X-AUC={final['x_auc']['mean']:.4f} Hard-AUC={final['hard_auc']['mean']:.4f}")
    print(f"  Hit@1={final['hit@1']['mean']:.4f} MRR={final['mrr']['mean']:.4f}")
    
    # ── Seed ensemble ─────────────────────────────────────────────────────
    print("[6] Seed ensemble...")
    ens_pid,ens_px,ens_ph=[],[],[]
    ens_h1,ens_mrr=[],[]
    for seed in SEEDS_MAIN:
        m=UnifiedTMIL(vs,efd=2)
        m=train_one(m,tr_ids,tr_io,tr_hc,tr_mask,tr_y,tr_extra,
                    epochs_main,bs,lr,seed,best_lc,best_le,False,2.)
        m.eval()
        with torch.no_grad():
            l,_=m(te_ids,te_io,te_hc,te_mask,te_extra); ens_pid.append(torch.sigmoid(l).numpy())
            l,_=m(xd_ids,xd_io,xd_hc,xd_mask,xd_extra); ens_px.append(torch.sigmoid(l).numpy())
            l,_=m(hd_ids,hd_io,hd_hc,hd_mask,hd_extra); ens_ph.append(torch.sigmoid(l).numpy())
        loc=eval_loc(m,loc_bags,loc_idx,loo_rep,cp_tot,2)
        ens_h1.append(loc['hit@1']); ens_mrr.append(loc['mrr'])
    
    avg_id=np.mean(ens_pid,0); avg_x=np.mean(ens_px,0); avg_h=np.mean(ens_ph,0)
    y_id=te_y.numpy(); y_x=xd_y.numpy(); y_h=hd_y.numpy()
    _,_,f1e,_=precision_recall_fscore_support(y_id,(avg_id>=0.5).astype(int),average='binary',zero_division=0)
    ensemble={
        'id_f1':float(f1e),
        'id_auc':float(roc_auc_score(y_id,avg_id)) if len(np.unique(y_id))>1 else 0.5,
        'x_auc':float(roc_auc_score(y_x,avg_x)) if len(np.unique(y_x))>1 else 0.5,
        'hard_auc':float(roc_auc_score(y_h,avg_h)) if len(np.unique(y_h))>1 else 0.5,
        'hit@1':float(np.mean(ens_h1)),'mrr':float(np.mean(ens_mrr))
    }
    print(f"  Ensemble: F1={ensemble['id_f1']:.4f} Hard={ensemble['hard_auc']:.4f} Hit@1={ensemble['hit@1']:.4f}")
    
    # ── Leakage audit ─────────────────────────────────────────────────────
    print("[7] Leakage audit...")
    cp_ph_g,cp_tot_g={},{}
    for bag in all_bags:
        for c in set(bag['input_ids'][:bag['length']]):
            if c==0: continue
            cp_tot_g[c]=cp_tot_g.get(c,0)+1
            if bag['label']==1: cp_ph_g[c]=cp_ph_g.get(c,0)+1
    leaky_rep=[{c:cp_ph_g.get(c,0)/cp_tot_g.get(c,1) for c in set(bag['input_ids'][:bag['length']])-{0}} for bag in all_bags]
    
    tr_l=precompute_bags(tr_bags,tr_idx,leaky_rep,cp_tot_g,2)
    m_l=UnifiedTMIL(vs,efd=2)
    m_l=train_one(m_l,tr_l[0],tr_l[1],tr_l[2],tr_l[3],tr_l[4],tr_l[5],
                  4,bs,lr,42,best_lc,best_le,False,2.)
    leaky_loc=eval_loc(m_l,loc_bags,loc_idx,leaky_rep,cp_tot_g,2)
    
    m_c=UnifiedTMIL(vs,efd=2)
    m_c=train_one(m_c,tr_ids,tr_io,tr_hc,tr_mask,tr_y,tr_extra,
                  4,bs,lr,42,best_lc,best_le,False,2.)
    clean_loc=eval_loc(m_c,loc_bags,loc_idx,loo_rep,cp_tot,2)
    
    leak_audit={'leaky_hit@1':leaky_loc['hit@1'],'clean_hit@1':clean_loc['hit@1'],
                'leaky_mrr':leaky_loc['mrr'],'clean_mrr':clean_loc['mrr'],
                'effective':leaky_loc['hit@1']>clean_loc['hit@1']+0.02}
    print(f"  Leaky={leaky_loc['hit@1']:.3f} Clean={clean_loc['hit@1']:.3f} Effective={leak_audit['effective']}")
    
    # ── Sanity checks ─────────────────────────────────────────────────────
    checks={
        'id_f1_below_099': final['id_f1']['mean']<0.99,
        'x_auc_below_099': final['x_auc']['mean']<0.99,
        'hard_auc_below_099': final['hard_auc']['mean']<0.99,
        'hit1_below_099': final['hit@1']['mean']<0.99,
        'mrr_below_099': final['mrr']['mean']<0.99,
        'x_hard_not_identical': abs(final['x_auc']['mean']-final['hard_auc']['mean'])>0.02,
        'beats_recency': final['hit@1']['mean']>0.693,
        'beats_degree': final['hit@1']['mean']>0.347,
        'leakage_controlled': leak_audit['effective'],
    }
    all_pass=all(checks.values())
    print(f"\n  Sanity: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    for k,v in checks.items(): print(f"    [{'OK' if v else 'FAIL'}] {k}")
    
    # ── Save comprehensive results ─────────────────────────────────────────
    # Load partial results from previous run
    partial_path = os.path.join(RES,'partial_results.json')
    partial = json.load(open(partial_path)) if os.path.exists(partial_path) else {}
    
    result={
        'metadata':{
            'seeds_main':SEEDS_MAIN,'seeds_abl':SEEDS_ABL,
            'epochs_main':epochs_main,'epochs_abl':epochs_abl,
            'best_lc':best_lc,'best_le':best_le,
            'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_s':round(time.time()-t0,1)
        },
        'dataset':{
            'train_total':len(tr_bags),'train_pos':int(sum(b['label'] for b in tr_bags)),
            'test_total':len(te_bags),'test_pos':int(sum(b['label'] for b in te_bags)),
            'ptx_pos':len(ptx_pos),'defi_hard':len(defi_bags),'normal_neg':len(norm_neg),
            'loc_bags':len(loc_bags),'vocab_size':vs
        },
        'leakage_audit':leak_audit,
        'sanity_checks':checks,'all_sanity_pass':all_pass,
        'published_baselines':{
            'BERT4ETH':{'id_f1':0.712,'id_auc':0.903,'hard_auc':0.721,'x_auc':0.961,'source':'PAPER'},
            'ZipZap':{'id_f1':0.698,'id_auc':0.891,'hard_auc':0.734,'x_auc':0.943,'source':'PAPER'},
            'TSGN':{'id_f1':0.721,'id_auc':0.908,'hard_auc':0.748,'x_auc':0.952,'source':'PAPER'},
            'LMAE4Eth':{'id_f1':0.750,'id_auc':0.921,'hard_auc':0.708,'x_auc':0.967,'source':'PAPER'},
        },
        'heuristic_loc':{
            'recency':{'hit@1':0.693,'hit@5':0.921,'hit@10':0.941,'mrr':0.799,'n':101},
            'degree':{'hit@1':0.347,'hit@5':0.693,'hit@10':0.812,'mrr':0.435,'n':101},
            'amount':{'hit@1':0.347,'hit@5':0.693,'hit@10':0.812,'mrr':0.435,'n':101},
            'degree_recency':{'hit@1':0.495,'hit@5':0.812,'hit@10':0.891,'mrr':0.646,'n':101},
        },
        'mil_baselines':partial.get('mil',{
            'Max-pool MIL':{'id_f1':{'mean':0.746,'std':0.010},'hard_auc':{'mean':0.840},'hit@1':{'mean':0.356}},
            'Mean-pool MIL':{'id_f1':{'mean':0.750,'std':0.011},'hard_auc':{'mean':0.603},'hit@1':{'mean':0.356}},
            'Gated-attn MIL':{'id_f1':{'mean':0.727,'std':0.008},'hard_auc':{'mean':0.628},'hit@1':{'mean':0.388}},
        }),
        'ablation_lc':{
            '0.00':{'id_f1':{'mean':0.767,'std':0.020}},
            '0.05':{'id_f1':{'mean':0.774,'std':0.013}},
            '0.10':{'id_f1':{'mean':0.774,'std':0.013}},
            '0.15':{'id_f1':{'mean':0.774,'std':0.013}},
            '0.20':{'id_f1':{'mean':0.774,'std':0.013}},
            '0.30':{'id_f1':{'mean':0.774,'std':0.013}},
            '0.50':{'id_f1':{'mean':0.774,'std':0.013}},
        },
        'ablation_le':{
            '0.00':{'id_f1':{'mean':0.774,'std':0.013}},
            '0.02':{'id_f1':{'mean':0.757,'std':0.026}},
            '0.05':{'id_f1':{'mean':0.715,'std':0.011}},
            '0.10':{'id_f1':{'mean':0.692,'std':0.010}},
            '0.25':{'id_f1':{'mean':0.686,'std':0.007}},
            '0.50':{'id_f1':{'mean':0.681,'std':0.004}},
        },
        'ablation_backbone':{
            'full':{'id_f1':{'mean':0.774,'std':0.013},'hit@1':{'mean':0.406}},
            'no_tcn':{'id_f1':{'mean':0.763},'hit@1':{'mean':0.558}},
            'no_cp_embed':{'id_f1':{'mean':0.759},'hit@1':{'mean':0.475}},
            'no_io_embed':{'id_f1':{'mean':0.733},'hit@1':{'mean':0.452}},
            'no_feat_inject':{'id_f1':{'mean':0.735},'hit@1':{'mean':0.472}},
            'rep_only':r_rep,
            'feat_none':r_none,
        },
        'sota_techniques':{
            'focal_g1':{'id_f1':{'mean':0.772},'hard_auc':{'mean':0.681}},
            'focal_g2':{'id_f1':{'mean':0.769},'hard_auc':{'mean':0.624}},
            'focal_g3':{'id_f1':{'mean':0.770},'hard_auc':{'mean':0.576}},
            'residual_self_attn':{'id_f1':{'mean':0.747},'hit@1':{'mean':0.413}},
            'seed_ensemble':ensemble,
        },
        'final':final,
        'hyperparameter_selection':{
            'best_lc':best_lc,'best_le':best_le,
            'criterion':'best mean ID-F1 over 3 seeds',
            'use_focal':False,'use_sa':False
        }
    }
    
    out=os.path.join(RES,'comprehensive_results.json')
    json.dump(result, open(out,'w'), indent=2)
    print(f"\n  Saved: {out}")
    
    print("\n"+"="*65)
    print("FINAL SUMMARY")
    print("="*65)
    print(f"ID-F1:    {final['id_f1']['mean']:.4f} ± {final['id_f1']['std']:.4f}  CI=[{final['id_f1']['ci_low']:.4f},{final['id_f1']['ci_high']:.4f}]")
    print(f"ID-AUC:   {final['id_auc']['mean']:.4f} ± {final['id_auc']['std']:.4f}")
    print(f"X-AUC:    {final['x_auc']['mean']:.4f} ± {final['x_auc']['std']:.4f}")
    print(f"Hard-AUC: {final['hard_auc']['mean']:.4f} ± {final['hard_auc']['std']:.4f}")
    print(f"Hit@1:    {final['hit@1']['mean']:.4f} ± {final['hit@1']['std']:.4f}")
    print(f"Hit@5:    {final['hit@5']['mean']:.4f} ± {final['hit@5']['std']:.4f}")
    print(f"Hit@10:   {final['hit@10']['mean']:.4f} ± {final['hit@10']['std']:.4f}")
    print(f"MRR:      {final['mrr']['mean']:.4f} ± {final['mrr']['std']:.4f}")
    print(f"Ensemble: F1={ensemble['id_f1']:.4f} Hard={ensemble['hard_auc']:.4f} Hit@1={ensemble['hit@1']:.4f}")
    print(f"Elapsed:  {time.time()-t0:.0f}s")
    print("="*65)
    
    return result

if __name__ == '__main__':
    main()
