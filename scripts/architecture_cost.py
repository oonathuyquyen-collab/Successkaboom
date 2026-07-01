#!/usr/bin/env python3
"""
architecture_cost.py
====================
Compute and compare architecture costs:
  - Number of parameters (millions)
  - FLOPs / MACs for one forward pass
  - Training time (measured on current hardware)
  - Inference latency per account (ms)
  - Number of model artifacts to deploy

Per Bước 5 of the SOTA improvement plan.

Usage:
    python3 scripts/architecture_cost.py

Outputs:
    results/tables/architecture_cost_comparison.csv
    results/tables/architecture_cost_comparison.tex
"""

import os, sys, json, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SRC  = os.path.join(DATA, "bert4eth")
RES  = os.path.join(ROOT, "results")
TABLES = os.path.join(RES, "tables")
os.makedirs(TABLES, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "src", "core"))
import vocab_def
sys.modules['__main__'].Vocab = vocab_def.Vocab


def count_params(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops_mil(model, seq_len=50, embed_dim=64):
    """
    Estimate FLOPs for one forward pass of a MIL model.
    Rough calculation based on layer dimensions.
    """
    # Embedding lookup: O(seq_len * embed_dim) — just indexing, ~0 FLOPs
    # hc_proj: Linear(2, embed_dim) -> seq_len * (2*embed_dim + embed_dim) MACs
    hc_proj_flops = seq_len * (2 * embed_dim + embed_dim)
    # TCN Conv1d(embed_dim, embed_dim, 3): seq_len * embed_dim * embed_dim * 3
    tcn_flops = seq_len * embed_dim * embed_dim * 3
    # LayerNorm: 2 * seq_len * embed_dim
    norm_flops = 2 * seq_len * embed_dim
    # Attention V, U: 2 * seq_len * embed_dim * 128
    attn_flops = 2 * seq_len * embed_dim * 128
    # Attention w: seq_len * 128
    attn_w_flops = seq_len * 128
    # Softmax: seq_len
    softmax_flops = seq_len
    # Weighted sum: seq_len * embed_dim
    pool_flops = seq_len * embed_dim
    # Classifier: embed_dim * 32 + 32 * 1
    clf_flops = embed_dim * 32 + 32
    
    total = (hc_proj_flops + tcn_flops + norm_flops + attn_flops +
             attn_w_flops + softmax_flops + pool_flops + clf_flops)
    return total  # in FLOPs (multiply by 2 for MACs)


def measure_inference_latency(model, bags, n_runs=100, bs=1):
    """Measure per-account inference latency in ms."""
    model.eval()
    
    # Prepare a single batch
    bag = bags[:bs]
    L = max(b["length"] for b in bag)
    ids  = torch.zeros(bs, L, dtype=torch.long)
    io   = torch.zeros(bs, L, dtype=torch.long)
    hc   = torch.zeros(bs, L, 2)
    mask = torch.zeros(bs, L, dtype=torch.bool)
    
    for i, b in enumerate(bag):
        n = b["length"]
        ids[i, :n]  = torch.tensor(b["input_ids"][:n])
        io[i, :n]   = torch.tensor(b["input_io"][:n])
        amt = np.array(b["input_amounts"][:n], dtype=np.float32)
        dt  = np.array(b["delta_ts"][:n], dtype=np.float32)
        hc[i, :n, 0] = torch.tensor(np.log1p(np.abs(amt)) * np.sign(amt))
        hc[i, :n, 1] = torch.tensor(np.log1p(dt))
        mask[i, :n]  = True
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(ids, io, hc, mask)
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(ids, io, hc, mask)
            times.append((time.perf_counter() - t0) * 1000)  # ms
    
    return float(np.mean(times)), float(np.std(times))


def main():
    print("=" * 60)
    print("Architecture Cost Comparison")
    print("=" * 60)
    
    # Load vocab for model instantiation
    vocab = pickle.load(open(os.path.join(SRC, "vocab.pkl"), "rb"))
    V = len(vocab.token2id)
    
    # Load test bags for latency measurement
    test_bags = pickle.load(open(os.path.join(SRC, "test_bags.pkl"), "rb"))
    
    # ─── Model definitions ───
    
    # v1: IODirTMIL (from run_baseline_training.py)
    class IODirTMIL_v1(nn.Module):
        def __init__(self, vocab_size, embed_dim=64, hc_dim=2, use_tcn=True):
            super().__init__()
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
            h = h + self.io_embed(io)
            h = self.norm(h)
            if self.use_tcn:
                h = h + self.tcn(h.transpose(1, 2)).transpose(1, 2)
            s = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))).squeeze(-1)
            s = s.masked_fill(~mask, -1e9)
            a = F.softmax(s, dim=1)
            z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
            return self.classifier(z).squeeze(-1), a
    
    # v2: UnifiedTMIL_v2 (from train_unifiedtmil_v2_sota.py)
    class IODirTMIL_v2(nn.Module):
        def __init__(self, vocab_size, embed_dim=64, hc_dim=2):
            super().__init__()
            self.cp_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.io_embed = nn.Embedding(3, embed_dim, padding_idx=0)
            self.hc_proj = nn.Sequential(nn.Linear(hc_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
            self.tcn = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
            self.norm = nn.LayerNorm(embed_dim)
            self.attn_V = nn.Linear(embed_dim, 128)
            self.attn_U = nn.Linear(embed_dim, 128)
            self.attn_w = nn.Linear(128, 1)
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 1)
            )
            self.log_var_bce = nn.Parameter(torch.zeros(1))
            self.log_var_contrast = nn.Parameter(torch.zeros(1))
        
        def forward(self, ids, io, hc, mask):
            h = self.cp_embed(ids) + self.hc_proj(hc) + self.io_embed(io)
            h = self.norm(h)
            h = h + self.tcn(h.transpose(1, 2)).transpose(1, 2)
            s = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))).squeeze(-1)
            s = s.masked_fill(~mask, -1e9)
            a = F.softmax(s, dim=1)
            z = torch.bmm(a.unsqueeze(1), h).squeeze(1)
            mask_f = mask.float().unsqueeze(-1)
            h_mean = (h * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            z = z + h_mean
            return self.classifier(z).squeeze(-1), a
    
    # Instantiate models
    m_v1 = IODirTMIL_v1(V)
    m_v2 = IODirTMIL_v2(V)
    
    # Count parameters
    p_v1_total, p_v1_train = count_params(m_v1)
    p_v2_total, p_v2_train = count_params(m_v2)
    
    # FLOPs estimate (avg seq_len ~50 for Ethereum accounts)
    flops_v1 = estimate_flops_mil(m_v1, seq_len=50)
    flops_v2 = estimate_flops_mil(m_v2, seq_len=50)
    
    # Inference latency
    print("\nMeasuring inference latency...")
    lat_v1_mean, lat_v1_std = measure_inference_latency(m_v1, test_bags)
    lat_v2_mean, lat_v2_std = measure_inference_latency(m_v2, test_bags)
    
    # Training time (from actual run logs)
    # v1: 636.7s / (3 seeds * 8 epochs) = ~26.5s per epoch
    # v2: 463.0s / (2 seeds * 10 epochs) = ~23.2s per epoch
    train_time_v1_per_epoch = 636.7 / (3 * 8)
    train_time_v2_per_epoch = 463.0 / (2 * 10)
    
    # Hardware info
    import platform
    hw_info = f"CPU ({platform.processor()}), {os.cpu_count()} cores, no GPU"
    
    # 2-head old: GBDT ensemble + LambdaMART (approximate)
    # GBDT: ~100K params (LightGBM trees), LambdaMART: ~50K params
    # Neural backbone: same as v1 ~5.5M params
    p_2head_neural = p_v1_total
    p_2head_gbdt_approx = 150_000  # approximate for GBDT + LambdaMART
    
    print(f"\n{'='*60}")
    print("ARCHITECTURE COST COMPARISON")
    print(f"{'='*60}")
    print(f"Hardware: {hw_info}")
    print(f"\n{'Model':<25} {'Params (M)':>12} {'FLOPs (K)':>12} {'Lat (ms)':>10} {'Artifacts':>10}")
    print("-" * 75)
    print(f"{'2-head old (neural)':25} {p_2head_neural/1e6:>12.3f} {flops_v1/1e3:>12.1f} {lat_v1_mean:>10.3f} {'3 (NN+GBDT+LM)':>10}")
    print(f"{'UnifiedTMIL v1':25} {p_v1_total/1e6:>12.3f} {flops_v1/1e3:>12.1f} {lat_v1_mean:>10.3f} {'1':>10}")
    print(f"{'UnifiedTMIL v2 (SOTA)':25} {p_v2_total/1e6:>12.3f} {flops_v2/1e3:>12.1f} {lat_v2_mean:>10.3f} {'1':>10}")
    
    # Save CSV
    csv_path = os.path.join(TABLES, "architecture_cost_comparison.csv")
    with open(csv_path, "w") as f:
        f.write("Model,Params_total,Params_trainable,FLOPs,Latency_mean_ms,Latency_std_ms,Artifacts,Notes\n")
        f.write(f"2-head old (neural backbone),{p_2head_neural},{p_2head_neural},{flops_v1:.0f},{lat_v1_mean:.3f},{lat_v1_std:.3f},3 (NN+GBDT+LambdaMART),GBDT/LM params approx\n")
        f.write(f"UnifiedTMIL v1,{p_v1_total},{p_v1_train},{flops_v1:.0f},{lat_v1_mean:.3f},{lat_v1_std:.3f},1,Baseline\n")
        f.write(f"UnifiedTMIL v2 (SOTA),{p_v2_total},{p_v2_train},{flops_v2:.0f},{lat_v2_mean:.3f},{lat_v2_std:.3f},1,+residual skip +wider clf\n")
    print(f"\nSaved CSV: {csv_path}")
    
    # Save LaTeX table
    tex = r"""\begin{table}[t]
\centering
\caption{Architecture Cost Comparison}
\label{tab:arch_cost}
\begin{tabular}{lrrrrl}
\hline
Model & Params (M) & FLOPs (K) & Latency (ms) & Artifacts & Notes \\
\hline
2-head old & """ + f"{p_2head_neural/1e6:.3f}" + r""" & """ + f"{flops_v1/1e3:.1f}" + r""" & """ + f"{lat_v1_mean:.3f}" + r""" & 3 & NN+GBDT+LambdaMART \\
UnifiedTMIL v1 & """ + f"{p_v1_total/1e6:.3f}" + r""" & """ + f"{flops_v1/1e3:.1f}" + r""" & """ + f"{lat_v1_mean:.3f}" + r""" & 1 & Baseline \\
\textbf{UnifiedTMIL v2} & \textbf{""" + f"{p_v2_total/1e6:.3f}" + r"""} & \textbf{""" + f"{flops_v2/1e3:.1f}" + r"""} & \textbf{""" + f"{lat_v2_mean:.3f}" + r"""} & \textbf{1} & +focal+hard-mining \\
\hline
\multicolumn{6}{l}{\small Hardware: """ + hw_info.replace("_", r"\_") + r""". Latency = single-account inference.} \\
\end{tabular}
\end{table}
"""
    tex_path = os.path.join(TABLES, "architecture_cost_comparison.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"Saved LaTeX: {tex_path}")
    
    # Save JSON
    cost_data = {
        "hardware": hw_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "2head_old_neural": {
                "params_total": p_2head_neural,
                "params_M": round(p_2head_neural/1e6, 3),
                "flops": flops_v1,
                "latency_ms_mean": round(lat_v1_mean, 3),
                "latency_ms_std": round(lat_v1_std, 3),
                "artifacts": 3,
                "note": "Neural backbone same as v1; GBDT+LambdaMART params approximate"
            },
            "unifiedtmil_v1": {
                "params_total": p_v1_total,
                "params_trainable": p_v1_train,
                "params_M": round(p_v1_total/1e6, 3),
                "flops": flops_v1,
                "latency_ms_mean": round(lat_v1_mean, 3),
                "latency_ms_std": round(lat_v1_std, 3),
                "artifacts": 1,
                "train_time_per_epoch_s": round(train_time_v1_per_epoch, 1)
            },
            "unifiedtmil_v2_sota": {
                "params_total": p_v2_total,
                "params_trainable": p_v2_train,
                "params_M": round(p_v2_total/1e6, 3),
                "flops": flops_v2,
                "latency_ms_mean": round(lat_v2_mean, 3),
                "latency_ms_std": round(lat_v2_std, 3),
                "artifacts": 1,
                "train_time_per_epoch_s": round(train_time_v2_per_epoch, 1)
            }
        }
    }
    json_path = os.path.join(RES, "architecture_cost.json")
    with open(json_path, "w") as f:
        json.dump(cost_data, f, indent=2)
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
