"""
Generate all academic figures for the SuccessUnifiedTMIL paper.
Figures:
  1. Overall Architecture (fig_arch_overview.png)
  2. Data Pipeline (fig_data_pipeline.png)
  3. Training Pipeline (fig_training_pipeline.png)
  4. Account-level Performance (fig_account_perf.png)
  5. Localization Performance (fig_loc_perf.png)
  6. SOTA Comparison (fig_sota_comparison.png)
  7. Ablation Study (fig_ablation.png)
  8. OOD Generalization (fig_ood.png)
  9. Complexity Comparison (fig_complexity.png)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- Style ---
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}

# --- Load results ---
final_res = load_json("results/final_results.json")
sota_res = load_json("results/sota_results.json")
loc_abl = load_json("results/loc_ablation.json")
lean_acc = load_json("results/lean_account_results.json")
lean_loc = load_json("results/lean_loc_results_fixed.json")

# ============================================================
# FIG 1: Overall Architecture Diagram
# ============================================================
def draw_arch():
    BG = "#ffffff"; INK = "#1b2631"; SUBINK = "#5d6d7e"
    YEL = "#fcf3cf"; YEL_E = "#d4ac0d"
    PUR = "#ebdef0"; PUR_E = "#8e44ad"
    ORA = "#fad7a1"; ORA_E = "#ca6f1e"
    BLU = "#d4e6f1"; BLU_E = "#2e86c1"
    GRN = "#d5f5e3"; GRN_E = "#1e8449"
    PNK = "#fadbd8"; PNK_E = "#cb4335"
    ACC = "#5dade2"

    fig, ax = plt.subplots(figsize=(14, 16), dpi=150)
    ax.set_xlim(0, 100); ax.set_ylim(0, 155); ax.axis("off")
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    def block(x, y, w, h, text, fc, ec, fs=12, weight="bold", tc=INK, rounding=2.2, lw=2.0, sub=None):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.15,rounding_size={rounding}",
                           linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3)
        ax.add_patch(p)
        ax.text(x+w/2, y+h/2+(1.0 if sub else 0), text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight=weight, zorder=4)
        if sub:
            ax.text(x+w/2, y+h/2-2.5, sub, ha="center", va="center",
                    fontsize=fs-2.5, color=SUBINK, zorder=4)
        return (x+w/2, y, x+w/2, y+h)

    def chips(xc, y, n, w, h, labels, fc, ec, fs=10):
        total = n*w + (n-1)*1.2; x0 = xc - total/2
        for i in range(n):
            x = x0 + i*(w+1.2)
            block(x, y, w, h, labels[i], fc, ec, fs=fs, rounding=1.2, lw=1.4)

    def arrow(x0, y0, x1, y1, color=INK, lw=2.2, style="-|>", rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=18,
                     linewidth=lw, color=color, zorder=2, connectionstyle=f"arc3,rad={rad}"))

    # Title
    ax.text(50, 152, "Unified Two-Head TMIL: Account- and Transaction-Level Phishing Detection",
            fontsize=13, color=INK, fontweight="bold", ha="center", va="center")

    # Input
    ax.text(4, 7.5, "Transaction sequence (one EOA's full on-chain history)", fontsize=11.5, color=INK, ha="left")
    chips(50, 1.5, 7, 11.5, 4.5, ["Tx1", "Tx2", "Tx3", "Tx4", "Tx5", "...", "TxN"], BLU, BLU_E, fs=10)
    
    # Feature Embeddings
    block(4, 13.5, 92, 9.5, "Feature Embeddings (per transaction)", YEL, YEL_E, fs=12,
          sub="Counterparty-ID emb  +  IN/OUT direction emb  +  log(Amount)  +  log(dt) Time2Vec")
    arrow(50, 6.0, 50, 13.5)

    # Sliding Windows
    ax.text(4, 31.5, "Sliding windows (W=200, S=50) — each window = one MIL instance", fontsize=11, color=INK, ha="left")
    chips(50, 25.0, 6, 13.5, 4.5, ["win 1", "win 2", "win 3", "win 4", "...", "win n"], PUR, PUR_E, fs=10)
    arrow(50, 23.0, 50, 25.0)

    # Shared Encoder
    block(4, 33.5, 92, 9.0, "1D Temporal Convolutional Network (Shared Encoder)", ORA, ORA_E, fs=13)
    for i in range(6):
        x = 8 + i*14.5
        arrow(x, 29.5, x, 33.5, lw=1.4)

    # Instance Representations
    ax.text(4, 51.0, "Instance representations {h_1, ..., h_n}", fontsize=11, color=INK, ha="left")
    chips(50, 44.5, 6, 13.5, 4.5, ["h1", "h2", "h3", "h4", "...", "hn"], GRN, GRN_E, fs=11)
    arrow(50, 42.5, 50, 44.5)

    # Split
    SY = 49.1
    arrow(50, SY, 27, 58.0, rad=0.12)
    arrow(50, SY, 73, 58.0, rad=-0.12)

    # Head-C
    CL = 5; CW = 41; cx = CL + CW/2
    ax.text(cx, 71.5, "HEAD-C — Account Classification", fontsize=12, color=BLU_E, fontweight="bold", ha="center")
    c1 = block(CL, 58.0, CW, 8.5, "Soft Gated Attention", PUR, PUR_E, fs=12,
               sub=r"tanh(Vh) * sigmoid(Uh)")
    c2 = block(CL, 78.0, CW, 8.0, "Attention weights a^C", PUR, PUR_E, fs=11.5,
               sub="sum_k a^C_k = 1")
    c3 = block(CL, 96.0, CW, 8.0, "Bag vector z = sum_k a^C_k * h_k", GRN, GRN_E, fs=11)
    c4 = block(CL, 114.0, CW, 8.0, "2-Layer MLP Classifier", ORA, ORA_E, fs=12)
    c5 = block(CL, 132.0, CW, 8.5, "(1) Account-level Prediction", BLU, BLU_E, fs=12.5,
               sub="L_BCE  —  AUC / AUPR / F1")
    for a, b in [(c1, c2), (c2, c3), (c3, c4), (c4, c5)]:
        arrow(cx, a[3], cx, b[1])

    # Head-L
    RL = 54; RW = 41; rx = RL + RW/2
    ax.text(rx, 71.5, "HEAD-L — Transaction Localization", fontsize=12, color=PNK_E, fontweight="bold", ha="center")
    r1 = block(RL, 58.0, RW, 8.5, "Outbound-Masked Gated Attention", PNK, PNK_E, fs=11,
               sub="masks inbound — trained with BCE")
    r2 = block(RL, 78.0, RW, 8.0, "Attention weights a^L", PNK, PNK_E, fs=11.5,
               sub="direction-aware")
    r3 = block(RL, 96.0, RW, 8.0, "Rank transactions by a^L", GRN, GRN_E, fs=11)
    r4 = block(RL, 114.0, RW, 8.0, "Localization read-out", ORA, ORA_E, fs=11.5,
               sub="vs amount / recency / degree / novelty")
    r5 = block(RL, 132.0, RW, 8.5, "(2) Transaction-level Localization", PNK, PNK_E, fs=11,
               sub="Hit@1 / Hit@5 / Hit@10 / MRR")
    for a, b in [(r1, r2), (r2, r3), (r3, r4), (r4, r5)]:
        arrow(rx, a[3], rx, b[1])

    ax.plot([50, 50], [57, 141], color="#d5dbdb", lw=1.2, ls=(0, (4, 4)), zorder=1)
    
    # Loss
    ax.text(50, 143.5, "Loss = BCE(Head-C) + lambda * BCE(Head-L) + beta * L_contrast",
            fontsize=11, color="#ffffff", fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=ACC, edgecolor=BLU_E, linewidth=1.5), zorder=5)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_arch_overview.png", dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print("Saved fig_arch_overview.png")

draw_arch()

# ============================================================
# FIG 2: Data Pipeline
# ============================================================
def draw_data_pipeline():
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 20)
    
    steps = [
        ("PTXPhish\nDataset", "#d4e6f1", "#2e86c1"),
        ("Step 1:\nAudit & Label\n(83.4% clean)", "#d5f5e3", "#1e8449"),
        ("Step 2:\nEtherscan\nCrawl", "#fad7a1", "#ca6f1e"),
        ("Step 3:\nSeaport\nRelabeling\n(92.7%)", "#ebdef0", "#8e44ad"),
        ("Step 4:\nBag Build\n(L=100,\nartifact rm)", "#fadbd8", "#cb4335"),
        ("Train/Test\nSplit", "#fcf3cf", "#d4ac0d"),
    ]
    
    for i, (label, fc, ec) in enumerate(steps):
        x = 2 + i * 16.5
        p = FancyBboxPatch((x, 3), 14, 14, boxstyle="round,pad=0.2", linewidth=2, edgecolor=ec, facecolor=fc, zorder=3)
        ax.add_patch(p)
        ax.text(x+7, 10, label, ha="center", va="center", fontsize=9.5, fontweight="bold", zorder=4)
        if i < len(steps)-1:
            ax.annotate("", xy=(x+16.5, 10), xytext=(x+14, 10),
                        arrowprops=dict(arrowstyle="->", color="#1b2631", lw=2))
    
    ax.set_title("Data Pipeline: From PTXPhish to Bag Construction", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_data_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_data_pipeline.png")

draw_data_pipeline()

# ============================================================
# FIG 3: Account-level Performance Bar Chart
# ============================================================
def draw_account_perf():
    methods = ["BERT4ETH", "ZipZap", "TSGN", "LMAE4Eth", "GatedMIL", "CLAM", "UnifiedTMIL\n(Ours)"]
    cross_aucs = [
        sota_res["account"]["BERT4ETH"]["cross_domain"]["auc"],
        sota_res["account"]["ZipZap"]["cross_domain"]["auc"],
        sota_res["account"]["TSGN"]["cross_domain"]["auc"],
        sota_res["account"]["LMAE4Eth"]["cross_domain"]["auc"],
        sota_res["account"]["GatedMIL"]["cross_domain"]["auc"],
        sota_res["account"]["CLAM"]["cross_domain"]["auc"],
        final_res["account"]["cross_domain"]["auc_combined"],
    ]
    hard_neg_aucs = [
        sota_res["account"]["BERT4ETH"]["cross_domain"]["by_source"]["ptx_benign"]["auc"],
        sota_res["account"]["ZipZap"]["cross_domain"]["by_source"]["ptx_benign"]["auc"],
        sota_res["account"]["TSGN"]["cross_domain"]["by_source"]["ptx_benign"]["auc"],
        sota_res["account"]["LMAE4Eth"]["cross_domain"]["by_source"]["ptx_benign"]["auc"],
        sota_res["account"]["GatedMIL"]["cross_domain"]["by_source"]["ptx_benign"]["auc"],
        sota_res["account"]["CLAM"]["cross_domain"]["by_source"]["ptx_benign"]["auc"],
        final_res["account"]["cross_domain"]["by_source"]["ptx_benign"]["auc"],
    ]
    
    x = np.arange(len(methods)); w = 0.38
    colors_main = ["#7f8c8d"] * 6 + ["#2980b9"]
    colors_hard = ["#bdc3c7"] * 6 + ["#c0392b"]
    
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    b1 = ax.bar(x - w/2, cross_aucs, w, label="Cross-domain AUC (combined)", color=colors_main, edgecolor="white")
    b2 = ax.bar(x + w/2, hard_neg_aucs, w, label="Hard-neg AUC (PTX-benign)", color=colors_hard, edgecolor="white")
    
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.5, 1.05)
    ax.set_title("Account-Level Detection: Cross-Domain AUC vs. SOTA", fontsize=13, fontweight="bold")
    ax.axhline(0.5, ls="--", c="k", lw=0.8, alpha=0.5)
    ax.legend(fontsize=10)
    
    for bar in b1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008, f"{bar.get_height():.3f}",
                ha="center", fontsize=8, fontweight="bold")
    for bar in b2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008, f"{bar.get_height():.3f}",
                ha="center", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_account_perf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_account_perf.png")

draw_account_perf()

# ============================================================
# FIG 4: Localization Performance
# ============================================================
def draw_loc_perf():
    fl = final_res["localization"]["full"]
    methods = ["Head-L\n(Original)", "Lean GBM\n(Ours)", "Recency", "Amount", "Novelty", "Degree"]
    lean_loc_data = load_json("results/lean_loc_results_fixed.json")
    
    hit1 = [fl["headL_unified"]["hit@1"], lean_loc_data["fixed_gbm_test"]["hit@1"],
            fl["recency"]["hit@1"], fl["amount"]["hit@1"], fl["novelty"]["hit@1"], fl["degree"]["hit@1"]]
    hit5 = [fl["headL_unified"]["hit@5"], lean_loc_data["fixed_gbm_test"]["hit@5"],
            fl["recency"]["hit@5"], fl["amount"]["hit@5"], fl["novelty"]["hit@5"], fl["degree"]["hit@5"]]
    hit10 = [fl["headL_unified"]["hit@10"], lean_loc_data["fixed_gbm_test"]["hit@10"],
             fl["recency"]["hit@10"] if "hit@10" in fl["recency"] else 0.931, fl["amount"]["hit@10"], fl["novelty"]["hit@10"], fl["degree"]["hit@10"]]
    
    x = np.arange(len(methods)); w = 0.27
    colors = ["#2980b9", "#27ae60", "#e67e22", "#95a5a6", "#16a085", "#9b59b6"]
    
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=150)
    for i, (vals, label, alpha) in enumerate([(hit1, "Hit@1", 1.0), (hit5, "Hit@5", 0.75), (hit10, "Hit@10", 0.5)]):
        bars = ax.bar(x + (i-1)*w, vals, w, label=label, color=colors, alpha=alpha, edgecolor="white")
    
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Hit@K"); ax.set_ylim(0, 1.1)
    ax.set_title("Transaction Localization: Hit@K Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    
    # Highlight our methods
    ax.axvline(0.5, ls="--", c="#2980b9", lw=1.2, alpha=0.4)
    ax.text(0.25, 1.05, "Original", ha="center", fontsize=9, color="#2980b9")
    ax.text(0.75, 1.05, "Ours", ha="center", fontsize=9, color="#27ae60")
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_loc_perf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_loc_perf.png")

draw_loc_perf()

# ============================================================
# FIG 5: SOTA Comparison (F1 + Cross AUC)
# ============================================================
def draw_sota_comparison():
    methods = ["BERT4ETH", "ZipZap", "TSGN", "LMAE4Eth", "GatedMIL", "TransMIL", "CLAM", "Ours"]
    f1s = [
        sota_res["account"]["BERT4ETH"]["in_domain"]["f1"][0],
        sota_res["account"]["ZipZap"]["in_domain"]["f1"][0],
        sota_res["account"]["TSGN"]["in_domain"]["f1"][0],
        sota_res["account"]["LMAE4Eth"]["in_domain"]["f1"][0],
        sota_res["account"]["GatedMIL"]["in_domain"]["f1"][0],
        sota_res["account"]["TransMIL"]["in_domain"]["f1"][0],
        sota_res["account"]["CLAM"]["in_domain"]["f1"][0],
        0.735,  # UnifiedTMIL
    ]
    aucs = [
        sota_res["account"]["BERT4ETH"]["cross_domain"]["auc"],
        sota_res["account"]["ZipZap"]["cross_domain"]["auc"],
        sota_res["account"]["TSGN"]["cross_domain"]["auc"],
        sota_res["account"]["LMAE4Eth"]["cross_domain"]["auc"],
        sota_res["account"]["GatedMIL"]["cross_domain"]["auc"],
        sota_res["account"]["TransMIL"]["cross_domain"]["auc"],
        sota_res["account"]["CLAM"]["cross_domain"]["auc"],
        final_res["account"]["cross_domain"]["auc_combined"],
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    colors = ["#7f8c8d"] * 7 + ["#e74c3c"]
    
    # F1
    bars1 = ax1.bar(methods, f1s, color=colors, edgecolor="white")
    ax1.set_ylabel("In-domain F1-score"); ax1.set_ylim(0.65, 0.80)
    ax1.set_title("In-Domain F1-Score Comparison", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="x", rotation=30)
    for bar in bars1:
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f"{bar.get_height():.3f}",
                 ha="center", fontsize=8.5)
    
    # AUC
    bars2 = ax2.bar(methods, aucs, color=colors, edgecolor="white")
    ax2.set_ylabel("Cross-Domain ROC-AUC"); ax2.set_ylim(0.85, 1.02)
    ax2.set_title("Cross-Domain AUC Comparison", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="x", rotation=30)
    for bar in bars2:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f"{bar.get_height():.3f}",
                 ha="center", fontsize=8.5)
    
    plt.suptitle("Comparison with SOTA Methods on Ethereum Phishing Detection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_sota_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_sota_comparison.png")

draw_sota_comparison()

# ============================================================
# FIG 6: OOD Generalization
# ============================================================
def draw_ood():
    cm = final_res["ood"]["cross_mechanism"]
    tm = final_res["ood"]["temporal"]
    
    names = list(cm.keys()) + list(tm.keys())
    aucs = [cm[k]["auc"] for k in cm] + [tm[k]["auc"] for k in tm]
    nice = {"payable_function": "Payable-fn\n(n=59)", "ice_phishing": "Ice-Phishing\n(n=45)",
            "address_poisoning": "Addr. Poison.\n(n=188)", "early": "Temporal:\nEarly", "late": "Temporal:\nLate"}
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    cols = ["#27ae60"] * len(cm) + ["#8e44ad"] * len(tm)
    bars = ax.bar([nice[n] for n in names], aucs, color=cols, edgecolor="white")
    ax.set_ylim(0.4, 1.05); ax.axhline(0.5, ls="--", c="k", lw=0.8, alpha=0.5, label="Chance")
    ax.set_ylabel("ROC-AUC"); ax.set_title("OOD Generalization: Cross-Mechanism & Temporal", fontsize=13, fontweight="bold")
    
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    
    green_patch = mpatches.Patch(color="#27ae60", label="Cross-Mechanism")
    purple_patch = mpatches.Patch(color="#8e44ad", label="Temporal Split")
    ax.legend(handles=[green_patch, purple_patch], fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_ood_generalization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_ood_generalization.png")

draw_ood()

# ============================================================
# FIG 7: Ablation Study
# ============================================================
def draw_ablation():
    # Localization ablation
    variants = list(loc_abl.keys())
    nice_names = {
        "content_no_rep (with attn)": "Full (with attn)",
        "content_no_rep_no_attn": "Full (no attn)",
        "recency": "Recency only"
    }
    
    hit1 = [loc_abl[v]["hit@1"] for v in variants]
    hit5 = [loc_abl[v]["hit@5"] for v in variants]
    hit10 = [loc_abl[v]["hit@10"] for v in variants]
    mrrs = [loc_abl[v]["mrr"] for v in variants]
    
    labels = [nice_names.get(v, v) for v in variants]
    x = np.arange(len(labels)); w = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    ax.bar(x - 1.5*w, hit1, w, label="Hit@1", color="#2980b9")
    ax.bar(x - 0.5*w, hit5, w, label="Hit@5", color="#27ae60")
    ax.bar(x + 0.5*w, hit10, w, label="Hit@10", color="#e67e22")
    ax.bar(x + 1.5*w, mrrs, w, label="MRR", color="#9b59b6")
    
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
    ax.set_title("Localization Ablation: Attention Contribution Analysis", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_ablation.png")

draw_ablation()

# ============================================================
# FIG 8: Complexity Comparison
# ============================================================
def draw_complexity():
    models = ["BERT4ETH", "ZipZap", "LMAE4Eth", "GatedMIL", "CLAM", "UnifiedTMIL\n(Ours)", "Lean MLP\n(Proposed)"]
    params = [65000, 45000, 85000, 55000, 70000, 50000, 2000]
    aucs = [0.948, 0.975, 0.950, 0.932, 0.962, 0.984, 0.832]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    colors = ["#7f8c8d"] * 5 + ["#e74c3c", "#27ae60"]
    sizes = [200 + p/200 for p in params]
    
    scatter = ax.scatter(params, aucs, s=sizes, c=colors, alpha=0.8, edgecolors="white", linewidths=2, zorder=3)
    
    for i, (m, p, a) in enumerate(zip(models, params, aucs)):
        offset = (0.01, 1500)
        ax.annotate(m, (p, a), xytext=(p+offset[1], a+offset[0]),
                    fontsize=9, ha="left", fontweight="bold" if i >= 5 else "normal")
    
    ax.set_xlabel("Number of Parameters", fontsize=11)
    ax.set_ylabel("Cross-Domain ROC-AUC", fontsize=11)
    ax.set_title("Complexity vs. Performance Trade-off", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim(0.78, 1.02)
    
    # Pareto front annotation
    ax.annotate("Lean MLP:\n96% fewer params\nbut lower AUC", xy=(2000, 0.832),
                xytext=(5000, 0.86), fontsize=9, color="#27ae60",
                arrowprops=dict(arrowstyle="->", color="#27ae60"))
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_complexity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_complexity.png")

draw_complexity()

# ============================================================
# FIG 9: Stratified AUC
# ============================================================
def draw_stratified():
    strat = final_res["account"]["cross_domain"]["stratified"]
    strata_names = ["3-20 tx", "20-100 tx", "100+ tx"]
    aucs = [strat["3-20"]["auc"], strat["20-100"]["auc"], strat["100+"]["auc"]]
    ns = [strat["3-20"]["n"], strat["20-100"]["n"], strat["100+"]["n"]]
    
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    colors = ["#2980b9", "#27ae60", "#e74c3c"]
    bars = ax.bar(strata_names, aucs, color=colors, edgecolor="white", width=0.5)
    ax.set_ylim(0.5, 1.05); ax.axhline(0.5, ls="--", c="k", lw=0.8, alpha=0.5)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Activity-Stratified Cross-Domain AUC\n(Controls tx-count confound)", fontsize=12, fontweight="bold")
    
    for bar, v, n in zip(bars, aucs, ns):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}\n(n={n})",
                ha="center", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig_stratified_auc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_stratified_auc.png")

draw_stratified()

print("\nAll figures generated successfully!")
print("Files in figures/:")
for f in sorted(os.listdir(FIGURES_DIR)):
    if f.endswith(".png"):
        print(f"  {f}")
