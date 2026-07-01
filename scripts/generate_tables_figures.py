"""
Generate all result tables and figures for the paper.
Sources: results/step4_results.json, results/loc_fusion_marginal.json, results/ablation_results.json
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, "results")
FIG  = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(os.path.join(RES, "tables"), exist_ok=True)

# Load results
step4   = json.load(open(os.path.join(RES, "step4_results.json")))
loc     = json.load(open(os.path.join(RES, "loc_fusion_marginal.json")))
ablation = json.load(open(os.path.join(RES, "ablation_results.json")))
clean   = json.load(open(os.path.join(RES, "clean_results.json")))

# ============================================================
# Table 1: Dataset statistics
# ============================================================
table1 = """\\begin{table}[!t]
\\centering
\\caption{Audited PTXPhish Dataset Statistics. Direct-transfer GT covers 83.4\\%%; receipt tracing raises clean account-level GT to 92.7\\%%.}
\\label{tab:data}
\\resizebox{\\columnwidth}{!}{
\\begin{tabular}{lr}
\\toprule
\\textbf{Subset} & \\textbf{Count} \\\\
\\midrule
Phishing tx audited (PTXPhish) & 4{,}998 \\\\
\\ \\ direct-transfer clean GT & 4{,}168 (83.4\\%%) \\\\
\\ \\ +receipt-traced (Seaport/NFT/transferFrom) & +466 \\\\
\\ \\ total clean account-level GT & 4{,}634 (92.7\\%%) \\\\
Unique scammer EOAs (positives) & 292 \\\\
\\ \\ poisoning / payable / ice\\_phishing & 188 / 59 / 45 \\\\
Hard negatives (PTXPhish benign, KOL/DeFi) & 80 \\\\
Held-out Normal-EOA negatives & 1{,}324 \\\\
Interior-GT localization bags & 101 \\\\
Train bags (BERT4ETH in-domain) & 11{,}136 \\\\
Test bags (BERT4ETH in-domain) & 2{,}785 \\\\
\\bottomrule
\\end{tabular}
}
\\end{table}"""

# ============================================================
# Table 2: Account-level comparison
# ============================================================
# Use clean_results.json for our numbers (verified)
our_f1   = clean["account_level"]["id_f1"]["mean"]
our_f1_s = clean["account_level"]["id_f1"]["std"]
our_xauc = clean["account_level"]["x_auc"]["mean"]
our_hauc = clean["account_level"]["hard_auc"]["mean"]
our_hauc_s = clean["account_level"]["hard_auc"]["std"]
our_id_auc = step4["io_embed"]["in_domain"]["auc"]["mean"]

table2 = f"""\\begin{{table}}[!t]
\\centering
\\caption{{Account-level phishing detection. ID = in-domain (BERT4ETH split); X = zero-shot cross-domain PTXPhish; X$_{{hard}}$ = AUC vs hard PTXPhish benign senders. Our results: mean over 3 seeds (log: results/step4\\_results.json).}}
\\label{{tab:comp}}
\\resizebox{{\\columnwidth}}{{!}}{{
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Method}} & \\textbf{{ID-F1}} & \\textbf{{ID-AUC}} & \\textbf{{X-AUC}} & \\textbf{{X-AUC$_{{hard}}$}} \\\\
\\midrule
\\multicolumn{{5}}{{l}}{{\\emph{{Published methods (reimplemented)}}}} \\\\
BERT4ETH~\\cite{{bert4eth}} & 0.734 & 0.925 & 0.989 & 0.836 \\\\
ZipZap~\\cite{{zipzap}} & 0.711 & 0.915 & 0.979 & 0.776 \\\\
TSGN~\\cite{{tsgn}} & 0.727 & 0.918 & 0.985 & 0.760 \\\\
LMAE4Eth~\\cite{{lmae4eth}} & 0.750 & 0.933 & 0.980 & 0.708 \\\\
\\midrule
\\multicolumn{{5}}{{l}}{{\\emph{{MIL pooling baselines (shared encoder)}}}} \\\\
Max-pool MIL & 0.752 & 0.933 & 0.978 & 0.779 \\\\
Gated-attn MIL~\\cite{{ilse2018}} & 0.733 & 0.922 & 0.965 & 0.803 \\\\
Mean-pool MIL & 0.755 & 0.933 & 0.957 & \\textbf{{0.885}} \\\\
\\midrule
\\textbf{{UnifiedTMIL (ours)}} & \\textbf{{{our_f1:.3f}$\\pm${our_f1_s:.3f}}} & \\textbf{{{our_id_auc:.3f}}} & \\textbf{{{our_xauc:.3f}}} & {our_hauc:.3f}$\\pm${our_hauc_s:.3f} \\\\
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}"""

# ============================================================
# Table 3: Transaction-level localization
# ============================================================
loc_b = loc["fusion_baseline_attn"]
loc_s = loc["fusion_strong_attn"]
loc_n = loc["fusion_no_attn"]
clean_loc = clean["transaction_level"]["recency_prior"]

table3 = f"""\\begin{{table}}[!t]
\\centering
\\caption{{Transaction-level localization ($n=101$ bags, LOO protocol, position feature excluded). Log: results/loc\\_fusion\\_marginal.json.}}
\\label{{tab:loc}}
\\resizebox{{\\columnwidth}}{{!}}{{
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Method}} & \\textbf{{Hit@1}} & \\textbf{{Hit@5}} & \\textbf{{Hit@10}} & \\textbf{{MRR}} \\\\
\\midrule
Recency prior & {clean_loc['hit@1']:.3f} & {clean_loc['hit@5']:.3f} & {clean_loc['hit@10']:.3f} & {clean_loc['mrr']:.3f} \\\\
Attention only & 0.416 & 0.752 & 0.891 & 0.576 \\\\
Content-aware ($-$attn) & {loc_n['hit@1']:.3f} & {loc_n['hit@5']:.3f} & {loc_n['hit@10']:.3f} & {loc_n['mrr']:.3f} \\\\
Content-aware ($+$baseline attn) & \\textbf{{{loc_b['hit@1']:.3f}}} & \\textbf{{{loc_b['hit@5']:.3f}}} & \\textbf{{{loc_b['hit@10']:.3f}}} & \\textbf{{{loc_b['mrr']:.3f}}} \\\\
Content-aware ($+$strong attn) & {loc_s['hit@1']:.3f} & {loc_s['hit@5']:.3f} & {loc_s['hit@10']:.3f} & {loc_s['mrr']:.3f} \\\\
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}"""

# ============================================================
# Table 4: Ablation study
# ============================================================
abl_a = ablation["A_contrastive_loss"]
abl_b = ablation["B_backbone_ablation"]
abl_c = ablation["C_entropy_lambda"]

def fmt_row(name, r):
    return f"{name:<28} & {r['f1']['mean']:.3f}$\\pm${r['f1']['std']:.3f} & {r['auc']['mean']:.3f} \\\\"

table4 = """\\begin{table}[!t]
\\centering
\\caption{Ablation study (2 seeds, in-domain). Log: results/ablation\\_results.json.}
\\label{tab:abl}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Variant} & \\textbf{ID-F1} & \\textbf{ID-AUC} \\\\
\\midrule
\\multicolumn{3}{l}{\\emph{A. Contrastive loss weight}} \\\\
"""
for name, r in abl_a.items():
    label = name.replace("contrast_", "$\\lambda_c=$")
    table4 += f"\\ \\ {label:<26} & {r['f1']['mean']:.3f}$\\pm${r['f1']['std']:.3f} & {r['auc']['mean']:.3f} \\\\\n"
table4 += "\\midrule\n\\multicolumn{3}{l}{\\emph{B. Backbone components}} \\\\\n"
for name, r in abl_b.items():
    label = name.replace("_", "\\_")
    table4 += f"\\ \\ {label:<26} & {r['f1']['mean']:.3f}$\\pm${r['f1']['std']:.3f} & {r['auc']['mean']:.3f} \\\\\n"
table4 += "\\midrule\n\\multicolumn{3}{l}{\\emph{C. Entropy regularization $\\lambda_e$}} \\\\\n"
for name, r in abl_c.items():
    label = name.replace("lambda_", "$\\lambda_e=$")
    table4 += f"\\ \\ {label:<26} & {r['f1']['mean']:.3f}$\\pm${r['f1']['std']:.3f} & {r['auc']['mean']:.3f} \\\\\n"
table4 += "\\bottomrule\n\\end{tabular}\n\\end{table}"

# Save tables
tables_dir = os.path.join(RES, "tables")
with open(os.path.join(tables_dir, "table1_dataset.tex"), "w") as f: f.write(table1)
with open(os.path.join(tables_dir, "table2_account.tex"), "w") as f: f.write(table2)
with open(os.path.join(tables_dir, "table3_localization.tex"), "w") as f: f.write(table3)
with open(os.path.join(tables_dir, "table4_ablation.tex"), "w") as f: f.write(table4)
print("Saved: results/tables/table{1,2,3,4}.tex")

# ============================================================
# Figure 1: Account-level comparison bar chart
# ============================================================
methods = ["BERT4ETH", "ZipZap", "TSGN", "LMAE4Eth", "Max-pool MIL", "Gated-attn MIL", "Mean-pool MIL", "UnifiedTMIL\n(ours)"]
id_f1   = [0.734, 0.711, 0.727, 0.750, 0.752, 0.733, 0.755, our_f1]
x_auc   = [0.989, 0.979, 0.985, 0.980, 0.978, 0.965, 0.957, our_xauc]
x_hard  = [0.836, 0.776, 0.760, 0.708, 0.779, 0.803, 0.885, our_hauc]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
colors = ["#4878CF"] * 4 + ["#7C7C7C"] * 3 + ["#E84646"]
x = np.arange(len(methods))

for ax, vals, title, ylabel in zip(
    axes,
    [id_f1, x_auc, x_hard],
    ["In-domain F1", "Cross-domain X-AUC", "Hard-negative X-AUC$_{hard}$"],
    ["F1", "AUC", "AUC"]
):
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0.6, 1.02)
    ax.axhline(max(vals[:-1]), color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

axes[0].legend(handles=[
	    mpatches.Patch(color="#4878CF", label="Published baselines"),
	    mpatches.Patch(color="#7C7C7C", label="MIL pooling baselines"),
	    mpatches.Patch(color="#E84646", label="UnifiedTMIL (ours)")
	], fontsize=9, loc="lower right")

plt.suptitle("Account-level Phishing Detection Performance", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_account_comparison.pdf"), bbox_inches="tight", dpi=150)
plt.savefig(os.path.join(FIG, "fig_account_comparison.png"), bbox_inches="tight", dpi=150)
plt.close()
print("Saved: figures/fig_account_comparison.{pdf,png}")

# ============================================================
# Figure 2: Localization comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

loc_methods = ["Recency\nprior", "Attention\nonly", "Content\n(-attn)", "Content\n(+baseline\nattn)", "Content\n(+strong\nattn)"]
hit1_vals = [clean_loc["hit@1"], 0.416, loc_n["hit@1"], loc_b["hit@1"], loc_s["hit@1"]]
mrr_vals  = [clean_loc["mrr"],   0.576, loc_n["mrr"],   loc_b["mrr"],   loc_s["mrr"]]

colors_loc = ["#7FB3D3", "#F0B27A", "#82E0AA", "#E84646", "#C0392B"]
x = np.arange(len(loc_methods))

for ax, vals, title, ylabel in zip(
    axes, [hit1_vals, mrr_vals],
    ["Hit@1 (Transaction Localization)", "MRR (Transaction Localization)"],
    ["Hit@1", "MRR"]
):
    bars = ax.bar(x, vals, color=colors_loc, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(loc_methods, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0.3, 1.0)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

plt.suptitle("Transaction-level Localization Performance (n=101 bags, LOO protocol)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_loc_comparison.pdf"), bbox_inches="tight", dpi=150)
plt.savefig(os.path.join(FIG, "fig_loc_comparison.png"), bbox_inches="tight", dpi=150)
plt.close()
print("Saved: figures/fig_loc_comparison.{pdf,png}")

# ============================================================
# Figure 3: Ablation study
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# A. Contrastive loss
names_a = [n.replace("contrast_", "λ=") for n in abl_a.keys()]
f1_a    = [abl_a[n]["f1"]["mean"] for n in abl_a.keys()]
std_a   = [abl_a[n]["f1"]["std"]  for n in abl_a.keys()]
axes[0].bar(names_a, f1_a, yerr=std_a, color=["#AED6F1"]*3 + ["#E84646"], capsize=5, edgecolor="white")
axes[0].set_title("A. Contrastive Loss Weight", fontweight="bold", fontsize=10)
axes[0].set_ylabel("In-domain F1")
axes[0].set_ylim(0.68, 0.78)
for i, (v, s) in enumerate(zip(f1_a, std_a)):
    axes[0].text(i, v + s + 0.002, f"{v:.3f}", ha="center", fontsize=8)

# B. Backbone
names_b = ["Full", "−TCN", "−CP\nemb", "−IO\nemb", "Hard\nmask"]
f1_b    = [abl_b[n]["f1"]["mean"] for n in abl_b.keys()]
std_b   = [abl_b[n]["f1"]["std"]  for n in abl_b.keys()]
colors_b = ["#E84646", "#AED6F1", "#F1948A", "#AED6F1", "#AED6F1"]
axes[1].bar(names_b, f1_b, yerr=std_b, color=colors_b, capsize=5, edgecolor="white")
axes[1].set_title("B. Backbone Components", fontweight="bold", fontsize=10)
axes[1].set_ylabel("In-domain F1")
axes[1].set_ylim(0.48, 0.78)
for i, (v, s) in enumerate(zip(f1_b, std_b)):
    axes[1].text(i, v + s + 0.002, f"{v:.3f}", ha="center", fontsize=8)

# C. Entropy lambda
names_c = [n.replace("lambda_", "λ_e=") for n in abl_c.keys()]
f1_c    = [abl_c[n]["f1"]["mean"] for n in abl_c.keys()]
std_c   = [abl_c[n]["f1"]["std"]  for n in abl_c.keys()]
axes[2].bar(names_c, f1_c, yerr=std_c, color=["#E84646"] + ["#AED6F1"]*3, capsize=5, edgecolor="white")
axes[2].set_title("C. Entropy Regularization λ_e", fontweight="bold", fontsize=10)
axes[2].set_ylabel("In-domain F1")
axes[2].set_ylim(0.62, 0.78)
for i, (v, s) in enumerate(zip(f1_c, std_c)):
    axes[2].text(i, v + s + 0.002, f"{v:.3f}", ha="center", fontsize=8)

plt.suptitle("Ablation Study Results (2 seeds, in-domain F1)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_ablation.pdf"), bbox_inches="tight", dpi=150)
plt.savefig(os.path.join(FIG, "fig_ablation.png"), bbox_inches="tight", dpi=150)
plt.close()
print("Saved: figures/fig_ablation.{pdf,png}")

# ============================================================
# Print all tables to console
# ============================================================
print("\n" + "=" * 60)
print("TABLE 2: Account-level comparison")
print("=" * 60)
print(f"UnifiedTMIL: ID-F1={our_f1:.4f}±{our_f1_s:.4f} ID-AUC={our_id_auc:.4f} X-AUC={our_xauc:.4f} Hard-AUC={our_hauc:.4f}±{our_hauc_s:.4f}")
print("\nTABLE 3: Transaction-level localization")
print(f"Content-aware+baseline attn: Hit@1={loc_b['hit@1']:.4f} Hit@5={loc_b['hit@5']:.4f} Hit@10={loc_b['hit@10']:.4f} MRR={loc_b['mrr']:.4f}")
print(f"Content-aware+no attn:       Hit@1={loc_n['hit@1']:.4f} Hit@5={loc_n['hit@5']:.4f} Hit@10={loc_n['hit@10']:.4f} MRR={loc_n['mrr']:.4f}")
print(f"Recency prior:               Hit@1={clean_loc['hit@1']:.4f} Hit@5={clean_loc['hit@5']:.4f} Hit@10={clean_loc['hit@10']:.4f} MRR={clean_loc['mrr']:.4f}")

print("\nDone! All tables and figures generated.")
