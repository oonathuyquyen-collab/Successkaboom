"""
Draw metric comparison charts: Original vs Lean SOTA
Outputs: figures/fig_metrics_comparison.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0d1117')

colors = {
    'BERT4ETH': '#388bfd',
    'ZipZap': '#6e40c9',
    'TSGN': '#3fb950',
    'LMAE4Eth': '#f0883e',
    'Readykaboom\n(TMIL)': '#da3633',
    'Lean SOTA\n(Proposed)': '#58a6ff',
}

# ---- Plot 1: Account-level F1 ----
ax = axes[0]
ax.set_facecolor('#161b22')
models = ['BERT4ETH', 'ZipZap', 'TSGN', 'LMAE4Eth', 'Readykaboom\n(TMIL)', 'Lean SOTA\n(Proposed)']
f1_scores = [0.734, 0.711, 0.721, 0.718, 0.735, 0.742]
bar_colors = [colors[m] for m in models]
bars = ax.bar(models, f1_scores, color=bar_colors, edgecolor='#30363d', linewidth=1.2, width=0.6)
for bar, val in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')
ax.set_ylim(0.65, 0.78)
ax.set_title('Account-Level F1 (In-Domain)', fontsize=12, color='white', fontweight='bold', pad=10)
ax.set_ylabel('F1 Score', color='white', fontsize=10)
ax.tick_params(colors='white', labelsize=8)
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#161b22')
ax.yaxis.label.set_color('white')
ax.xaxis.label.set_color('white')
# Highlight lean SOTA bar
bars[-1].set_edgecolor('#3fb950')
bars[-1].set_linewidth(3)

# ---- Plot 2: Cross-domain AUC ----
ax = axes[1]
ax.set_facecolor('#161b22')
auc_combined = [0.948, 0.979, 0.985, 0.980, 0.984, 0.992]
auc_hard = [0.836, 0.776, 0.760, 0.708, 0.725, 0.749]
x = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width/2, auc_combined, width, label='Combined AUC', color=[colors[m] for m in models], alpha=0.9, edgecolor='#30363d')
bars2 = ax.bar(x + width/2, auc_hard, width, label='Hard-neg AUC', color=[colors[m] for m in models], alpha=0.5, edgecolor='#30363d', hatch='//')
for bar, val in zip(bars1, auc_combined):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7, color='white', fontweight='bold')
for bar, val in zip(bars2, auc_hard):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7, color='#8b949e')
ax.set_ylim(0.65, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=8, color='white')
ax.set_title('Cross-Domain AUC (Combined vs Hard-neg)', fontsize=12, color='white', fontweight='bold', pad=10)
ax.set_ylabel('AUC Score', color='white', fontsize=10)
ax.tick_params(colors='white', labelsize=8)
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=8)

# ---- Plot 3: Localization Hit@K ----
ax = axes[2]
ax.set_facecolor('#161b22')
loc_methods = ['Recency\nPrior', 'Head-L\n(Strong)', 'GatedMIL', 'CLAM', 'Readykaboom\nFusion', 'Lean SOTA\n(GBM Only)']
hit1 = [0.693, 0.515, 0.446, 0.455, 0.832, 0.845]
hit5 = [0.921, 0.851, 0.752, 0.792, 0.931, 0.941]
hit10 = [0.931, 0.921, 0.881, 0.871, 0.941, 0.950]
x = np.arange(len(loc_methods))
width = 0.25
loc_colors = ['#6e40c9', '#f0883e', '#388bfd', '#3fb950', '#da3633', '#58a6ff']
bars1 = ax.bar(x - width, hit1, width, label='Hit@1', color=loc_colors, alpha=1.0, edgecolor='#30363d')
bars2 = ax.bar(x, hit5, width, label='Hit@5', color=loc_colors, alpha=0.7, edgecolor='#30363d')
bars3 = ax.bar(x + width, hit10, width, label='Hit@10', color=loc_colors, alpha=0.4, edgecolor='#30363d')
ax.set_ylim(0.3, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(loc_methods, fontsize=7.5, color='white')
ax.set_title('Transaction Localization (Hit@K)', fontsize=12, color='white', fontweight='bold', pad=10)
ax.set_ylabel('Hit Rate', color='white', fontsize=10)
ax.tick_params(colors='white', labelsize=8)
ax.spines['bottom'].set_color('#30363d')
ax.spines['left'].set_color('#30363d')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(['Hit@1', 'Hit@5', 'Hit@10'], facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=8)
# Annotate lean SOTA
ax.annotate('★ SOTA', xy=(5 - width, 0.845), xytext=(4.2, 0.92),
            arrowprops=dict(arrowstyle='->', color='#3fb950', lw=1.5),
            fontsize=9, color='#3fb950', fontweight='bold')

plt.suptitle('Performance Comparison: Original Readykaboom vs Lean SOTA',
             fontsize=14, color='white', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/ubuntu/Readykaboom/figures/fig_metrics_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("Saved: figures/fig_metrics_comparison.png")
