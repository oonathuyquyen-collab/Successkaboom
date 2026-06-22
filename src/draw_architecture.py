"""
Draw the architecture comparison diagram: Original Readykaboom vs Lean SOTA.
Outputs: figures/fig_lean_architecture.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(18, 12))
fig.patch.set_facecolor('#0d1117')

def draw_box(ax, x, y, w, h, text, color='#1f6feb', text_color='white', fontsize=9, alpha=0.9, radius=0.3):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          linewidth=1.5, edgecolor='#58a6ff', facecolor=color, alpha=alpha)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', wrap=True,
            multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2, color='#58a6ff'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.0))

def draw_original(ax):
    ax.set_facecolor('#161b22')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.set_title('Original Readykaboom Architecture\n(Over-engineered)', 
                 fontsize=13, color='#f0883e', fontweight='bold', pad=15)
    ax.axis('off')

    # Input
    draw_box(ax, 5, 13, 6, 0.8, 'INPUT: Ethereum Account\n(bag of L=100 transactions)', '#1f6feb')
    draw_arrow(ax, 5, 12.6, 5, 12.1)

    # Feature extraction
    draw_box(ax, 5, 11.7, 6, 0.8, 'Feature Extraction\nCounterparty ID | IO Direction | Amount | Time-delta', '#388bfd')
    draw_arrow(ax, 5, 11.3, 5, 10.8)

    # Embeddings
    draw_box(ax, 2.5, 10.4, 2.5, 0.8, 'Counterparty\nEmbedding\n(V × 64)', '#3fb950')
    draw_box(ax, 5, 10.4, 2, 0.8, 'IO Direction\nEmbedding\n(3 × 64)', '#3fb950')
    draw_box(ax, 7.5, 10.4, 2.5, 0.8, 'Amount+Time\nProjection\n(2 → 64)', '#3fb950')
    draw_arrow(ax, 5, 10.8, 2.5, 10.8)
    draw_arrow(ax, 5, 10.8, 5, 10.8)
    draw_arrow(ax, 5, 10.8, 7.5, 10.8)
    draw_arrow(ax, 5, 10.0, 5, 9.5)

    # Residual add + LayerNorm
    draw_box(ax, 5, 9.2, 4, 0.6, 'Residual Add + LayerNorm (64-dim)', '#6e40c9')
    draw_arrow(ax, 5, 8.9, 5, 8.4)

    # TCN
    draw_box(ax, 5, 8.1, 4, 0.6, 'TCN: Conv1D(64→64, k=3) + Residual', '#f0883e', alpha=0.9)
    ax.text(8.5, 8.1, '⚠ REDUNDANT', fontsize=7, color='#f85149', fontweight='bold', va='center')
    draw_arrow(ax, 5, 7.8, 5, 7.3)

    # Dual heads
    draw_box(ax, 2.5, 7.0, 3.5, 0.6, 'Head-C: Gated Attention\n(V=128, U=128, w=1)', '#6e40c9')
    draw_box(ax, 7.5, 7.0, 3.5, 0.6, 'Head-L: Gated Attention\n(outbound-masked)', '#6e40c9')
    draw_arrow(ax, 5, 7.3, 2.5, 7.3)
    draw_arrow(ax, 5, 7.3, 7.5, 7.3)
    ax.text(8.5, 6.6, '⚠ SUBSUMED', fontsize=7, color='#f85149', fontweight='bold', va='center')

    # Loss
    draw_box(ax, 2.5, 6.0, 3.5, 0.6, 'BCE Loss (account)', '#da3633')
    draw_box(ax, 7.5, 6.0, 3.5, 0.6, 'BCE Loss (localization)', '#da3633')
    draw_arrow(ax, 2.5, 6.7, 2.5, 6.3)
    draw_arrow(ax, 7.5, 6.7, 7.5, 6.3)

    # Combined loss
    draw_box(ax, 5, 5.0, 6, 0.6, 'L = BCE(Head-C) + λ·BCE(Head-L) + β·L_contrast', '#da3633')
    draw_arrow(ax, 2.5, 5.7, 5, 5.3)
    draw_arrow(ax, 7.5, 5.7, 5, 5.3)

    # Outputs
    draw_box(ax, 2.5, 3.8, 3.5, 0.8, 'Account Score\n(Head-C → sigmoid)', '#1f6feb')
    draw_box(ax, 7.5, 3.8, 3.5, 0.8, 'Attention Weights\n(Head-L → localization)', '#1f6feb')
    draw_arrow(ax, 2.5, 4.7, 2.5, 4.2)
    draw_arrow(ax, 7.5, 4.7, 7.5, 4.2)

    # GBM fusion
    draw_box(ax, 7.5, 2.8, 3.5, 0.8, 'GBM Fusion\n(attn + 17 features)', '#6e40c9')
    draw_arrow(ax, 7.5, 3.4, 7.5, 3.2)
    ax.text(9.5, 2.8, '⚠ ATTN\nNOT NEEDED', fontsize=7, color='#f85149', fontweight='bold', va='center')

    # Final outputs
    draw_box(ax, 2.5, 1.8, 3.5, 0.6, '✓ Account Classification', '#3fb950')
    draw_box(ax, 7.5, 1.8, 3.5, 0.6, '✓ Transaction Ranking', '#3fb950')
    draw_arrow(ax, 2.5, 3.4, 2.5, 2.1)
    draw_arrow(ax, 7.5, 2.4, 7.5, 2.1)

    # Complexity annotation
    ax.text(5, 0.8, '~50K parameters | 8 epochs | 3 seeds | Complex batching',
            ha='center', fontsize=8, color='#8b949e', style='italic')

def draw_lean(ax):
    ax.set_facecolor('#161b22')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.set_title('Lean SOTA Architecture\n(Proposed Optimization)', 
                 fontsize=13, color='#3fb950', fontweight='bold', pad=15)
    ax.axis('off')

    # Input
    draw_box(ax, 5, 13, 6, 0.8, 'INPUT: Ethereum Account\n(bag of L=100 transactions)', '#1f6feb')
    draw_arrow(ax, 5, 12.6, 5, 12.1)

    # Aggregate features
    draw_box(ax, 5, 11.7, 7, 0.8, 'Aggregate Feature Extraction\n(No padding, No sequence modeling)', '#388bfd')
    draw_arrow(ax, 5, 11.3, 5, 10.8)

    # Feature groups
    draw_box(ax, 2, 10.4, 2.5, 0.8, 'Amount Stats\nmean/std/max\nlog(1+amt)', '#3fb950')
    draw_box(ax, 5, 10.4, 2, 0.8, 'Time Stats\nmean/std\nlog(1+Δt)', '#3fb950')
    draw_box(ax, 7.8, 10.4, 2.5, 0.8, 'Structural\n|bag|, out-ratio\nuniq-cp-ratio', '#3fb950')
    draw_arrow(ax, 5, 10.8, 2, 10.8)
    draw_arrow(ax, 5, 10.8, 5, 10.8)
    draw_arrow(ax, 5, 10.8, 7.8, 10.8)
    draw_arrow(ax, 5, 10.0, 5, 9.5)

    # Concat
    draw_box(ax, 5, 9.2, 4, 0.6, 'Concat → 8-dim feature vector', '#6e40c9')
    draw_arrow(ax, 5, 8.9, 5, 8.4)

    # MLP
    draw_box(ax, 5, 8.1, 4, 0.6, 'MLP: 8 → 32 → 16 → 1 (ReLU)', '#388bfd')
    draw_arrow(ax, 5, 7.8, 5, 7.3)

    # BCE Loss
    draw_box(ax, 5, 7.0, 4, 0.6, 'BCE Loss + Margin Contrast', '#da3633')
    draw_arrow(ax, 5, 6.7, 5, 6.2)

    # Account output
    draw_box(ax, 5, 5.9, 4, 0.6, '✓ Account Score (sigmoid)', '#3fb950')
    draw_arrow(ax, 5, 5.6, 5, 5.1)

    # Conditional branch
    draw_box(ax, 5, 4.8, 5, 0.6, 'IF score > threshold → Localize', '#f0883e')
    draw_arrow(ax, 5, 4.5, 5, 4.0)

    # Engineered features for localization
    draw_box(ax, 5, 3.7, 7, 0.8, 'Per-Transaction Feature Engineering\namt-z | novelty | cp-reputation | position | zero-out', '#388bfd')
    draw_arrow(ax, 5, 3.3, 5, 2.8)

    # GBM ranker
    draw_box(ax, 5, 2.5, 5, 0.6, 'GBM Ranker (LightGBM, LOO protocol)', '#6e40c9')
    draw_arrow(ax, 5, 2.2, 5, 1.7)

    # Final output
    draw_box(ax, 5, 1.4, 5, 0.6, '✓ Transaction Ranking (Hit@1 = 0.845)', '#3fb950')

    # Complexity annotation
    ax.text(5, 0.6, '~2K parameters | 5 epochs | 1 seed | Simple batching',
            ha='center', fontsize=8, color='#8b949e', style='italic')
    ax.text(5, 0.2, '95% parameter reduction | 3× faster training | Reproducible',
            ha='center', fontsize=8, color='#3fb950', style='italic', fontweight='bold')

draw_original(axes[0])
draw_lean(axes[1])

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#1f6feb', label='Input/Output'),
    mpatches.Patch(facecolor='#388bfd', label='Feature Extraction'),
    mpatches.Patch(facecolor='#3fb950', label='Core Component'),
    mpatches.Patch(facecolor='#6e40c9', label='Aggregation/Pooling'),
    mpatches.Patch(facecolor='#da3633', label='Loss Function'),
    mpatches.Patch(facecolor='#f0883e', label='Conditional/Redundant'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=6, 
           facecolor='#161b22', edgecolor='#30363d', labelcolor='white',
           fontsize=9, bbox_to_anchor=(0.5, 0.01))

plt.suptitle('Readykaboom: Original vs Lean SOTA Architecture Comparison',
             fontsize=15, color='white', fontweight='bold', y=0.99)

plt.tight_layout(rect=[0, 0.06, 1, 0.97])
plt.savefig('/home/ubuntu/Readykaboom/figures/fig_lean_architecture.png', 
            dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("Saved: figures/fig_lean_architecture.png")
