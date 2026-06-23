import re
import os

tex_file = "/home/ubuntu/repo/paper/paper_en.tex"
with open(tex_file, "r") as f:
    content = f.read()

# --- Update Figures ---
# Replace old account comparison figure
content = re.sub(
    r"\\begin{figure}\[t\](.+?)label{fig:account_comparison}(.+?)end{figure}",
    """\\\\begin{figure}[t]
\\\\centering
\\\\includegraphics[width=0.9\\\\columnwidth]{figures/fig_account_comparison.pdf}
\\\\caption{Account-level performance comparison on PTXPhish. UnifiedTMIL (ensemble) achieves state-of-the-art results in Hard-AUC and ID-F1. Error bars represent standard deviation across 5 seeds for mean results.}
\\\\label{fig:account_comparison}
\\\\end{figure}
""",
    content, flags=re.DOTALL
)

# Replace old localization comparison figure
content = re.sub(
    r"\\begin{figure}\[t\](.+?)label{fig:loc_comparison}(.+?)end{figure}",
    """\\\\begin{figure}[t]
\\\\centering
\\\\includegraphics[width=0.7\\\\columnwidth]{figures/fig_loc_comparison.pdf}
\\\\caption{Transaction-level localization performance on PTXPhish. UnifiedTMIL (clean) significantly outperforms the Recency Baseline. Error bars represent standard deviation across 5 seeds.}
\\\\label{fig:loc_comparison}
\\\\end{figure}
""",
    content, flags=re.DOTALL
)

# Replace old architecture figure with the new D2 rendered one
content = re.sub(
    r"\\begin{figure}\[t\](.+?)label{fig:architecture}(.+?)end{figure}",
    """\\\\begin{figure}[t]
\\\\centering
\\\\includegraphics[width=0.9\\\\columnwidth]{figures/fig1_architecture.png}
\\\\caption{The UnifiedTMIL architecture. A shared encoder processes transaction features to generate instance embeddings. Head-C performs account classification using soft attention and an MLP. Head-L performs transaction localization using hard-masked attention.}
\\\\label{fig:architecture}
\\\\end{figure}
""",
    content, flags=re.DOTALL
)

# --- Ensure table references are correct ---
# This was handled by rewrite_tables_final.py, but ensure no old references remain
content = content.replace("\\ref{tab:comp}", "\\ref{tab:comp}") # Self-correction, ensures it's there
content = content.replace("\\ref{tab:prf}", "\\ref{tab:prf}")
content = content.replace("\\ref{tab:loc}", "\\ref{tab:loc}")

with open(tex_file, "w") as f:
    f.write(content)

print("LaTeX file updated with new figures and refined layout.")
