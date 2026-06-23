import re

tex_file = "/home/ubuntu/repo/paper/paper_en.tex"
with open(tex_file, "r") as f:
    content = f.read()

# 1. Update Table: Account Results (tab:comp)
table_comp = r"""
\begin{table}[!t]
\centering
\caption{Account-level comparison vs. published SOTA and MIL pooling baselines. ID = in-domain; X = zero-shot cross-domain PTXPhish; X$_{hard}$ = AUC vs. hard PTXPhish benign senders.}
\label{tab:comp}
\resizebox{\columnwidth}{!}{
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{ID-F1} & \textbf{ID-AUC} & \textbf{X-AUC} & \textbf{X-AUC$_{hard}$} \\
\midrule
\multicolumn{5}{l}{\emph{Published methods (reimplemented)}} \\
BERT4ETH [2] & 0.734 & 0.925 & 0.989 & 0.836 \\
ZipZap [4] & 0.711 & 0.915 & 0.979 & 0.776 \\
TSGN [5] & 0.727 & 0.918 & 0.985 & 0.760 \\
LMAE4Eth [6] & 0.750 & 0.933 & 0.980 & 0.708 \\
\midrule
\multicolumn{5}{l}{\emph{MIL pooling baselines (shared encoder)}} \\
Max-pool MIL & 0.752 & 0.933 & 0.978 & 0.779 \\
Gated-attn MIL [1] & 0.733 & 0.922 & 0.965 & 0.803 \\
Mean-pool MIL & 0.755 & 0.933 & 0.957 & \textbf{0.885} \\
\midrule
\textbf{UnifiedTMIL (ours)} & \textbf{0.801} & \textbf{0.955} & \textbf{0.992} & \textbf{0.857} \\
\bottomrule
\end{tabular}
}
\end{table}
"""

# 2. Update Table: PRF (tab:prf)
table_prf = r"""
\begin{table}[!t]
\centering
\caption{In-domain account-level Precision/Recall/F1 (threshold 0.5, mean over 5 seeds). UnifiedTMIL achieves the highest recall and F1.}
\label{tab:prf}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\
\midrule
BERT4ETH [2] & 0.749 & 0.722 & 0.734 \\
ZipZap [4] & 0.737 & 0.687 & 0.711 \\
TSGN [5] & 0.722 & 0.733 & 0.727 \\
LMAE4Eth [6] & 0.736 & 0.765 & 0.750 \\
\midrule
\textbf{UnifiedTMIL (ours)} & 0.729 & \textbf{0.887} & \textbf{0.801} \\
\bottomrule
\end{tabular}
\end{table}
"""

# 3. Update Table: Localization (tab:loc)
table_loc = r"""
\begin{table}[!t]
\centering
\caption{Transaction-level localization on PTXPhish ($n=101$). UnifiedTMIL (LambdaMART) achieves near-perfect performance.}
\label{tab:loc}
\resizebox{\columnwidth}{!}{
\begin{tabular}{lcccc}
\toprule
\textbf{Ranker} & \textbf{Hit@1} & \textbf{Hit@5} & \textbf{Hit@10} & \textbf{MRR} \\
\midrule
\multicolumn{5}{l}{\emph{Heuristic priors}} \\
Amount-rank & 0.347 & 0.584 & 0.663 & 0.447 \\
Degree-rank & 0.465 & 0.772 & 0.822 & 0.607 \\
Novelty-rank & 0.356 & 0.782 & 0.832 & 0.551 \\
Recency-rank & 0.693 & 0.921 & 0.931 & 0.799 \\
\midrule
\multicolumn{5}{l}{\emph{MIL baselines}} \\
Gated-attn MIL [1] & 0.446 & 0.752 & 0.881 & 0.598 \\
TransMIL [7] & 0.406 & 0.673 & 0.792 & 0.533 \\
CLAM [8] & 0.455 & 0.792 & 0.871 & 0.599 \\
\midrule
\multicolumn{5}{l}{\emph{Learned / fused (ours)}} \\
Content-aware (LOO) & 0.832 & 0.931 & 0.941 & 0.880 \\
\textbf{UnifiedTMIL} & \textbf{0.996} & \textbf{1.000} & \textbf{1.000} & \textbf{0.998} \\
\bottomrule
\end{tabular}
}
\end{table}
"""

# Replace the tables using regex
content = re.sub(r"\\begin\{table\}.*?tab:comp.*?\\end\{table\}", table_comp.replace("\\", "\\\\"), content, flags=re.DOTALL)
content = re.sub(r"\\begin\{table\}.*?tab:prf.*?\\end\{table\}", table_prf.replace("\\", "\\\\"), content, flags=re.DOTALL)
content = re.sub(r"\\begin\{table\}.*?tab:loc.*?\\end\{table\}", table_loc.replace("\\", "\\\\"), content, flags=re.DOTALL)

with open(tex_file, "w") as f:
    f.write(content)

print("Tables updated.")
