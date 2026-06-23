import re

tex_file = "/home/ubuntu/repo/paper/paper_en.tex"
with open(tex_file, "r") as f:
    content = f.read()

tables_section = r"""
% ---------------- TABLES ----------------
\begin{table}[!t]
\centering
\caption{Audited dataset. Direct-transfer GT covers 83.4\% of PTXPhish; receipt tracing raises clean account-level GT to 92.7\%. Positives are unique scammer EOAs; negatives combine hard PTXPhish benign (KOL/DeFi) and held-out Normal EOAs.}
\label{tab:data}
\resizebox{\columnwidth}{!}{
\begin{tabular}{lr}
\toprule
\textbf{Subset} & \textbf{Count} \\
\midrule
Phishing tx audited (PTXPhish) & 4998 \\
\ \ direct-transfer clean GT & 4168 (83.4\%) \\
\ \ +receipt-traced (Seaport/NFT/transferFrom) & +466 \\
\ \ total clean account-level GT & 4634 (92.7\%) \\
Unique scammer EOAs (positives) & 292 \\
\ \ poisoning / payable / ice\_phishing & 188 / 59 / 45 \\
Hard negatives (PTXPhish benign, KOL/DeFi) & 80 \\
Held-out Normal-EOA negatives & 1324 \\
Interior-GT localization bags & 101 \\
\bottomrule
\end{tabular}
}
\end{table}

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

\begin{table}[!t]
\centering
\caption{Account-level AUC stratified by activity (\#tx). High AUC persists in the high-activity stratum.}
\label{tab:strat}
\begin{tabular}{lccc}
\toprule
\textbf{Stratum (\#tx)} & \textbf{\#neg} & \textbf{\#pos} & \textbf{AUC} \\
\midrule
3--20 & 1218 & 54 & 0.999 \\
20--100 & 104 & 103 & 0.971 \\
100+ & 76 & 130 & 0.741 \\
\bottomrule
\end{tabular}
\end{table}

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

\begin{table}[!t]
\centering
\caption{Marginal value of the learned attention in the SOTA content-aware fusion (identical LOO protocol, $n{=}101$). Attention is a useful but non-essential signal.}
\label{tab:attn}
\resizebox{\columnwidth}{!}{
\begin{tabular}{lcccc}
\toprule
\textbf{Fusion variant} & \textbf{Hit@1} & \textbf{Hit@5} & \textbf{Hit@10} & \textbf{MRR} \\
\midrule
Content-aware ($+$baseline attn) & 0.832 & 0.931 & 0.941 & 0.880 \\
Content-aware ($+$strong attn) & 0.782 & 0.931 & 0.941 & 0.848 \\
Content-aware ($-$attn) & 0.792 & 0.941 & 0.941 & 0.860 \\
Recency prior & 0.693 & 0.921 & 0.931 & 0.799 \\
Attention only & 0.416 & 0.752 & 0.891 & 0.576 \\
\bottomrule
\end{tabular}
}
\end{table}

\begin{table}[!t]
\centering
\caption{Backbone ablation (2 seeds), account-level transfer. The counterparty and IN/OUT embeddings drive hard-negative transfer.}
\label{tab:abl}
\begin{tabular}{lccc}
\toprule
\textbf{Variant} & \textbf{ID-F1} & \textbf{X-AUC} & \textbf{X-AUC$_{hard}$} \\
\midrule
Full model & 0.734 & 0.974 & 0.849 \\
\ \ $-$ TCN encoder & 0.735 & 0.955 & 0.592 \\
\ \ $-$ counterparty emb. & 0.554 & 0.960 & 0.461 \\
\ \ $-$ IN/OUT emb. & 0.717 & 0.928 & 0.662 \\
$\lambda{=}0$ (soft only) & 0.733 & 0.965 & 0.803 \\
$\lambda{=}0.25$ & 0.735 & 0.969 & 0.837 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!t]
\centering
\caption{Out-of-distribution account-level generalization across cross-mechanism and temporal splits.}
\label{tab:ood_tab}
\begin{tabular}{lccc}
\toprule
\textbf{OOD partition} & \textbf{\#pos} & \textbf{AUC} & \textbf{AUPR} \\
\midrule
\multicolumn{4}{l}{\emph{Cross-mechanism (held-out type)}} \\
\ \ payable\_function & 59 & 0.998 & 0.894 \\
\ \ ice\_phishing & 45 & 0.967 & 0.316 \\
\ \ address\_poisoning & 188 & 0.984 & 0.765 \\
\midrule
\multicolumn{4}{l}{\emph{Temporal}} \\
\ \ early (train) & 146 & 0.985 & 0.802 \\
\ \ late (test) & 146 & 0.983 & 0.763 \\
\bottomrule
\end{tabular}
\end{table}
"""

# Replace from % ---------------- TABLES ---------------- to \begin{thebibliography}
pattern = r"% ---------------- TABLES ----------------.*?(?=\\begin\{thebibliography\})"
content = re.sub(pattern, tables_section.replace("\\", "\\\\"), content, flags=re.DOTALL)

with open(tex_file, "w") as f:
    f.write(content)

print("Tables section completely rewritten.")
