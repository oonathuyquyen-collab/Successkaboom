import re

tex_file = "/home/ubuntu/repo/paper/paper_en.tex"
with open(tex_file, "r") as f:
    content = f.read()

# Missing Table: Data (tab:data)
table_data = r"""
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
"""

# Missing Table: Stratified (tab:strat)
table_strat = r"""
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
"""

# Insert them before the first existing table
content = content.replace(r"\begin{table}[!t]", table_data + table_strat + r"\begin{table}[!t]", 1)

with open(tex_file, "w") as f:
    f.write(content)

print("Missing tables restored.")
