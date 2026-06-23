import re
import os

tex_file = "/home/ubuntu/repo/paper/paper_en.tex"
with open(tex_file, "r") as f:
    content = f.read()

# 1. Update preamble
preamble_updates = r"""\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{url}
\usepackage{xcolor}
\usepackage{multirow}
\usepackage{placeins}
\usepackage{tabularx}
"""
content = content.replace(r"\usepackage{cite,amsmath,amssymb,graphicx,booktabs,url,xcolor,multirow}", preamble_updates)

# 2. New figures
fig1_new = r"""
\begin{figure}[!t]
\centering
\includegraphics[width=0.9\columnwidth]{../figures/fig1_architecture.png}
\caption{UnifiedTMIL Architecture. A single bag of transactions is processed into shared aggregate features (26) and per-transaction features (16). The framework jointly performs account classification via an MLP and transaction localization via a LambdaMART meta-learner.}
\label{fig:arch}
\end{figure}
"""

fig2_new = r"""
\begin{figure}[!t]
\centering
\includegraphics[width=0.9\columnwidth]{../figures/fig_account_comparison.pdf}
\caption{Account-level phishing detection performance. UnifiedTMIL (highlighted) achieves state-of-the-art ID-F1 (0.801) and Hard-AUC (0.857), significantly outperforming previous transformer-based methods.}
\label{fig:acct_comp}
\end{figure}
"""

fig3_new = r"""
\begin{figure}[!t]
\centering
\includegraphics[width=0.9\columnwidth]{../figures/fig_loc_comparison.pdf}
\caption{Transaction-level localization performance. Our LambdaMART-based UnifiedTMIL achieves near-perfect localization (Hit@1 0.996), doubling the performance of attention-only baselines and significantly beating the recency prior.}
\label{fig:loc_comp}
\end{figure}
"""

fig4_new = r"""
\begin{figure}[!t]
\centering
\includegraphics[width=0.85\columnwidth]{../figures/fig_complexity.pdf}
\caption{Complexity vs. Performance. UnifiedTMIL's aggregate head is significantly leaner (fewer parameters) than full transformer baselines while achieving higher in-domain AUC.}
\label{fig:complexity}
\end{figure}
"""

# 3. Use regex to find and replace the entire figures section
# This regex matches from the first \begin{figure} to the last \end{figure} before \section{Conclusion}
pattern = r"\\begin\{figure\}.*\\end\{figure\}"
new_figures = fig1_new + fig2_new + fig3_new + fig4_new
content = re.sub(pattern, new_figures.replace("\\", "\\\\"), content, flags=re.DOTALL)

with open(tex_file, "w") as f:
    f.write(content)

print("LaTeX updated.")
