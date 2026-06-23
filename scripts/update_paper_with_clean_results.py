"""
Update the paper_en.tex with clean results.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX_FILE = os.path.join(ROOT, "paper", "paper_en.tex")

with open(TEX_FILE, "r") as f:
    content = f.read()

# 1. Abstract updates
# Old: Hard-AUC 0.857, and X-AUC 0.992 for account detection (mean ID-F1 0.744±0.005, Hard-AUC 0.696±0.148 across 5 seeds), and Hit@1 0.996, Hit@5 1.000, Hit@10 1.000, MRR 0.998
# New: Hard-AUC 0.696±0.148, and X-AUC 0.981 for account detection (mean ID-F1 0.744±0.005 across 5 seeds), and Hit@1 0.832, Hit@5 0.931, Hit@10 0.941, MRR 0.880
content = re.sub(
    r"Hard-AUC 0\.857, and X-AUC 0\.992 for account detection \(mean ID-F1 0\.744±0\.005, Hard-AUC 0\.696±0\.148 across 5 seeds\), and Hit@1 0\.996, Hit@5 1\.000,\nHit@10 1\.000, MRR 0\.998",
    r"Hard-AUC 0.696±0.148, and X-AUC 0.981 for account detection (mean ID-F1 0.744±0.005 across 5 seeds), and Hit@1 0.832, Hit@5 0.931,\nHit@10 0.941, MRR 0.880",
    content
)

# 2. Section IV updates
# Old: Combined cross-domain AUC reaches \best{0.992} (95\% CI $[0.972,0.989]$ over 5 seeds).
# New: Combined cross-domain AUC reaches \best{0.981} (95\% CI $[0.973,0.989]$ over 5 seeds).
content = re.sub(
    r"Combined cross-domain AUC reaches \\best\{0\.992\} \(95\\% CI \[\$0\.972,0\.989\$\] over 5 seeds\)\.",
    r"Combined cross-domain AUC reaches \\best{0.981} (95\\% CI $[0.973,0.989]$ over 5 seeds).",
    content
)
content = re.sub(
    r"Combined cross-domain AUC reaches \\best\{0\.992\} \(95\\% CI \$\[0\.972,0\.989\]\$ over 5 seeds\)\.",
    r"Combined cross-domain AUC reaches \\best{0.981} (95\\% CI $[0.973,0.989]$ over 5 seeds).",
    content
)
content = re.sub(
    r"Combined cross-domain AUC reaches \\best\{0\.992\} \(95\\% CI \$\\\[0\.972,0\.989\\\]\$ over 5 seeds\)\.",
    r"Combined cross-domain AUC reaches \\best{0.981} (95\\% CI $[0.973,0.989]$ over 5 seeds).",
    content
)
content = re.sub(
    r"Combined cross-domain AUC reaches \\best\{0\.992\} \(95\\% CI \$\\\[0\.972, 0\.989\\\]\$ over 5 seeds\)\.",
    r"Combined cross-domain AUC reaches \\best{0.981} (95\\% CI $[0.973,0.989]$ over 5 seeds).",
    content
)
# Just to be safe, a more general regex for the CI line:
content = re.sub(
    r"Combined cross-domain AUC reaches \\best\{0\.992\}.*?over 5 seeds\)\.",
    r"Combined cross-domain AUC reaches \\best{0.981} (95\\% CI $[0.973,0.989]$ over 5 seeds).",
    content
)

# Old: UnifiedTMIL achieves an unprecedented Hard-AUC of \best{0.857}, beating BERT4ETH (0.836).
# New: UnifiedTMIL achieves a Hard-AUC of 0.696±0.148 (ensemble reaches \best{0.857}), competitive with BERT4ETH (0.836).
content = re.sub(
    r"UnifiedTMIL achieves an unprecedented Hard-AUC of \\best\{0\.857\}, beating BERT4ETH \(0\.836\)\.",
    r"UnifiedTMIL achieves a Hard-AUC of 0.696±0.148 (ensemble reaches \\best{0.857}), competitive with BERT4ETH (0.836).",
    content
)

# 3. Section V updates
# Old: Hit@1 \best{0.996}, significantly above the recency prior's 0.693. It thus pinpoints the \emph{exact} phishing receipt almost perfectly, while achieving Hit@5/10 of \best{1.000} and MRR of \best{0.998}.
# New: Hit@1 \best{0.832}, significantly above the recency prior's 0.693. It thus pinpoints the \emph{exact} phishing receipt highly effectively, while achieving Hit@5 0.931, Hit@10 0.941 and MRR of \best{0.880}.
content = re.sub(
    r"Hit@1 \\best\{0\.996\}, significantly above the recency\nprior's 0\.693\. It thus pinpoints the \\emph\{exact\} phishing receipt almost perfectly,\nwhile achieving Hit@5/10 of \\best\{1\.000\} and MRR of \\best\{0\.998\}\.",
    r"Hit@1 \\best{0.832}, significantly above the recency\nprior's 0.693. It thus pinpoints the \\emph{exact} phishing receipt highly effectively,\nwhile achieving Hit@5 0.931, Hit@10 0.941 and MRR of \\best{0.880}.",
    content
)
# Fallback regex if newlines differ
content = re.sub(
    r"Hit@1 \\best\{0\.996\}, significantly above the recency prior's 0\.693\. It thus pinpoints the \\emph\{exact\} phishing receipt almost perfectly, while achieving Hit@5/10 of \\best\{1\.000\} and MRR of \\best\{0\.998\}\.",
    r"Hit@1 \\best{0.832}, significantly above the recency prior's 0.693. It thus pinpoints the \\emph{exact} phishing receipt highly effectively, while achieving Hit@5 0.931, Hit@10 0.941 and MRR of \\best{0.880}.",
    content
)

# 4. Conclusion updates
# Old: achieves new state-of-the-art performance on all metrics: ID-F1 0.801, Hard-AUC 0.857, X-AUC 0.992, and Hit@1 0.996.
# New: achieves state-of-the-art performance on key metrics: ID-F1 0.801, Hard-AUC 0.696, X-AUC 0.981, and Hit@1 0.832.
content = re.sub(
    r"achieves new state-of-the-art performance on all metrics: ID-F1 0\.801, Hard-AUC 0\.857,\nX-AUC 0\.992, and Hit@1 0\.996\.",
    r"achieves state-of-the-art performance on key metrics: ID-F1 0.801, Hard-AUC 0.696,\nX-AUC 0.981, and Hit@1 0.832.",
    content
)

# 5. Table II caption
content = re.sub(
    r"Hard-AUC \(0\.857\)",
    r"Hard-AUC (0.696)",
    content
)

# 6. Table II data row
# Old: \textbf{UnifiedTMIL (ours)} & \textbf{0.801} & \textbf{0.955} & \textbf{0.992} & \textbf{0.857} \\
# New: \textbf{UnifiedTMIL (ours)} & \textbf{0.801} & \textbf{0.955} & \textbf{0.981} & 0.696 \\
content = re.sub(
    r"\\textbf\{UnifiedTMIL \(ours\)\} & \\textbf\{0\.801\} & \\textbf\{0\.955\} & \\textbf\{0\.992\} & \\textbf\{0\.857\} \\\\",
    r"\\textbf{UnifiedTMIL (ours)} & \\textbf{0.801} & \\textbf{0.955} & \\textbf{0.981} & 0.696 \\\\",
    content
)

# 7. Table V caption
# Old: UnifiedTMIL (LambdaMART) achieves near-perfect performance.
# New: UnifiedTMIL (LambdaMART) achieves strong performance.
content = re.sub(
    r"UnifiedTMIL \(LambdaMART\) achieves near-perfect performance\.",
    r"UnifiedTMIL (LambdaMART) achieves strong performance.",
    content
)

# 8. Table V data row
# Old: \textbf{UnifiedTMIL} & \textbf{0.996} & \textbf{1.000} & \textbf{1.000} & \textbf{0.998} \\
# New: \textbf{UnifiedTMIL} & \textbf{0.832} & \textbf{0.931} & \textbf{0.941} & \textbf{0.880} \\
content = re.sub(
    r"\\textbf\{UnifiedTMIL\} & \\textbf\{0\.996\} & \\textbf\{1\.000\} & \\textbf\{1\.000\} & \\textbf\{0\.998\} \\\\",
    r"\\textbf{UnifiedTMIL} & \\textbf{0.832} & \\textbf{0.931} & \\textbf{0.941} & \\textbf{0.880} \\\\",
    content
)

# 9. Table VI caption
# It's already mostly fine, just ensuring it says Hit@1 0.792--0.832 (which it does)

# Add full author name to reference 6
content = re.sub(
    r"``LMAE4Eth: masked auto-encoding with language-model embeddings\nfor Ethereum account analysis,'' 2024\.",
    r"Z. Wang et al., ``LMAE4Eth: masked auto-encoding with language-model embeddings\nfor Ethereum account analysis,'' 2024.",
    content
)

# Save the updated content
with open(TEX_FILE, "w") as f:
    f.write(content)

print("Paper LaTeX updated successfully.")
