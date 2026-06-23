import re
import os
import json

tex_file = "/home/ubuntu/repo/paper/paper_en.tex"
with open(tex_file, "r") as f:
    content = f.read()

# 1. Naming Consistency: Unified TMIL -> UnifiedTMIL
content = re.sub(r'Unified TMIL', 'UnifiedTMIL', content)

# 2. Update Abstract with honest numbers and CIs
with open('/home/ubuntu/repo/results/honest_account_results.json', 'r') as f:
    honest_acc_res = json.load(f)
with open('/home/ubuntu/repo/results/clean_localization_results.json', 'r') as f:
    clean_loc_res = json.load(f)

id_f1_mean = honest_acc_res['id_f1'][0]
id_f1_std = honest_acc_res['id_f1'][1]
hard_auc_mean = honest_acc_res['hard_auc'][0]
hard_auc_std = honest_acc_res['hard_auc'][1]
x_auc_mean = honest_acc_res['x_auc'][0]
x_auc_std = honest_acc_res['x_auc'][1]

loc_hit1_mean = clean_loc_res['hit@1'][0]
loc_hit1_std = clean_loc_res['hit@1'][1]
loc_mrr_mean = clean_loc_res['mrr'][0]
loc_mrr_std = clean_loc_res['mrr'][1]

old_abstract_pattern = r"We study Ethereum phishing at two granularities with a single model: deciding\nwhether an account is a scammer (account level) and pinpointing the fraudulent\ntransactions inside that account's history (transaction level). We propose\n\\emph\\{Unified TMIL\\}, a unified framework that solves both tasks simultaneously\nusing a single set of lightweight aggregate features. By incorporating an\nensemble meta-learner and LambdaMART ranking, UnifiedTMIL achieves new\nstate-of-the-art results across all metrics on the PTXPhish benchmark: ID-F1 0.801,\nHard-AUC 0.857, and X-AUC 0.992 for account detection (mean ID-F1 0.744\\pm0.005, Hard-AUC 0.696\\pm0.148 across 5 seeds), and Hit@1 0.996, Hit@5 1.000,\nHit@10 1.000, MRR 0.998 for transaction localization. We contribute three\nthings. (i) \\emph\\{Unified TMIL\\}: a shared encoder with two attention read-out\nheads, enhanced by a LightGBM/XGBoost ensemble and LambdaMART learning-to-rank.\n(ii) An \\emph\\{audited benchmark\\} built from PTXPhish, including a contract-mediated\n\\emph\\{relabeling protocol\\} that traces execution receipts and raises clean ground\ntruth to 92.7\\%, yielding 292 deduplicated scammer EOAs.\n(iii) An \\emph\\{honest evaluation\\}: multi-seed runs, cluster-aware bootstrap\nconfidence intervals, by-source negative breakdowns, and out-of-distribution\nsplits. Our results demonstrate that careful feature engineering and ranking\nalgorithms significantly outperform pure attention-based localization. We release\nthe audited benchmark and protocol as an open challenge."

# Construct the new abstract text with numerical values first, then apply LaTeX escapes
new_abstract_text_raw = (
    f"We study Ethereum phishing at two granularities with a single model: deciding\n"
    f"whether an account is a scammer (account level) and pinpointing the fraudulent\n"
    f"transactions inside that account's history (transaction level). We propose\n"
    f"UnifiedTMIL, a unified framework that solves both tasks simultaneously\n"
    f"using a single set of lightweight aggregate features. By incorporating an\n"
    f"ensemble meta-learner and LambdaMART ranking, UnifiedTMIL achieves state-of-the-art\n"
    f"results across all metrics on the PTXPhish benchmark. For account detection, we report\n"
    f"ID-F1 {id_f1_mean:.3f}+-{id_f1_std:.3f}, Hard-AUC {hard_auc_mean:.3f}+-{hard_auc_std:.3f}, and X-AUC {x_auc_mean:.3f}+-{x_auc_std:.3f} (mean+- std across 5 seeds). For transaction localization,\n"
    f"we achieve Hit@1 {loc_hit1_mean:.3f}+-{loc_hit1_std:.3f} and MRR {loc_mrr_mean:.3f}+-{loc_mrr_std:.3f}. We contribute three\n"
    f"things. (i) UnifiedTMIL: a shared encoder with two attention read-out\n"
    f"heads, enhanced by a LightGBM/XGBoost ensemble and LambdaMART learning-to-rank.\n"
    f"(ii) An audited benchmark built from PTXPhish, including a contract-mediated\n"
    f"relabeling protocol that traces execution receipts and raises clean ground\n"
    f"truth to 92.7%, yielding 292 deduplicated scammer EOAs.\n"
    f"(iii) An honest evaluation: multi-seed runs, cluster-aware bootstrap\n"
    f"confidence intervals, by-source negative breakdowns, and out-of-distribution\n"
    f"splits. Our results demonstrate that careful feature engineering and ranking\n"
    f"algorithms significantly outperform pure attention-based localization. We release\n"
    f"the audited benchmark and protocol as an open challenge."
)

# Apply LaTeX specific escapes to the raw text
new_abstract_text = new_abstract_text_raw.replace('+-', '\\pm')
new_abstract_text = new_abstract_text.replace('UnifiedTMIL', '\\emph\\{UnifiedTMIL\\}')
new_abstract_text = new_abstract_text.replace('audited benchmark', '\\emph\\{audited benchmark\\}')
new_abstract_text = new_abstract_text.replace('relabeling protocol', '\\emph\\{relabeling protocol\\}')
new_abstract_text = new_abstract_text.replace('honest evaluation', '\\emph\\{honest evaluation\\}')
new_abstract_text = new_abstract_text.replace('92.7%', '92.7\\%')

# Now, escape backslashes for re.sub to treat them literally in the replacement string
new_abstract_text_escaped = new_abstract_text.replace('\\', '\\\\')

content = re.sub(old_abstract_pattern, new_abstract_text_escaped, content, flags=re.DOTALL)

# 7. Ensure author is Anonymous Submission
content = content.replace(r'\\author{\\IEEEauthorblockN{Anonymous Submission}}', r'\\author{\\IEEEauthorblockN{Anonymous Submission}}')

with open(tex_file, "w") as f:
    f.write(content)

print("Paper updated with honest narrative and naming consistency.")
