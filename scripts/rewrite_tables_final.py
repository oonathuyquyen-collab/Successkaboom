import re
import os
import json

tex_file = "/home/ubuntu/repo/paper/paper_en.tex"
with open(tex_file, "r") as f:
    content = f.read()

# Load honest account results
with open("/home/ubuntu/repo/results/honest_account_results.json", "r") as f:
    honest_acc_res = json.load(f)

# Load clean localization results
with open("/home/ubuntu/repo/results/clean_localization_results.json", "r") as f:
    clean_loc_res = json.load(f)

# --- Table: Account Results (tab:comp) ---
id_f1_mean = honest_acc_res["id_f1"][0]
id_f1_std = honest_acc_res["id_f1"][1]
hard_auc_mean = honest_acc_res["hard_auc"][0]
hard_auc_std = honest_acc_res["hard_auc"][1]
x_auc_mean = honest_acc_res["x_auc"][0]
x_auc_std = honest_acc_res["x_auc"][1]

bert4eth_id_f1 = 0.735
bert4eth_hard_auc = 0.836
bert4eth_x_auc = 0.984

lmae4eth_id_f1 = 0.750
lmae4eth_hard_auc = 0.613
lmae4eth_x_auc = 0.982

ensemble_hard_auc = honest_acc_res["ensemble_hard_auc"]
ensemble_x_auc = honest_acc_res["ensemble_x_auc"]
ensemble_id_f1 = honest_acc_res["ensemble_id_f1"]

tab_comp_content = (
    "\\begin{table}[t]\n"
    "\\caption{Account-level performance comparison on PTXPhish.}\n"
    "\\label{tab:comp}\n"
    "\\centering\n"
    "\\begin{tabular}{l c c c}\n"
    "\\toprule\n"
    "Method & ID-F1 & Hard-AUC & X-AUC \\\\\n"
    "\\midrule\n"
    "BERT4ETH & %.3f & %.3f & %.3f \\\\\n"
    "LMAE4Eth & %.3f & %.3f & %.3f \\\\\n"
    "\\midrule\n"
    "UnifiedTMIL (mean\\pm std) & %.3f\\pm%.3f & %.3f\\pm%.3f & %.3f\\pm%.3f \\\\\n"
    "UnifiedTMIL (ensemble) & \\textbf{%.3f} & \\textbf{%.3f} & \\textbf{%.3f} \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
) % (
    bert4eth_id_f1, bert4eth_hard_auc, bert4eth_x_auc,
    lmae4eth_id_f1, lmae4eth_hard_auc, lmae4eth_x_auc,
    id_f1_mean, id_f1_std, hard_auc_mean, hard_auc_std, x_auc_mean, x_auc_std,
    ensemble_id_f1, ensemble_hard_auc, ensemble_x_auc
)

content = re.sub(r"\\begin{table}\[t\](.+?)label{tab:comp}(.+?)end{table}", tab_comp_content.replace("\\", "\\\\"), content, flags=re.DOTALL)

# --- Table: PRF (tab:prf) ---
# Hardcode these from unified_tmil_multiseed.json as they are not in honest_account_results.json
ensemble_id_precision = 0.729
ensemble_id_recall = 0.887
ensemble_id_f1_prf = honest_acc_res["ensemble_id_f1"] # Use the same f1 as above
ensemble_id_auc = 0.955
ensemble_id_aupr = 0.883

tab_prf_content = (
    "\\begin{table}[t]\n"
    "\\caption{In-domain PRF metrics for UnifiedTMIL ensemble.}\n"
    "\\label{tab:prf}\n"
    "\\centering\n"
    "\\begin{tabular}{l c c c c c}\n"
    "\\toprule\n"
    "Method & Precision & Recall & F1 & AUC & AUPR \\\\\n"
    "\\midrule\n"
    "UnifiedTMIL (ensemble) & %.3f & %.3f & %.3f & %.3f & %.3f \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
) % (
    ensemble_id_precision, ensemble_id_recall, ensemble_id_f1_prf, ensemble_id_auc, ensemble_id_aupr
)

content = re.sub(r"\\begin{table}\[t\](.+?)label{tab:prf}(.+?)end{table}", tab_prf_content.replace("\\", "\\\\"), content, flags=re.DOTALL)

# --- Table: Localization (tab:loc) ---
loc_hit1_mean = clean_loc_res["hit@1"][0]
loc_hit1_std = clean_loc_res["hit@1"][1]
loc_mrr_mean = clean_loc_res["mrr"][0]
loc_mrr_std = clean_loc_res["mrr"][1]

recency_hit1 = 0.6931
recency_mrr = 0.7992

tab_loc_content = (
    "\\begin{table}[t]\n"
    "\\caption{Transaction-level localization performance on PTXPhish.}\n"
    "\\label{tab:loc}\n"
    "\\centering\n"
    "\\begin{tabular}{l c c}\n"
    "\\toprule\n"
    "Method & Hit@1 & MRR \\\\\n"
    "\\midrule\n"
    "Recency Baseline & %.3f & %.3f \\\\\n"
    "UnifiedTMIL (clean) & \\textbf{%.3f}\\pm%.3f & \\textbf{%.3f}\\pm%.3f \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
) % (
    recency_hit1, recency_mrr,
    loc_hit1_mean, loc_hit1_std, loc_mrr_mean, loc_mrr_std
)

content = re.sub(r"\\begin{table}\[t\](.+?)label{tab:loc}(.+?)end{table}", tab_loc_content.replace("\\", "\\\\"), content, flags=re.DOTALL)

with open(tex_file, "w") as f:
    f.write(content)

print("Tables updated with honest narrative and booktabs formatting.")
