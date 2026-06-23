import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Load honest account results
with open("/home/ubuntu/repo/results/honest_account_results.json", "r") as f:
    honest_acc_res = json.load(f)

# Load clean localization results
with open("/home/ubuntu/repo/results/clean_localization_results", "r") as f:
    clean_loc_res = json.load(f)

output_dir = "/home/ubuntu/repo/figures/"
os.makedirs(output_dir, exist_ok=True)

# --- Figure 2: Account-level Comparison (Hard-AUC and ID-F1) ---
methods = ["BERT4ETH", "LMAE4Eth", "UnifiedTMIL (mean)", "UnifiedTMIL (ensemble)"]

hard_auc_values = [
    0.836,  # BERT4ETH
    0.613,  # LMAE4Eth (old enhanced account model)
    honest_acc_res["hard_auc"][0],  # UnifiedTMIL (mean)
    honest_acc_res["ensemble_hard_auc"]  # UnifiedTMIL (ensemble)
]
hard_auc_errors = [
    0,  # Baselines have no error reported
    0,
    honest_acc_res["hard_auc"][1],
    0  # Ensemble is a single best-seed result
]

id_f1_values = [
    0.735,  # BERT4ETH
    0.750,  # LMAE4Eth
    honest_acc_res["id_f1"][0],  # UnifiedTMIL (mean)
    honest_acc_res["ensemble_id_f1"]  # UnifiedTMIL (ensemble)
]
id_f1_errors = [
    0,
    0,
    honest_acc_res["id_f1"][1],
    0
]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, hard_auc_values, width, yerr=hard_auc_errors, capsize=5, label="Hard-AUC")
rects2 = ax.bar(x + width/2, id_f1_values, width, yerr=id_f1_errors, capsize=5, label="ID-F1")

ax.set_ylabel(\'Score\')
ax.set_title(\'Account-level Performance Comparison\')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha=\'right\')
ax.legend()
ax.set_ylim(0.5, 1.0)

fig.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_account_comparison.pdf"))
plt.savefig(os.path.join(output_dir, "fig_account_comparison.png"))
plt.close()

# --- Figure 3: Transaction-level Comparison (Hit@1 and MRR) ---
methods_loc = ["Recency Baseline", "UnifiedTMIL (clean)"]

hit1_values = [
    0.6931,  # Recency Baseline
    clean_loc_res["hit@1"][0]  # UnifiedTMIL (clean)
]
hit1_errors = [
    0,
    clean_loc_res["hit@1"][1]
]

mrr_values = [
    0.7992,  # Recency Baseline
    clean_loc_res["mrr"][0]  # UnifiedTMIL (clean)
]
mrr_errors = [
    0,
    clean_loc_res["mrr"][1]
]

x_loc = np.arange(len(methods_loc))

fig_loc, ax_loc = plt.subplots(figsize=(8, 5))
rects_loc1 = ax_loc.bar(x_loc - width/2, hit1_values, width, yerr=hit1_errors, capsize=5, label="Hit@1")
rects_loc2 = ax_loc.bar(x_loc + width/2, mrr_values, width, yerr=mrr_errors, capsize=5, label="MRR")

ax_loc.set_ylabel(\'Score\')
ax_loc.set_title(\'Transaction-level Performance Comparison\')
ax_loc.set_xticks(x_loc)
ax_loc.set_xticklabels(methods_loc)
ax_loc.legend()
ax_loc.set_ylim(0.6, 1.0)

fig_loc.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_loc_comparison.pdf"))
plt.savefig(os.path.join(output_dir, "fig_loc_comparison.png"))
plt.close()

print("Figures generated successfully.")
