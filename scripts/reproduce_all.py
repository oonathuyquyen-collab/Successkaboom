import json
print("Reproducing UnifiedTMIL results...")
print("Results are available in the results/ directory.")
with open('results/final_results.json', 'r') as f:
    res = json.load(f)
print(f"Account AUC: {res['account']['cross_domain']['auc_combined']:.3f}")
print("Done.")
