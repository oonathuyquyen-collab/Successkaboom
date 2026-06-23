import os
import sys
import json
import pickle
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

ROOT = "/home/ubuntu/repo"
DATA = os.path.join(ROOT, "data")
BERT4ETH = os.path.join(DATA, "bert4eth")
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))
import vocab_def
sys.modules['__main__'].Vocab = vocab_def.Vocab

SEEDS = [42, 1337, 7, 99, 2024]

def boot_ci(vals, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    stats = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]

def main():
    print("=" * 60)
    print("UnifiedTMIL Honest Account-Level Re-evaluation")
    print("=" * 60)
    
    # Load multi-seed results if they exist, otherwise we'd need to retrain.
    # Since retraining takes time, let's first check if the per-seed data is in the JSON.
    json_path = os.path.join(RES, "unified_tmil_multiseed.json")
    if not os.path.exists(json_path):
        print("Error: unified_tmil_multiseed.json not found. Run training first.")
        return
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # 1. Account-Level ID (In-Domain)
    id_f1s = data["tmil_multi_seed"]["in_domain"]["per_seed"]
    print(f"In-Domain F1: {np.mean(id_f1s):.4f} ± {np.std(id_f1s):.4f}")
    
    # 2. Account-Level Hard-AUC (PTX Benign)
    hard_aucs = data["tmil_multi_seed"]["hard_auc"]["per_seed"]
    print(f"Hard-AUC:     {np.mean(hard_aucs):.4f} ± {np.std(hard_aucs):.4f}")
    
    # 3. Account-Level X-AUC (PTX Total)
    x_aucs = data["tmil_multi_seed"]["x_auc"]["per_seed"]
    print(f"X-AUC:        {np.mean(x_aucs):.4f} ± {np.std(x_aucs):.4f}")
    
    # Fix the CI for X-AUC (the one reported as [0.972, 0.989] for 0.992)
    # The 0.992 was likely the ENSEMBLE result, not the mean.
    # Let's compute the CI for the ENSEMBLE result using bootstrap on the test set.
    
    # To do this honestly, we need the actual predictions of the ensemble.
    # We can load them from loc_scores.pkl or similar if available, or ptx_bags.
    
    print("\nReporting HONEST headlines (mean ± std):")
    print(f"  ID-F1:    {np.mean(id_f1s):.4f} ± {np.std(id_f1s):.4f}")
    print(f"  Hard-AUC: {np.mean(hard_aucs):.4f} ± {np.std(hard_aucs):.4f}")
    print(f"  X-AUC:    {np.mean(x_aucs):.4f} ± {np.std(x_aucs):.4f}")

    # Create final honest results file
    honest_res = {
        "id_f1": [float(np.mean(id_f1s)), float(np.std(id_f1s))],
        "hard_auc": [float(np.mean(hard_aucs)), float(np.std(hard_aucs))],
        "x_auc": [float(np.mean(x_aucs)), float(np.std(x_aucs))],
        "ensemble_hard_auc": data["tmil_ensemble"]["hard_auc"],
        "ensemble_x_auc": data["tmil_ensemble"]["x_auc"],
        "ensemble_id_f1": data["tmil_ensemble"]["in_domain"]["f1"]
    }
    with open(os.path.join(RES, "honest_account_results.json"), "w") as f:
        json.dump(honest_res, f, indent=2)

if __name__ == "__main__":
    main()
