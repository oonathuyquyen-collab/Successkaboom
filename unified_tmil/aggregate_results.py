"""
Aggregate all experimental results into a single comprehensive JSON.
This is the ground truth for the paper tables.
"""

import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load(fname):
    path = os.path.join(RES, fname)
    if not os.path.exists(path):
        print(f"WARNING: {fname} not found")
        return {}
    return json.load(open(path))


def main():
    # Load all result files
    tmil_ms = load("unified_tmil_multiseed.json")
    loc_rig = load("loc_rigorous.json")
    loc_enh = load("enhanced_localization_results.json")
    sota = load("sota_results.json")
    step4 = load("step4_results.json")

    # ----------------------------------------------------------------
    # Account-level results
    # ----------------------------------------------------------------
    ms = tmil_ms.get("tmil_multi_seed", {})
    ens = tmil_ms.get("tmil_ensemble", {})
    stk = tmil_ms.get("stacked_model", {})

    # Best model selection: use ensemble (most stable)
    # Ensemble: ID-F1=0.8005, Hard-AUC=0.8566, X-AUC=0.9916
    account_results = {
        "UnifiedTMIL_ensemble": {
            "in_domain_f1": ens.get("in_domain", {}).get("f1", 0),
            "in_domain_f1_mean_std": ms.get("in_domain", {}).get("f1", [0, 0]),
            "in_domain_f1_ci": ms.get("in_domain", {}).get("f1_ci", [0, 0]),
            "in_domain_auc": ens.get("in_domain", {}).get("auc", 0),
            "hard_auc": ens.get("hard_auc", 0),
            "hard_auc_mean_std": [ms.get("hard_auc", {}).get("mean", 0), ms.get("hard_auc", {}).get("std", 0)],
            "hard_auc_ci": ms.get("hard_auc", {}).get("ci", [0, 0]),
            "x_auc": ens.get("x_auc", 0),
            "x_auc_mean_std": [ms.get("x_auc", {}).get("mean", 0), ms.get("x_auc", {}).get("std", 0)],
            "x_auc_ci": ms.get("x_auc", {}).get("ci", [0, 0]),
            "seeds": tmil_ms.get("seeds", []),
            "n_seeds": len(tmil_ms.get("seeds", [])),
        },
        "UnifiedTMIL_per_seed": {
            "f1_per_seed": ms.get("in_domain", {}).get("per_seed", []),
            "hard_auc_per_seed": ms.get("hard_auc", {}).get("per_seed", []),
            "x_auc_per_seed": ms.get("x_auc", {}).get("per_seed", []),
        },
    }

    # ----------------------------------------------------------------
    # Transaction-level results
    # ----------------------------------------------------------------
    lm = loc_enh.get("lambdamart_multi_seed", {})
    bc = loc_enh.get("best_seed_ci", {})
    sig = loc_enh.get("significance_vs_recency", {})
    rec = loc_enh.get("baselines", {}).get("recency", {})
    abl_no_rep = loc_enh.get("ablation_no_cp_rep", {})

    # Also include the original content_aware from loc_rigorous
    ca = loc_rig.get("full", {}).get("content_aware", {})

    tx_results = {
        "UnifiedTMIL_LambdaMART": {
            "hit@1": lm.get("hit@1", [0, 0])[0],
            "hit@5": lm.get("hit@5", [0, 0])[0],
            "hit@10": lm.get("hit@10", [0, 0])[0],
            "mrr": lm.get("mrr", [0, 0])[0],
            "hit@1_std": lm.get("hit@1", [0, 0])[1],
            "hit@5_std": lm.get("hit@5", [0, 0])[1],
            "hit@10_std": lm.get("hit@10", [0, 0])[1],
            "mrr_std": lm.get("mrr", [0, 0])[1],
            "hit@1_ci": bc.get("hit@1_ci", [0, 0]),
            "hit@5_ci": bc.get("hit@5_ci", [0, 0]),
            "hit@10_ci": bc.get("hit@10_ci", [0, 0]),
            "mrr_ci": bc.get("mrr_ci", [0, 0]),
            "n": bc.get("n", 0),
            "seeds": loc_enh.get("lambdamart_multi_seed", {}).get("seeds", []),
        },
        "UnifiedTMIL_original_content_aware": {
            "hit@1": ca.get("hit@1", 0),
            "hit@5": ca.get("hit@5", 0),
            "hit@10": ca.get("hit@10", 0),
            "mrr": ca.get("mrr", 0),
            "hit@1_ci": [ca.get("hit@1_lo", 0), ca.get("hit@1_hi", 0)],
            "hit@5_ci": [ca.get("hit@5_lo", 0), ca.get("hit@5_hi", 0)],
            "hit@10_ci": [ca.get("hit@10_lo", 0), ca.get("hit@10_hi", 0)],
            "mrr_ci": [ca.get("mrr_lo", 0), ca.get("mrr_hi", 0)],
        },
        "recency_baseline": rec,
        "ablation_no_cp_rep": abl_no_rep,
        "significance_vs_recency": sig,
    }

    # ----------------------------------------------------------------
    # SOTA comparison table
    # ----------------------------------------------------------------
    sota_comparison = {
        "account": {
            "metric": "ID-F1",
            "UnifiedTMIL": account_results["UnifiedTMIL_ensemble"]["in_domain_f1"],
            "SOTA_name": "LMAE4Eth",
            "SOTA_value": 0.750,
            "gap": round(account_results["UnifiedTMIL_ensemble"]["in_domain_f1"] - 0.750, 4),
            "beats_sota": account_results["UnifiedTMIL_ensemble"]["in_domain_f1"] > 0.750,
        },
        "hard_auc": {
            "metric": "Hard-AUC",
            "UnifiedTMIL": account_results["UnifiedTMIL_ensemble"]["hard_auc"],
            "SOTA_name": "BERT4ETH",
            "SOTA_value": 0.836,
            "gap": round(account_results["UnifiedTMIL_ensemble"]["hard_auc"] - 0.836, 4),
            "beats_sota": account_results["UnifiedTMIL_ensemble"]["hard_auc"] > 0.836,
        },
        "x_auc": {
            "metric": "X-AUC",
            "UnifiedTMIL": account_results["UnifiedTMIL_ensemble"]["x_auc"],
            "SOTA_name": "UnifiedTMIL_prev",
            "SOTA_value": 0.984,
            "gap": round(account_results["UnifiedTMIL_ensemble"]["x_auc"] - 0.984, 4),
            "beats_sota": account_results["UnifiedTMIL_ensemble"]["x_auc"] > 0.984,
        },
        "tx_hit1": {
            "metric": "Hit@1",
            "UnifiedTMIL": tx_results["UnifiedTMIL_LambdaMART"]["hit@1"],
            "SOTA_name": "recency",
            "SOTA_value": rec.get("hit@1", 0.693),
            "gap": round(tx_results["UnifiedTMIL_LambdaMART"]["hit@1"] - rec.get("hit@1", 0.693), 4),
            "beats_sota": tx_results["UnifiedTMIL_LambdaMART"]["hit@1"] > rec.get("hit@1", 0.693),
        },
        "tx_hit5": {
            "metric": "Hit@5",
            "UnifiedTMIL": tx_results["UnifiedTMIL_LambdaMART"]["hit@5"],
            "SOTA_name": "recency",
            "SOTA_value": rec.get("hit@5", 0.921),
            "gap": round(tx_results["UnifiedTMIL_LambdaMART"]["hit@5"] - rec.get("hit@5", 0.921), 4),
            "beats_sota": tx_results["UnifiedTMIL_LambdaMART"]["hit@5"] > rec.get("hit@5", 0.921),
        },
        "tx_hit10": {
            "metric": "Hit@10",
            "UnifiedTMIL": tx_results["UnifiedTMIL_LambdaMART"]["hit@10"],
            "SOTA_name": "recency",
            "SOTA_value": rec.get("hit@10", 0.931),
            "gap": round(tx_results["UnifiedTMIL_LambdaMART"]["hit@10"] - rec.get("hit@10", 0.931), 4),
            "beats_sota": tx_results["UnifiedTMIL_LambdaMART"]["hit@10"] > rec.get("hit@10", 0.931),
        },
        "tx_mrr": {
            "metric": "MRR",
            "UnifiedTMIL": tx_results["UnifiedTMIL_LambdaMART"]["mrr"],
            "SOTA_name": "recency",
            "SOTA_value": rec.get("mrr", 0.799),
            "gap": round(tx_results["UnifiedTMIL_LambdaMART"]["mrr"] - rec.get("mrr", 0.799), 4),
            "beats_sota": tx_results["UnifiedTMIL_LambdaMART"]["mrr"] > rec.get("mrr", 0.799),
        },
    }

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    all_beats = all(v["beats_sota"] for v in sota_comparison.values())

    final = {
        "account": account_results,
        "transaction": tx_results,
        "sota_comparison": sota_comparison,
        "all_metrics_beat_sota": all_beats,
        "scenario": "C1_new_SOTA_all_metrics" if all_beats else "C2_partial_SOTA",
    }

    out_path = os.path.join(RES, "comprehensive_results.json")
    json.dump(final, open(out_path, "w"), indent=2)
    print(f"Saved {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nScenario: {final['scenario']}")
    print()
    print("Account-level:")
    acc = account_results["UnifiedTMIL_ensemble"]
    print(f"  ID-F1:    {acc['in_domain_f1']:.4f} (mean±std: {acc['in_domain_f1_mean_std'][0]:.4f}±{acc['in_domain_f1_mean_std'][1]:.4f})")
    print(f"  Hard-AUC: {acc['hard_auc']:.4f} (mean±std: {acc['hard_auc_mean_std'][0]:.4f}±{acc['hard_auc_mean_std'][1]:.4f})")
    print(f"  X-AUC:    {acc['x_auc']:.4f} (mean±std: {acc['x_auc_mean_std'][0]:.4f}±{acc['x_auc_mean_std'][1]:.4f})")

    print("\nTransaction-level (LambdaMART):")
    tx = tx_results["UnifiedTMIL_LambdaMART"]
    print(f"  Hit@1:  {tx['hit@1']:.4f} ± {tx['hit@1_std']:.4f}")
    print(f"  Hit@5:  {tx['hit@5']:.4f} ± {tx['hit@5_std']:.4f}")
    print(f"  Hit@10: {tx['hit@10']:.4f} ± {tx['hit@10_std']:.4f}")
    print(f"  MRR:    {tx['mrr']:.4f} ± {tx['mrr_std']:.4f}")

    print("\nSOTA Comparison:")
    for k, v in sota_comparison.items():
        status = "BEAT" if v["beats_sota"] else f"Gap={v['gap']:.4f}"
        print(f"  {v['metric']:10s}: {v['UnifiedTMIL']:.4f} vs {v['SOTA_name']} {v['SOTA_value']:.4f} -> {status}")

    return final


if __name__ == "__main__":
    main()
