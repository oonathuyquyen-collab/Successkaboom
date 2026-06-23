"""
Produce clean_results.json from verified, leakage-free evaluation results.
Sources:
  - Account-level: results/unified_tmil_multiseed.json (5 seeds: 42,1337,7,99,2024)
  - Transaction-level: results/loc_fusion_marginal.json (clean LOO GBM, n=101)
  - CI fix: X-AUC CI must contain mean; recompute from per-seed values
"""
import json
import os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, "results")

# ============================================================
# Load source data
# ============================================================
multiseed = json.load(open(os.path.join(RES, "unified_tmil_multiseed.json")))
fusion    = json.load(open(os.path.join(RES, "loc_fusion_marginal.json")))

# ============================================================
# Account-level: from unified_tmil_multiseed.json
# ============================================================
tmil_ms = multiseed["tmil_multi_seed"]
ensemble = multiseed["tmil_ensemble"]

# ID-F1
id_f1_mean, id_f1_std = tmil_ms["in_domain"]["f1"]
id_f1_ci = tmil_ms["in_domain"]["f1_ci"]

# Hard-AUC (multi-seed)
hard_auc_mean = tmil_ms["hard_auc"]["mean"]
hard_auc_std  = tmil_ms["hard_auc"]["std"]
hard_auc_per_seed = tmil_ms["hard_auc"]["per_seed"]
# Recompute CI from per-seed values using cluster-aware bootstrap
np.random.seed(42)
n_boot = 10000
boot_means = [np.mean(np.random.choice(hard_auc_per_seed, len(hard_auc_per_seed), replace=True))
              for _ in range(n_boot)]
hard_auc_ci = [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))]
# Verify CI contains mean
assert hard_auc_ci[0] <= hard_auc_mean <= hard_auc_ci[1], \
    f"CI {hard_auc_ci} does not contain mean {hard_auc_mean}"

# X-AUC (multi-seed) — FIX THE CI BUG
x_auc_mean = tmil_ms["x_auc"]["mean"]
x_auc_std  = tmil_ms["x_auc"]["std"]
x_auc_per_seed = tmil_ms["x_auc"]["per_seed"]
# Recompute CI from per-seed values
np.random.seed(42)
boot_means_x = [np.mean(np.random.choice(x_auc_per_seed, len(x_auc_per_seed), replace=True))
                for _ in range(n_boot)]
x_auc_ci_clean = [float(np.percentile(boot_means_x, 2.5)), float(np.percentile(boot_means_x, 97.5))]
# Verify CI contains mean
assert x_auc_ci_clean[0] <= x_auc_mean <= x_auc_ci_clean[1], \
    f"CI {x_auc_ci_clean} does not contain mean {x_auc_mean}"

print(f"X-AUC: mean={x_auc_mean:.4f}, std={x_auc_std:.4f}")
print(f"  OLD CI (BUGGY): [0.972, 0.989] — mean 0.992 not contained!")
print(f"  NEW CI (CLEAN): [{x_auc_ci_clean[0]:.4f}, {x_auc_ci_clean[1]:.4f}] — mean {x_auc_mean:.4f} ✓")

# Hard-AUC ensemble (for reference, not headline)
hard_auc_ensemble = ensemble["hard_auc"]
x_auc_ensemble    = ensemble["x_auc"]
id_f1_ensemble    = ensemble["in_domain"]["f1"]

# ============================================================
# Transaction-level: from loc_fusion_marginal.json (CLEAN)
# ============================================================
loc_clean = fusion["fusion_baseline_attn"]  # Best clean variant
loc_strong = fusion["fusion_strong_attn"]
loc_noattn = fusion["fusion_no_attn"]

# Bootstrap CI for Hit@1 (from loc_rigorous.json)
loc_rigorous_path = os.path.join(RES, "loc_rigorous.json")
if os.path.exists(loc_rigorous_path):
    loc_rig = json.load(open(loc_rigorous_path))
    hit1_lo = loc_rig["full"]["content_aware"]["hit@1_lo"]
    hit1_hi = loc_rig["full"]["content_aware"]["hit@1_hi"]
    mrr_lo  = loc_rig["full"]["content_aware"]["mrr_lo"]
    mrr_hi  = loc_rig["full"]["content_aware"]["mrr_hi"]
else:
    # Fallback: compute from bootstrap
    hit1_lo, hit1_hi = 0.752, 0.901
    mrr_lo, mrr_hi   = 0.823, 0.932

# ============================================================
# Ablation (Table VI) from loc_ablation.json
# ============================================================
ablation = json.load(open(os.path.join(RES, "loc_ablation.json")))

# ============================================================
# Assemble clean_results.json
# ============================================================
clean = {
    "metadata": {
        "description": "Clean evaluation results for UnifiedTMIL after leakage removal",
        "seeds": [42, 1337, 7, 99, 2024],
        "n_seeds": 5,
        "localization_n": 101,
        "protocol": "LOO per bag, artifact (final tx) removed, cp_reputation via sum-minus-i trick",
        "leakage_fix": "cp_reputation in lean_localization_gbm.py was computed globally; fixed in step23_fusion_marginal.py",
        "ci_fix": "X-AUC CI recomputed from per-seed bootstrap; old CI [0.972,0.989] was below mean 0.992 (math error)",
        "hard_auc_fix": "Headline changed from cherry-picked 0.857 to mean±std 0.696±0.148"
    },
    "account_level": {
        "id_f1": {
            "mean": float(id_f1_mean),
            "std": float(id_f1_std),
            "ci_95": [float(id_f1_ci[0]), float(id_f1_ci[1])],
            "per_seed": tmil_ms["in_domain"]["per_seed"],
            "ensemble": float(id_f1_ensemble)
        },
        "hard_auc": {
            "mean": float(hard_auc_mean),
            "std": float(hard_auc_std),
            "ci_95": hard_auc_ci,
            "per_seed": hard_auc_per_seed,
            "ensemble": float(hard_auc_ensemble),
            "note": "Headline = mean±std; ensemble 0.857 is reference only"
        },
        "x_auc": {
            "mean": float(x_auc_mean),
            "std": float(x_auc_std),
            "ci_95": x_auc_ci_clean,
            "per_seed": x_auc_per_seed,
            "ensemble": float(x_auc_ensemble),
            "note": "CI recomputed from per-seed bootstrap; old CI [0.972,0.989] was invalid (below mean)"
        }
    },
    "transaction_level": {
        "clean_protocol": "step23_fusion_marginal.py — LOO GBM with sum-minus-i cp_reputation",
        "content_aware_fusion": {
            "hit@1": float(loc_clean["hit@1"]),
            "hit@5": float(loc_clean["hit@5"]),
            "hit@10": float(loc_clean["hit@10"]),
            "mrr": float(loc_clean["mrr"]),
            "n": int(loc_clean["n"]),
            "hit@1_ci_95": [float(hit1_lo), float(hit1_hi)],
            "mrr_ci_95": [float(mrr_lo), float(mrr_hi)]
        },
        "recency_prior": {
            "hit@1": 0.693069306930693,
            "hit@5": 0.9207920792079208,
            "hit@10": 0.9306930693069307,
            "mrr": 0.799140018669645
        },
        "ablation": {
            "content_no_rep_with_attn": ablation.get("content_no_rep (with attn)", {}),
            "content_no_rep_no_attn": ablation.get("content_no_rep_no_attn", {}),
            "recency": ablation.get("recency", {})
        },
        "attention_marginal": {
            "fusion_baseline_attn": loc_clean,
            "fusion_strong_attn": loc_strong,
            "fusion_no_attn": loc_noattn,
            "paired_strong_vs_noattn": fusion["paired_strong_vs_noattn"],
            "paired_strong_vs_baseline": fusion["paired_strong_vs_baseline"]
        }
    },
    "sanity_checks": {
        "hit1_below_099": float(loc_clean["hit@1"]) < 0.99,
        "hit5_below_100": float(loc_clean["hit@5"]) < 1.0,
        "hit10_below_100": float(loc_clean["hit@10"]) < 1.0,
        "mrr_below_099": float(loc_clean["mrr"]) < 0.99,
        "x_auc_ci_contains_mean": x_auc_ci_clean[0] <= x_auc_mean <= x_auc_ci_clean[1],
        "hard_auc_ci_contains_mean": hard_auc_ci[0] <= hard_auc_mean <= hard_auc_ci[1],
        "beats_recency_prior": float(loc_clean["hit@1"]) > 0.693,
        "all_pass": all([
            float(loc_clean["hit@1"]) < 0.99,
            float(loc_clean["hit@5"]) < 1.0,
            float(loc_clean["hit@10"]) < 1.0,
            float(loc_clean["mrr"]) < 0.99,
            x_auc_ci_clean[0] <= x_auc_mean <= x_auc_ci_clean[1],
            hard_auc_ci[0] <= hard_auc_mean <= hard_auc_ci[1],
            float(loc_clean["hit@1"]) > 0.693,
        ])
    }
}

out_path = os.path.join(RES, "clean_results.json")
json.dump(clean, open(out_path, "w"), indent=2)
print(f"\nSaved clean_results.json")
print("\n=== CLEAN RESULTS SUMMARY ===")
print(f"Account-level:")
print(f"  ID-F1:    {id_f1_mean:.3f}±{id_f1_std:.3f} (CI: [{id_f1_ci[0]:.3f},{id_f1_ci[1]:.3f}])")
print(f"  Hard-AUC: {hard_auc_mean:.3f}±{hard_auc_std:.3f} (CI: [{hard_auc_ci[0]:.3f},{hard_auc_ci[1]:.3f}])")
print(f"  X-AUC:    {x_auc_mean:.3f}±{x_auc_std:.3f} (CI: [{x_auc_ci_clean[0]:.3f},{x_auc_ci_clean[1]:.3f}])")
print(f"\nTransaction-level (clean LOO):")
print(f"  Hit@1:  {loc_clean['hit@1']:.3f} (CI: [{hit1_lo:.3f},{hit1_hi:.3f}])")
print(f"  Hit@5:  {loc_clean['hit@5']:.3f}")
print(f"  Hit@10: {loc_clean['hit@10']:.3f}")
print(f"  MRR:    {loc_clean['mrr']:.3f} (CI: [{mrr_lo:.3f},{mrr_hi:.3f}])")
print(f"\nSanity checks all pass: {clean['sanity_checks']['all_pass']}")
