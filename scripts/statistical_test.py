#!/usr/bin/env python3
"""
statistical_test.py
===================
Paired bootstrap test and paired t-test comparing:
  - UnifiedTMIL v2 (SOTA) vs UnifiedTMIL v1 vs 2-head old
  
Per Bước 6 of the SOTA improvement plan.

Usage:
    python3 scripts/statistical_test.py

Outputs:
    results/statistical_tests.json
    results/tables/statistical_test_table.tex
"""

import os, sys, json, pickle
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, "results")
TABLES = os.path.join(RES, "tables")
os.makedirs(TABLES, exist_ok=True)

# ─────────────────────────────────────────────
# Per-seed results from logs
# ─────────────────────────────────────────────

# v1 baseline: from clean_results.json (5 seeds)
V1_ID_F1 = [0.7458745874587459, 0.7439670394349618, 0.7422434367541766,
             0.7519280205655527, 0.7358722358722358]
V1_HARD_AUC = [0.4609375, 0.6434289383561643, 0.7770119863013698,
                0.9065710616438357, 0.6940924657534246]
V1_X_AUC = [0.9652216758381141, 0.9794154665730008, 0.9863964992389649,
             0.9935714494789837, 0.9815924657534247]
# Transaction localization (v1 clean): from clean_results.json content_aware_fusion
V1_HIT1 = 0.8316831683168316
V1_MRR  = 0.8804737253923169

# v2 best (2 seeds quick run — will be updated with 5-seed GPU run)
# These are the actual numbers from our run
V2_ID_F1 = [0.7402, 0.7459]  # per seed from quick run
V2_HARD_AUC = [0.7687, 0.8889]  # estimated from quick run (re-evaluated per seed)

# 2-head old (LambdaMART-based, from paper Table II/III)
# These are reference values from the paper — marked as REFERENCE ONLY
OLD_2HEAD_HIT1 = 0.832
OLD_2HEAD_MRR  = 0.880
OLD_2HEAD_ID_F1 = 0.744  # approximately same as v1


def paired_bootstrap_test(scores_a, scores_b, n_boot=10000, seed=42):
    """
    Paired bootstrap difference test.
    H0: mean(A) - mean(B) <= 0
    Returns: mean_diff, p_value (one-sided: A > B), 95% CI of diff
    """
    rng = np.random.RandomState(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    obs_diff = a.mean() - b.mean()
    
    # Bootstrap
    diffs = []
    n = len(a)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs = np.array(diffs)
    
    # p-value: fraction of bootstrap diffs <= 0 (one-sided test A > B)
    p_val = float((diffs <= 0).mean())
    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))
    
    return {
        "mean_diff": float(obs_diff),
        "p_value_A_gt_B": p_val,
        "ci_95": [ci_lo, ci_hi],
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(a.mean()),
        "std_a": float(a.std()),
        "mean_b": float(b.mean()),
        "std_b": float(b.std()),
    }


def paired_ttest(scores_a, scores_b):
    """Paired t-test (two-sided)."""
    a = np.array(scores_a)
    b = np.array(scores_b)
    if len(a) != len(b):
        # Use independent t-test if different lengths
        t_stat, p_val = stats.ttest_ind(a, b)
    else:
        t_stat, p_val = stats.ttest_rel(a, b)
    return {
        "t_statistic": float(t_stat),
        "p_value_two_sided": float(p_val),
        "mean_diff": float(a.mean() - b.mean()),
        "n_a": len(a), "n_b": len(b)
    }


def main():
    print("=" * 60)
    print("Statistical Tests: UnifiedTMIL v2 vs v1 vs 2-head old")
    print("=" * 60)
    
    results = {}
    
    # ─── ID-F1: v2 vs v1 ───
    print("\n[1] ID-F1: v2 vs v1")
    r = paired_bootstrap_test(V2_ID_F1, V1_ID_F1[:len(V2_ID_F1)])
    t = paired_ttest(V2_ID_F1, V1_ID_F1[:len(V2_ID_F1)])
    results["id_f1_v2_vs_v1"] = {"bootstrap": r, "ttest": t}
    print(f"  v2: {np.mean(V2_ID_F1):.4f}±{np.std(V2_ID_F1):.4f}")
    print(f"  v1: {np.mean(V1_ID_F1):.4f}±{np.std(V1_ID_F1):.4f}")
    print(f"  diff={r['mean_diff']:+.4f}, p(v2>v1)={1-r['p_value_A_gt_B']:.4f}, CI={r['ci_95']}")
    
    # ─── Hard-AUC: v2 vs v1 ───
    print("\n[2] Hard-AUC: v2 vs v1")
    r = paired_bootstrap_test(V2_HARD_AUC, V1_HARD_AUC[:len(V2_HARD_AUC)])
    t = paired_ttest(V2_HARD_AUC, V1_HARD_AUC[:len(V2_HARD_AUC)])
    results["hard_auc_v2_vs_v1"] = {"bootstrap": r, "ttest": t}
    print(f"  v2: {np.mean(V2_HARD_AUC):.4f}±{np.std(V2_HARD_AUC):.4f}")
    print(f"  v1: {np.mean(V1_HARD_AUC):.4f}±{np.std(V1_HARD_AUC):.4f}")
    print(f"  diff={r['mean_diff']:+.4f}, p(v2>v1)={1-r['p_value_A_gt_B']:.4f}, CI={r['ci_95']}")
    
    # ─── X-AUC: v2 vs v1 ───
    print("\n[3] X-AUC: v2 vs v1")
    # v2 X-AUC is 0.7369 (quick run) — this is lower than v1 due to hard mining shift
    # Note: this will be re-evaluated with full 5-seed GPU run
    v2_x_auc = [0.7369, 0.7369]  # placeholder from quick run (single ensemble value)
    r_xauc = {
        "mean_v2": 0.7369,
        "mean_v1": float(np.mean(V1_X_AUC)),
        "note": "PENDING full 5-seed run — quick run X-AUC may be underestimated due to hard mining distribution shift"
    }
    results["x_auc_v2_vs_v1"] = r_xauc
    print(f"  v2: 0.7369 (quick run, PENDING full run)")
    print(f"  v1: {np.mean(V1_X_AUC):.4f}±{np.std(V1_X_AUC):.4f}")
    print(f"  NOTE: X-AUC drop likely due to hard mining — needs investigation with full run")
    
    # ─── Localization: v2 vs v1 vs 2-head old ───
    print("\n[4] Localization Hit@1/MRR comparison")
    # v2 attention-only localization (quick run): 0.1062 / 0.2666
    # v1 clean (GBM fusion): 0.8317 / 0.8805
    # 2-head old (LambdaMART): 0.832 / 0.880
    loc_comparison = {
        "v2_attn_only": {"hit@1": 0.1062, "mrr": 0.2666,
                          "note": "attention-only, no GBM fusion — needs localization_ranker.py"},
        "v1_gbm_fusion": {"hit@1": V1_HIT1, "mrr": V1_MRR,
                           "note": "clean GBM fusion from clean_results.json"},
        "old_2head_lambdamart": {"hit@1": OLD_2HEAD_HIT1, "mrr": OLD_2HEAD_MRR,
                                  "note": "REFERENCE from paper Table III — not re-run"},
        "gap_v1_vs_old": {
            "hit@1_diff": V1_HIT1 - OLD_2HEAD_HIT1,
            "mrr_diff": V1_MRR - OLD_2HEAD_MRR,
            "interpretation": "v1 GBM fusion is within 0.001 of LambdaMART — effectively equivalent"
        }
    }
    results["localization"] = loc_comparison
    print(f"  v2 attn-only: Hit@1={0.1062:.4f} MRR={0.2666:.4f}")
    print(f"  v1 GBM fusion: Hit@1={V1_HIT1:.4f} MRR={V1_MRR:.4f}")
    print(f"  2-head LambdaMART: Hit@1={OLD_2HEAD_HIT1:.4f} MRR={OLD_2HEAD_MRR:.4f}")
    print(f"  Gap v1 vs LambdaMART: Hit@1={V1_HIT1-OLD_2HEAD_HIT1:+.4f} MRR={V1_MRR-OLD_2HEAD_MRR:+.4f}")
    
    # ─── Summary ───
    results["summary"] = {
        "hard_auc_improvement": f"+{np.mean(V2_HARD_AUC) - np.mean(V1_HARD_AUC[:len(V2_HARD_AUC)]):.4f}",
        "id_f1_change": f"{np.mean(V2_ID_F1) - np.mean(V1_ID_F1[:len(V2_ID_F1)]):+.4f}",
        "x_auc_note": "PENDING full 5-seed GPU run",
        "localization_note": "GBM fusion needed for fair comparison — attention-only not comparable",
        "status": "PARTIAL — quick run (2 seeds). Full 5-seed GPU run required for final numbers."
    }
    
    # Save results
    out_path = os.path.join(RES, "statistical_tests.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    
    # Generate LaTeX table
    tex = r"""\begin{table}[t]
\centering
\caption{Statistical Significance Tests: UnifiedTMIL v2 vs v1 (Bootstrap Difference Test)}
\label{tab:stat_tests}
\begin{tabular}{lcccc}
\hline
Metric & v2 Mean & v1 Mean & $\Delta$ & $p$-value \\
\hline
ID-F1 & """ + f"{np.mean(V2_ID_F1):.4f}" + r""" & """ + f"{np.mean(V1_ID_F1[:len(V2_ID_F1)]):.4f}" + r""" & """ + \
    f"{np.mean(V2_ID_F1)-np.mean(V1_ID_F1[:len(V2_ID_F1)]):+.4f}" + r""" & PENDING \\
Hard-AUC & """ + f"{np.mean(V2_HARD_AUC):.4f}" + r""" & """ + f"{np.mean(V1_HARD_AUC[:len(V2_HARD_AUC)]):.4f}" + r""" & """ + \
    f"{np.mean(V2_HARD_AUC)-np.mean(V1_HARD_AUC[:len(V2_HARD_AUC)]):+.4f}" + r""" & PENDING \\
X-AUC & PENDING & """ + f"{np.mean(V1_X_AUC):.4f}" + r""" & PENDING & PENDING \\
\hline
\multicolumn{5}{l}{\small Note: PENDING = requires full 5-seed GPU run. Current: 2-seed quick run.} \\
\end{tabular}
\end{table}
"""
    tex_path = os.path.join(TABLES, "statistical_test_table.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"LaTeX table: {tex_path}")


if __name__ == "__main__":
    main()
