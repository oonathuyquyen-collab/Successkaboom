#!/usr/bin/env python3
"""
generate_full_metric_tables.py
===============================
Generate complete metric tables for paper:
  - Account-level: Precision, Recall, F1, AUC-ROC, AUC-PR, Hard-AUC (in-domain + cross-domain)
  - Transaction-level: Hit@1, Hit@5, Hit@10, MRR (with CI)

Per Bước 7 of the SOTA improvement plan.

Usage:
    python3 scripts/generate_full_metric_tables.py

Outputs:
    results/tables/table2_account_full.tex
    results/tables/table3_localization_full.tex
    results/full_metric_tables.json
"""

import os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, "results")
TABLES = os.path.join(RES, "tables")
os.makedirs(TABLES, exist_ok=True)

# ─────────────────────────────────────────────
# Data: from clean_results.json + paper Table II/III
# Numbers marked [LOG] are from verified log files
# Numbers marked [PAPER] are from paper_final.pdf (reference only, not re-run)
# Numbers marked [PENDING] need GPU re-run
# ─────────────────────────────────────────────

# Load clean results
clean_path = os.path.join(RES, "clean_results.json")
with open(clean_path) as f:
    clean = json.load(f)

# Load v2 quick results
v2_path = os.path.join(RES, "unifiedtmil_v2_sota_results.json")
with open(v2_path) as f:
    v2_data = json.load(f)

v2_best = v2_data["configs"]["v2_best"]

# ─────────────────────────────────────────────
# Account-level baselines (from paper Table II — REFERENCE)
# ─────────────────────────────────────────────

BASELINES_ACCOUNT = {
    "BERT4ETH": {
        "id_f1": 0.712, "id_prec": None, "id_rec": None,
        "id_auc": 0.903, "id_aupr": None,
        "hard_auc": 0.721, "x_auc": 0.961,
        "source": "PAPER"
    },
    "ZipZap": {
        "id_f1": 0.698, "id_prec": None, "id_rec": None,
        "id_auc": 0.891, "id_aupr": None,
        "hard_auc": 0.734, "x_auc": 0.943,
        "source": "PAPER"
    },
    "TSGN": {
        "id_f1": 0.721, "id_prec": None, "id_rec": None,
        "id_auc": 0.908, "id_aupr": None,
        "hard_auc": 0.748, "x_auc": 0.952,
        "source": "PAPER"
    },
    "LMAE4Eth": {
        "id_f1": 0.750, "id_prec": None, "id_rec": None,
        "id_auc": 0.921, "id_aupr": None,
        "hard_auc": 0.708, "x_auc": 0.967,
        "source": "PAPER"
    },
    "Max-pool MIL": {
        "id_f1": 0.752, "id_prec": None, "id_rec": None,
        "id_auc": 0.919, "id_aupr": None,
        "hard_auc": 0.801, "x_auc": 0.971,
        "source": "PAPER"
    },
    "Gated-attn MIL": {
        "id_f1": 0.741, "id_prec": None, "id_rec": None,
        "id_auc": 0.917, "id_aupr": None,
        "hard_auc": 0.823, "x_auc": 0.968,
        "source": "PAPER"
    },
    "Mean-pool MIL": {
        "id_f1": 0.755, "id_prec": None, "id_rec": None,
        "id_auc": 0.922, "id_aupr": None,
        "hard_auc": 0.885, "x_auc": 0.974,
        "source": "PAPER"
    },
}

# UnifiedTMIL v1 (from clean_results.json — LOG)
UTMIL_V1 = {
    "id_f1": clean["account_level"]["id_f1"]["mean"],
    "id_f1_std": clean["account_level"]["id_f1"]["std"],
    "id_prec": None,  # not in clean_results — PENDING re-run
    "id_rec": None,   # not in clean_results — PENDING re-run
    "id_auc": None,   # not in clean_results (only cross-domain)
    "id_aupr": None,  # PENDING
    "hard_auc": clean["account_level"]["hard_auc"]["mean"],
    "hard_auc_std": clean["account_level"]["hard_auc"]["std"],
    "x_auc": clean["account_level"]["x_auc"]["mean"],
    "x_auc_std": clean["account_level"]["x_auc"]["std"],
    "source": "LOG:results/clean_results.json"
}

# UnifiedTMIL v2 (from quick run — 2 seeds, PENDING full 5-seed)
v2_indom = v2_best["in_domain"]
UTMIL_V2 = {
    "id_f1": v2_indom["f1"]["mean"],
    "id_f1_std": v2_indom["f1"]["std"],
    "id_prec": v2_indom["precision"]["mean"],
    "id_prec_std": v2_indom["precision"]["std"],
    "id_rec": v2_indom["recall"]["mean"],
    "id_rec_std": v2_indom["recall"]["std"],
    "id_auc": v2_indom["auc"]["mean"],
    "id_auc_std": v2_indom["auc"]["std"],
    "id_aupr": v2_indom["aupr"]["mean"],
    "id_aupr_std": v2_indom["aupr"]["std"],
    "hard_auc": v2_best["hard_auc"]["mean"],
    "hard_auc_std": v2_best["hard_auc"].get("std", 0),
    "x_auc": v2_best["cross_domain"]["auc"],
    "source": "LOG:results/unifiedtmil_v2_sota_results.json (2-seed quick run, PENDING full 5-seed)"
}

# ─────────────────────────────────────────────
# Transaction-level baselines
# ─────────────────────────────────────────────

BASELINES_TX = {
    "Degree-rank": {"hit1": 0.673, "hit5": 0.891, "hit10": 0.921, "mrr": 0.762, "source": "PAPER"},
    "Amount-rank": {"hit1": 0.693, "hit5": 0.901, "hit10": 0.931, "mrr": 0.779, "source": "PAPER"},
    "TransMIL": {"hit1": 0.762, "hit5": 0.911, "hit10": 0.941, "mrr": 0.831, "source": "PAPER"},
    "CLAM": {"hit1": 0.772, "hit5": 0.921, "hit10": 0.941, "mrr": 0.839, "source": "PAPER"},
    "LambdaMART (old)": {"hit1": 0.832, "hit5": 0.931, "hit10": 0.941, "mrr": 0.880, "source": "PAPER"},
}

# v1 localization (from clean_results.json — LOG)
TX_V1 = {
    "hit1": clean["transaction_level"]["content_aware_fusion"]["hit@1"],
    "hit1_ci": clean["transaction_level"]["content_aware_fusion"]["hit@1_ci_95"],
    "hit5": clean["transaction_level"]["content_aware_fusion"]["hit@5"],
    "hit10": clean["transaction_level"]["content_aware_fusion"]["hit@10"],
    "mrr": clean["transaction_level"]["content_aware_fusion"]["mrr"],
    "mrr_ci": clean["transaction_level"]["content_aware_fusion"]["mrr_ci_95"],
    "n": clean["transaction_level"]["content_aware_fusion"]["n"],
    "source": "LOG:results/clean_results.json"
}

# v2 localization (attention-only from quick run — GBM fusion PENDING)
TX_V2 = {
    "hit1": v2_best["loc_attention"]["hit@1"],
    "hit5": v2_best["loc_attention"].get("hit@5", None),
    "hit10": v2_best["loc_attention"].get("hit@10", None),
    "mrr": v2_best["loc_attention"]["mrr"],
    "n": v2_best["loc_attention"]["n"],
    "source": "LOG:results/unifiedtmil_v2_sota_results.json (attention-only, GBM fusion PENDING)",
    "note": "PENDING — GBM fusion with v2 attention scores needed for fair comparison"
}


def fmt(val, std=None, pending=False):
    if pending or val is None:
        return "PENDING"
    if std is not None:
        return f"{val:.3f}$\\pm${std:.3f}"
    return f"{val:.3f}"


def main():
    print("=" * 60)
    print("Generating Full Metric Tables")
    print("=" * 60)

    # ─── Table 2: Account-level ───
    print("\n[1] Account-level Table")
    
    tex2 = r"""\begin{table*}[t]
\centering
\caption{Account-Level Detection Results. Best results in \textbf{bold}. 
[LOG] = verified from training log. [PAPER] = from paper\_final.pdf (reference). 
[PENDING] = requires full 5-seed GPU run.}
\label{tab:account_results}
\begin{tabular}{lcccccc}
\hline
Method & ID-F1 & ID-Prec & ID-Rec & ID-AUC-PR & Hard-AUC & X-AUC \\
\hline
"""
    
    for name, b in BASELINES_ACCOUNT.items():
        f1 = fmt(b["id_f1"])
        pr = fmt(b["id_prec"]) if b["id_prec"] else "---"
        rc = fmt(b["id_rec"]) if b["id_rec"] else "---"
        aupr = fmt(b["id_aupr"]) if b["id_aupr"] else "---"
        hauc = fmt(b["hard_auc"])
        xauc = fmt(b["x_auc"])
        tex2 += f"{name} & {f1} & {pr} & {rc} & {aupr} & {hauc} & {xauc} \\\\\n"
    
    tex2 += r"\hline" + "\n"
    
    # v1
    f1_v1 = fmt(UTMIL_V1["id_f1"], UTMIL_V1["id_f1_std"])
    hauc_v1 = fmt(UTMIL_V1["hard_auc"], UTMIL_V1["hard_auc_std"])
    xauc_v1 = fmt(UTMIL_V1["x_auc"], UTMIL_V1["x_auc_std"])
    tex2 += f"UnifiedTMIL v1 [LOG] & {f1_v1} & PENDING & PENDING & PENDING & {hauc_v1} & {xauc_v1} \\\\\n"
    
    # v2
    f1_v2 = fmt(UTMIL_V2["id_f1"], UTMIL_V2["id_f1_std"])
    pr_v2 = fmt(UTMIL_V2["id_prec"], UTMIL_V2.get("id_prec_std"))
    rc_v2 = fmt(UTMIL_V2["id_rec"], UTMIL_V2.get("id_rec_std"))
    aupr_v2 = fmt(UTMIL_V2["id_aupr"], UTMIL_V2.get("id_aupr_std"))
    hauc_v2 = fmt(UTMIL_V2["hard_auc"], UTMIL_V2.get("hard_auc_std"))
    xauc_v2 = "PENDING (full run)"
    tex2 += f"\\textbf{{UnifiedTMIL v2}} [LOG*] & \\textbf{{{f1_v2}}} & {pr_v2} & {rc_v2} & {aupr_v2} & \\textbf{{{hauc_v2}}} & {xauc_v2} \\\\\n"
    
    tex2 += r"""\hline
\multicolumn{7}{l}{\small [LOG*] = 2-seed quick run. Full 5-seed GPU run PENDING.} \\
\end{tabular}
\end{table*}
"""
    
    tex2_path = os.path.join(TABLES, "table2_account_full.tex")
    with open(tex2_path, "w") as f:
        f.write(tex2)
    print(f"  Saved: {tex2_path}")
    
    # ─── Table 3: Transaction-level ───
    print("\n[2] Transaction-level Table")
    
    tex3 = r"""\begin{table}[t]
\centering
\caption{Transaction-Level Localization Results (LOO protocol, $n=101$ bags).
[LOG] = verified from training log. [PAPER] = reference only.}
\label{tab:tx_results}
\begin{tabular}{lcccc}
\hline
Method & Hit@1 & Hit@5 & Hit@10 & MRR \\
\hline
"""
    
    for name, b in BASELINES_TX.items():
        tex3 += f"{name} & {b['hit1']:.3f} & {b['hit5']:.3f} & {b['hit10']:.3f} & {b['mrr']:.3f} \\\\\n"
    
    tex3 += r"\hline" + "\n"
    
    # v1
    h1_v1 = f"{TX_V1['hit1']:.3f} [{TX_V1['hit1_ci'][0]:.3f},{TX_V1['hit1_ci'][1]:.3f}]"
    mrr_v1 = f"{TX_V1['mrr']:.3f} [{TX_V1['mrr_ci'][0]:.3f},{TX_V1['mrr_ci'][1]:.3f}]"
    tex3 += f"UnifiedTMIL v1 [LOG] & {h1_v1} & {TX_V1['hit5']:.3f} & {TX_V1['hit10']:.3f} & {mrr_v1} \\\\\n"
    
    # v2
    tex3 += f"UnifiedTMIL v2 [PENDING] & PENDING & PENDING & PENDING & PENDING \\\\\n"
    
    tex3 += r"""\hline
\multicolumn{5}{l}{\small v2 localization requires GBM fusion re-run with v2 attention scores.} \\
\end{tabular}
\end{table}
"""
    
    tex3_path = os.path.join(TABLES, "table3_localization_full.tex")
    with open(tex3_path, "w") as f:
        f.write(tex3)
    print(f"  Saved: {tex3_path}")
    
    # ─── Save JSON ───
    full_data = {
        "account_level": {
            "baselines": BASELINES_ACCOUNT,
            "unifiedtmil_v1": UTMIL_V1,
            "unifiedtmil_v2": UTMIL_V2,
        },
        "transaction_level": {
            "baselines": BASELINES_TX,
            "unifiedtmil_v1": TX_V1,
            "unifiedtmil_v2": TX_V2,
        }
    }
    json_path = os.path.join(RES, "full_metric_tables.json")
    with open(json_path, "w") as f:
        json.dump(full_data, f, indent=2)
    print(f"  Saved JSON: {json_path}")
    
    # ─── Print summary ───
    print("\n" + "=" * 60)
    print("METRIC SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'v1 (LOG)':>15} {'v2 (SOTA)':>15} {'Best Baseline':>15}")
    print("-" * 65)
    print(f"{'ID-F1':<20} {UTMIL_V1['id_f1']:.4f}±{UTMIL_V1['id_f1_std']:.4f} {UTMIL_V2['id_f1']:.4f}±{UTMIL_V2['id_f1_std']:.4f} {'Mean-pool 0.755':>15}")
    print(f"{'Hard-AUC':<20} {UTMIL_V1['hard_auc']:.4f}±{UTMIL_V1['hard_auc_std']:.4f} {UTMIL_V2['hard_auc']:.4f}±{UTMIL_V2['hard_auc_std']:.4f} {'Mean-pool 0.885':>15}")
    print(f"{'X-AUC':<20} {UTMIL_V1['x_auc']:.4f}±{UTMIL_V1['x_auc_std']:.4f} {'PENDING':>15} {'LMAE4Eth 0.967':>15}")
    print(f"{'Loc Hit@1':<20} {TX_V1['hit1']:.4f} [LOG] {'PENDING':>15} {'LambdaMART 0.832':>15}")
    print(f"{'Loc MRR':<20} {TX_V1['mrr']:.4f} [LOG] {'PENDING':>15} {'LambdaMART 0.880':>15}")
    print("\n[LOG*] = 2-seed quick run. Full 5-seed GPU run PENDING for final paper numbers.")


if __name__ == "__main__":
    main()
