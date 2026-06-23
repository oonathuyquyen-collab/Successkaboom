#!/usr/bin/env python3
"""
reproduce_all.py
================
One-stop script to reproduce all UnifiedTMIL results from scratch.

Usage:
    python3 scripts/reproduce_all.py [--quick] [--skip-build]

Options:
    --quick       Run with 1 seed instead of 5 (for CI/testing)
    --skip-build  Skip building the LaTeX PDF

Steps:
    1. Verify data files exist
    2. Train UnifiedTMIL (account-level, 5 seeds)
    3. Run enhanced transaction localization (LambdaMART, 5 seeds)
    4. Aggregate all results
    5. Build LaTeX PDF (optional)
    6. Print final SOTA comparison table

Requirements:
    pip install lightgbm xgboost scikit-learn scipy torch
"""

import os
import sys
import json
import argparse
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
PAPER = os.path.join(ROOT, "paper")
UTMIL = os.path.join(ROOT, "unified_tmil")


def run(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    start = time.time()
    ret = subprocess.run(cmd, shell=True, cwd=ROOT)
    elapsed = time.time() - start
    if ret.returncode != 0:
        print(f"ERROR: Command failed with code {ret.returncode}")
        sys.exit(1)
    print(f"  Done in {elapsed:.1f}s")


def check_data():
    required = [
        "data/ptx_bags.pkl",
        "data/allb_bags.pkl",
        "data/bert4eth/vocab.pkl",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(ROOT, f))]
    if missing:
        print("ERROR: Missing required data files:")
        for f in missing:
            print(f"  {f}")
        print("\nPlease download the PTXPhish dataset first.")
        sys.exit(1)
    print("Data files verified OK")


def main():
    parser = argparse.ArgumentParser(description="Reproduce all UnifiedTMIL results")
    parser.add_argument("--quick", action="store_true", help="Quick run with 1 seed")
    parser.add_argument("--skip-build", action="store_true", help="Skip LaTeX PDF build")
    args = parser.parse_args()

    print("=" * 60)
    print("UnifiedTMIL Full Reproduction Pipeline")
    print("=" * 60)

    # Step 1: Check data
    print("\n[1/5] Checking data files...")
    check_data()

    # Step 2: Train account-level model
    print("\n[2/5] Training UnifiedTMIL (account-level, 5 seeds)...")
    run(
        f"python3 {os.path.join(UTMIL, 'train_unified_tmil.py')}",
        "Account-level: TMIL ensemble + stacking"
    )

    # Step 3: Transaction localization
    print("\n[3/5] Running enhanced transaction localization (LambdaMART, 5 seeds)...")
    run(
        f"python3 {os.path.join(UTMIL, 'enhanced_localization.py')}",
        "Transaction-level: LambdaMART LOO"
    )

    # Step 4: Aggregate results
    print("\n[4/5] Aggregating all results...")
    run(
        f"python3 {os.path.join(UTMIL, 'aggregate_results.py')}",
        "Results aggregation"
    )

    # Step 5: Build PDF
    if not args.skip_build:
        print("\n[5/5] Building LaTeX PDF...")
        ieetran_cls = os.path.join(PAPER, "IEEEtran.cls")
        if not os.path.exists(ieetran_cls):
            print("  Downloading IEEEtran.cls...")
            subprocess.run(
                "wget -q https://mirrors.ctan.org/macros/latex/contrib/IEEEtran/IEEEtran.cls",
                shell=True, cwd=PAPER
            )
        run(
            "pdflatex -interaction=nonstopmode paper_en.tex && "
            "pdflatex -interaction=nonstopmode paper_en.tex",
            "LaTeX PDF build"
        )
    else:
        print("\n[5/5] Skipping PDF build (--skip-build)")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    res_path = os.path.join(RES, "comprehensive_results.json")
    if os.path.exists(res_path):
        R = json.load(open(res_path))
        print(f"\nScenario: {R.get('scenario', 'N/A')}")
        print("\nAccount-level (UnifiedTMIL Ensemble):")
        acc = R["account"]["UnifiedTMIL_ensemble"]
        print(f"  ID-F1:    {acc['in_domain_f1']:.4f}")
        print(f"  Hard-AUC: {acc['hard_auc']:.4f}")
        print(f"  X-AUC:    {acc['x_auc']:.4f}")
        print("\nTransaction-level (LambdaMART):")
        tx = R["transaction"]["UnifiedTMIL_LambdaMART"]
        print(f"  Hit@1:  {tx['hit@1']:.4f} +/- {tx['hit@1_std']:.4f}")
        print(f"  Hit@5:  {tx['hit@5']:.4f} +/- {tx['hit@5_std']:.4f}")
        print(f"  Hit@10: {tx['hit@10']:.4f} +/- {tx['hit@10_std']:.4f}")
        print(f"  MRR:    {tx['mrr']:.4f} +/- {tx['mrr_std']:.4f}")
        print("\nSOTA Comparison:")
        for k, v in R["sota_comparison"].items():
            status = "BEAT" if v["beats_sota"] else f"Gap={v['gap']:.4f}"
            print(f"  {v['metric']:10s}: {v['UnifiedTMIL']:.4f} vs {v['SOTA_name']} {v['SOTA_value']:.4f} -> {status}")
        all_beat = R.get("all_metrics_beat_sota", False)
        print(f"\nAll metrics beat SOTA: {'YES' if all_beat else 'NO'}")

    pdf_path = os.path.join(PAPER, "paper_en.pdf")
    if os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) // 1024
        print(f"\nPaper PDF: paper/paper_en.pdf ({size_kb} KB)")

    print("\nReproduction complete!")


if __name__ == "__main__":
    # If called with --clean-only, just produce clean_results.json without retraining
    if "--clean-only" in sys.argv:
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "produce_clean_results.py")],
            cwd=ROOT
        )
        sys.exit(result.returncode)
    main()
