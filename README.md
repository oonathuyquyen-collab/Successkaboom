# UnifiedTMIL: One Forward Pass for Account- and Transaction-Level Ethereum Phishing Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

This repository contains the **official implementation**, audited benchmark, and evaluation protocol for **UnifiedTMIL**, a unified framework for detecting Ethereum phishing scammers at the account level and pinpointing fraudulent transactions at the transaction level — all in a single forward pass.

---

## Highlights

**UnifiedTMIL** addresses two deeply coupled tasks simultaneously:

1. **Account-level detection:** Classify whether an Ethereum address is a phishing scammer based on its transaction history.
2. **Transaction-level localization:** Identify the specific fraudulent transaction(s) within a scammer's history, providing actionable forensic evidence.

Key contributions:

- **Streamlined End-to-End Architecture.** A single differentiable model with a BERT4ETH+TCN backbone and a gated attention head that replaces separate GBDT and LambdaMART pipelines.
- **Audited Benchmark.** Contract-mediated relabeling of PTXPhish raises clean account-level ground truth from 83.4% to 92.7%.
- **Honest Evaluation Protocol.** Cluster-aware bootstrap CIs, five-seed runs, leave-one-out cross-validation for the localization meta-learner, and rigorous leakage controls.

### Performance Summary

| Metric | Value | Notes |
|--------|-------|-------|
| ID-F1 | **0.744 ± 0.005** | 5-seed mean, in-domain test set |
| X-AUC | **0.981** | Cross-domain (phishing vs. normal EOA) |
| Hard-AUC | 0.696 ± 0.148 | Phishing vs. DeFi/KOL hard negatives |
| Hit@1 | **0.832** | LOO localization, interior-GT bags (n=101) |
| Hit@5 | 0.931 | |
| MRR | 0.880 | |

---

## Repository Structure

```
Successkaboom/
├── src/                        # Core model source code
│   ├── unified_model.py        # UnifiedTMIL architecture
│   ├── localization_ranker.py  # LOO GBM localization pipeline
│   └── vocab_def.py            # Vocabulary class (BERT4ETH-compatible)
├── unified_tmil/               # Training and pipeline scripts
│   ├── train_unified_tmil.py   # Full training pipeline (5 seeds)
│   ├── train_fast.py           # Fast training (ablation / CI)
│   └── finish_pipeline.py      # Post-training evaluation pipeline
├── scripts/                    # Reproduction and audit scripts
│   ├── reproduce_all.py        # One-stop reproduction script
│   ├── audit_leakage.py        # Data leakage audit
│   ├── statistical_test.py     # Paired bootstrap and t-tests
│   ├── run_ablation.py         # Ablation study runner
│   └── ...
├── paper/                      # LaTeX source and compiled PDF
│   ├── paper_en.tex            # Main LaTeX source (IEEE format)
│   ├── paper_en.pdf            # Compiled paper
│   └── references.bib          # BibTeX references (30 entries)
├── results/                    # Pre-computed results
│   ├── clean_results.json      # Authoritative headline numbers
│   ├── comprehensive_results.json  # Full per-seed breakdown
│   ├── tables/                 # LaTeX table fragments
│   └── logs/                   # Per-run experiment logs
├── figures/                    # High-resolution paper figures
├── data/                       # Audited PTXPhish dataset
│   ├── bert4eth/               # Pre-processed BERT4ETH bags
│   └── PTXPhish_source/        # Raw PTXPhish source files
├── docs/                       # Audit reports and documentation
│   ├── leakage_audit.md        # Data leakage audit report
│   ├── NOVELTY_STATEMENT.md    # Novelty and contribution statement
│   └── ...
├── unifiedTMILpaper.pdf        # Final compiled paper (convenience copy)
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/oonathuyquyen-collab/Successkaboom.git
cd Successkaboom
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, scikit-learn 1.3+, LightGBM 4.0+.

---

## Data

The audited PTXPhish dataset is included in `data/`. Pre-processed BERT4ETH bags (`train_bags.pkl`, `test_bags.pkl`, `vocab.pkl`) are in `data/bert4eth/`. Raw source files are in `data/PTXPhish_source/`.

---

## Reproduction

### Quick verification (pre-computed results, no GPU needed)

```bash
python scripts/reproduce_all.py --clean-only
```

### Full reproduction (requires GPU, ~2–4 hours)

```bash
python scripts/reproduce_all.py
```

### Ablation study

```bash
python scripts/run_ablation.py
```

### Statistical significance tests

```bash
python scripts/statistical_test.py
```

---

## Leakage Audit

```bash
python scripts/audit_leakage.py
```

The most critical finding: global counterparty reputation (without LOO) yields an artificial Hit@1 = 1.000. The LOO protocol reduces this to a realistic 0.832. See [`docs/leakage_audit.md`](docs/leakage_audit.md) for the full report.

---

## Citation

```bibtex
@article{UnifiedTMIL2026,
  title     = {UnifiedTMIL: One Forward Pass for Account- and Transaction-Level
               Ethereum Phishing Detection},
  author    = {Anonymous},
  journal   = {arXiv preprint},
  year      = {2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
