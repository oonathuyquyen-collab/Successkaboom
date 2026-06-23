# UnifiedTMIL: Unified Ethereum Phishing Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**UnifiedTMIL** is a unified attention-based Multiple Instance Learning (MIL) framework for Ethereum phishing detection, solving both account-level classification and transaction-level localization in a single forward pass.

## New State-of-the-Art Results (v2)

Results averaged over **5 random seeds** with 95% bootstrap confidence intervals.

### Account-Level Detection

| Model | ID-F1 | Cross-Domain AUC | Hard-Negative AUC |
|-------|-------|------------------|-------------------|
| BERT4ETH | 0.734 | 0.948 | 0.836 |
| ZipZap | 0.711 | 0.975 | 0.776 |
| TSGN | 0.727 | 0.962 | 0.760 |
| LMAE4Eth | 0.750 | 0.950 | 0.708 |
| **UnifiedTMIL v2** | **0.801** | **0.992** | **0.857** |

### Transaction-Level Localization (n=101, artifact removed)

| Ranker | Hit@1 | Hit@5 | Hit@10 | MRR |
|--------|-------|-------|--------|-----|
| Recency | 0.693 | 0.921 | 0.931 | 0.799 |
| GatedMIL | 0.243 | 0.466 | 0.610 | 0.362 |
| TransMIL | 0.223 | 0.370 | 0.483 | 0.302 |
| CLAM | 0.301 | 0.651 | 0.784 | 0.465 |
| UnifiedTMIL v1 | 0.832 | 0.931 | 0.941 | 0.880 |
| **UnifiedTMIL v2 (LambdaMART)** | **0.996** | **1.000** | **1.000** | **0.998** |

## Key Improvements (v2)

1. **Enhanced Feature Engineering**: 26 aggregate features including temporal burst patterns, fund-flow concentration, counterparty novelty, and zero-value outbound detection.
2. **Ensemble Meta-Learner**: LightGBM + XGBoost stacking on top of TMIL attention features.
3. **LambdaMART Ranking**: Learning-to-rank replaces classification for transaction localization.
4. **Multi-seed Evaluation**: 5 seeds with CI and Wilcoxon significance tests.

## Repository Structure

```
Successkaboom/
├── data/                          # PTXPhish dataset (see below)
│   ├── ptx_bags.pkl               # Phishing transaction bags
│   ├── allb_bags.pkl              # All bags (train + test)
│   └── bert4eth/vocab.pkl         # BERT4ETH vocabulary
├── unified_tmil/                  # New v2 pipeline
│   ├── train_unified_tmil.py      # Account-level training (5 seeds)
│   ├── enhanced_localization.py   # Transaction-level LambdaMART
│   ├── aggregate_results.py       # Results aggregation
│   └── enhanced_account_model.py  # Feature extraction utilities
├── src/                           # Original v1 pipeline
├── src_lean/                      # Lean baseline implementations
├── paper/                         # LaTeX source + PDF
│   ├── paper_en.tex               # IEEEtran LaTeX source
│   └── paper_en.pdf               # Compiled PDF
├── results/                       # All experimental results (JSON)
│   ├── comprehensive_results.json # Final aggregated results
│   ├── unified_tmil_multiseed.json
│   └── enhanced_localization_results.json
├── scripts/
│   └── reproduce_all.py           # Full reproduction script
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Reproduce All Results

```bash
python3 scripts/reproduce_all.py
```

This will:
- Train UnifiedTMIL over 5 seeds (account-level)
- Run LambdaMART LOO evaluation (transaction-level)
- Aggregate all results
- Build the LaTeX PDF

### 3. Quick Test (1 seed)

```bash
python3 scripts/reproduce_all.py --quick --skip-build
```

## Dataset

The PTXPhish dataset is derived from [BERT4ETH](https://github.com/git-disl/BERT4ETH). Please follow their data download instructions and place the processed `.pkl` files in the `data/` directory.

## Paper

The paper is available at `paper/paper_en.pdf`. It is formatted for IEEE conference submission.

## Citation

```bibtex
@inproceedings{unifiedtmil2024,
  title={Unified TMIL: One Forward Pass for Account- and Transaction-Level Ethereum Phishing Detection},
  author={Anonymous},
  booktitle={Proceedings of the IEEE},
  year={2024}
}
```

## License

MIT License. See [LICENSE](LICENSE).
