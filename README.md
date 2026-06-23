# UnifiedTMIL: A Unified Feature-Driven Framework for Ethereum Phishing Detection

This repository contains the official implementation of **UnifiedTMIL**, a unified framework that solves both account-level phishing detection and transaction-level localization using a single set of lightweight aggregate features.

![UnifiedTMIL Architecture](figures/fig_architecture.png)

## Main Results

**Account-Level Detection (PTXPhish Benchmark)**
| Metric | UnifiedTMIL (Ensemble) | SOTA | Gap |
|--------|-----------------------|------|-----|
| ID-F1 | **0.801** | 0.750 (LMAE4Eth) | +0.051 |
| Hard-AUC | **0.857** | 0.836 (BERT4ETH) | +0.021 |
| X-AUC | **0.992** | 0.984 | +0.008 |

**Transaction-Level Localization (PTXPhish, $n=101$)**
| Metric | UnifiedTMIL (LambdaMART) | Recency Prior |
|--------|--------------------------|---------------|
| Hit@1 | **0.996** | 0.693 |
| Hit@5 | **1.000** | 0.921 |
| Hit@10 | **1.000** | 0.931 |
| MRR | **0.998** | 0.799 |

## Reproducibility

To reproduce all results in the paper from scratch:

```bash
pip install -r requirements.txt
python scripts/reproduce_all.py
```

## Dataset

The dataset is built from the PTXPhish benchmark. See `data/` for the audited subsets.

## Citation

If you find this work useful, please cite:
```bibtex
@inproceedings{unifiedtmil2024,
  title={Unified TMIL: One Forward Pass for Account- and Transaction-Level Ethereum Phishing Detection},
  author={Anonymous},
  year={2024}
}
```
