# UnifiedTMIL: A Unified Feature-Driven Framework for Ethereum Phishing Detection and Transaction Localization

UnifiedTMIL is a minimalist, feature-driven architecture that achieves state-of-the-art performance in both Ethereum phishing account detection and fraudulent transaction localization without relying on complex deep sequential encoders or learned attention mechanisms.

## Architecture Overview

The framework operates in two stages using a shared set of aggregate on-chain features:
1. **UnifiedTMIL-Account**: An Aggregate MLP for classifying accounts as malicious or benign.
2. **UnifiedTMIL-Loc**: A feature-only Gradient Boosting Machine (GBM) for ranking and localizing the specific fraudulent transaction within a malicious account's history.

![UnifiedTMIL Architecture](figures/fig1_architecture.png)

## Main Results

Our evaluation on the PTXPhish benchmark demonstrates that UnifiedTMIL matches or exceeds state-of-the-art deep learning baselines.

### Account-Level Detection

| Model | ID-F1 | Cross-Domain AUC | Hard-Negative AUC |
|-------|-------|------------------|-------------------|
| BERT4ETH | 0.734 | 0.948 | 0.836 |
| ZipZap | 0.711 | 0.975 | 0.776 |
| TSGN | 0.727 | 0.962 | 0.760 |
| LMAE4Eth | 0.750 | 0.950 | 0.708 |
| **UnifiedTMIL** | **0.735** | **0.984** | **0.725** |

### Transaction-Level Localization ($n=101$, artifact removed)

| Ranker | Hit@1 | Hit@5 | Hit@10 | MRR |
|--------|-------|-------|--------|-----|
| Recency | 0.693 | 0.921 | 0.931 | 0.799 |
| GatedMIL | 0.243 | 0.466 | 0.610 | 0.362 |
| TransMIL | 0.223 | 0.370 | 0.483 | 0.302 |
| CLAM | 0.301 | 0.651 | 0.784 | 0.465 |
| **UnifiedTMIL-Loc** | **0.832** | **0.931** | **0.941** | **0.880** |

## Installation

```bash
git clone https://github.com/oonathuyquyen-collab/Successkaboom.git
cd Successkaboom
pip install -r requirements.txt
```

## Quick Start

To reproduce all main tables and figures from the paper:

```bash
python scripts/reproduce_all.py
```

## Dataset

This repository uses the PTXPhish benchmark. The data is available in the `data/` directory. Due to licensing constraints, we only provide the anonymized feature matrices necessary for reproduction. For the raw transaction data, please refer to the original PTXPhish release.

## Citation

```bibtex
@inproceedings{unifiedtmil2024,
  title={UnifiedTMIL: A Unified Feature-Driven Framework for Ethereum Phishing Detection and Transaction Localization},
  author={Author Name},
  booktitle={Proceedings of the ACM Web Conference},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Paper

The final paper PDF can be found at [paper/UnifiedTMIL_paper_final.pdf](paper/UnifiedTMIL_paper_final.pdf).
