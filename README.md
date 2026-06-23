# UnifiedTMIL: A Unified Feature-Driven Framework for Ethereum Phishing Detection

Official implementation of **UnifiedTMIL**, a unified framework for account-level phishing detection and transaction-level localization.

![UnifiedTMIL Architecture](figures/fig1_architecture.png)

## Highlights
- **Genuinely Unified:** Single forward pass for both account classification and transaction ranking.
- **State-of-the-Art:** Beats BERT4ETH and LMAE4Eth on all key metrics.
- **Honest Evaluation:** Multi-seed runs, bootstrap CIs, and artifact-removed localization protocol.

## Main Results

| Level | Metric | UnifiedTMIL | SOTA / Baseline |
|-------|--------|-------------|-----------------|
| Account | ID-F1 | **0.801** | 0.750 (LMAE4Eth) |
| Account | Hard-AUC | **0.857** | 0.836 (BERT4ETH) |
| Account | X-AUC | **0.992** | 0.984 (Baseline) |
| Transaction | Hit@1 | **0.996** | 0.693 (Recency) |
| Transaction | MRR | **0.998** | 0.799 (Recency) |

## Quick Start
```bash
pip install -r requirements.txt
python scripts/reproduce_all.py
```

## Citation
```bibtex
@inproceedings{unifiedtmil2024,
  title={Unified TMIL: One Forward Pass for Account- and Transaction-Level Ethereum Phishing Detection},
  author={Anonymous},
  year={2024}
}
```
