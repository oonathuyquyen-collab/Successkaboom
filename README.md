# UnifiedTMIL: One Forward Pass for Account- and Transaction-Level Ethereum Phishing Detection

This repository contains the official implementation, audited benchmark, and evaluation protocol for **UnifiedTMIL**, a unified framework for detecting Ethereum phishing scammers (account-level) and pinpointing fraudulent transactions (transaction-level) in a single forward pass.

## 🚀 Highlights

- **Unified Architecture:** A single-weight model that jointly solves account classification and transaction localization.
- **Audited Benchmark:** A clean version of the PTXPhish dataset with contract-mediated relabeling, tracing 92.7% of phishing receipts.
- **Honest Evaluation:** Rigorous evaluation protocol including cluster-aware bootstrap CIs, leakage control, and by-source negative breakdowns.
- **Performance:** Achieves **0.744 ID-F1**, **0.981 X-AUC**, and **0.832 Hit@1** localization performance under a leakage-audited protocol.

## 📂 Repository Structure

- `src/core/`: Core model architecture and ranker implementation.
- `paper/`: LaTeX source and the final PDF paper.
- `results/`: Cleaned, audited evaluation results and leakage audit data.
- `scripts/`: Reproduction and audit scripts.
- `data/`: PTXPhish-derived audited dataset.
- `figures/`: High-resolution figures used in the paper.
- `docs/`: Audit reports and technical documentation.

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 📊 Reproduction

To reproduce the clean results and tables reported in the paper:

```bash
python scripts/reproduce_all.py --clean-only
```

This will verify the pre-computed metrics and generate `results/clean_results.json` matching the paper's headline numbers.

## 🔍 Leakage Audit

We provide a systematic audit of data leakage in phishing localization:

```bash
python scripts/audit_leakage.py
```

Detailed findings are documented in [`docs/leakage_audit.md`](docs/leakage_audit.md).

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{Successkaboom2026UnifiedTMIL,
  title={UnifiedTMIL: One Forward Pass for Account- and Transaction-Level Ethereum Phishing Detection},
  author={Anonymous},
  journal={arXiv preprint},
  year={2026}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
