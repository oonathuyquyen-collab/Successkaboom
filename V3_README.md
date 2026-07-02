# UnifiedTMIL v3 — Single-Forward End-to-End Architecture

## Overview

UnifiedTMIL v3 solves two Ethereum phishing detection tasks in **one forward pass**:

1. **Account-level detection** — Is this account a phisher? (binary classification)
2. **Transaction-level localization** — Which transactions are fraudulent? (ranking)

## Key Architectural Changes from v1

| Component | v1 (paper) | v3 (this branch) |
|-----------|-----------|------------------|
| Head-C (account) | Neural + optional LGBM/XGB ensemble | **Pure MLP** (no ensemble) |
| Head-L (localization) | GBM LOO + LambdaMART + Feature Fusion | **Attention-as-Ranking** (end-to-end) |
| Positional features | prel, dist_nl, rank, z-score | **Removed entirely** |
| Engineered features | spike, novelty, cummax, runmax, avc, zero_out, ... | **Only LOO Reputation** (ablation-verified) |
| Contrastive loss weight | Hardcoded λc=0.3 | **Tuned** λc ∈ [0.0, 0.5] via grid search |
| Learnable temperature | No | **Yes** (log_tau) for attention sharpness |
| Entropy regularization | No | **Yes** λe ∈ [0.0, 0.25] for peaked attention |
| Optimizer | Adam | **AdamW** with CosineAnnealingLR |

## Architecture

```
Input: (cp_ids, io_dir, amounts, delta_ts)
    │
    ├── cp_embed (BERT4ETH vocab)  ──┐
    ├── io_embed (IN/OUT)          ──┤
    └── hc_proj (log-amount, log-Δt) ─┘
    │         ⊕
    ▼
  LayerNorm
    │
    ▼
  1D-TCN (kernel=3, residual)
    │
    ▼
  [⊕ LOO Reputation Feature]
    │
    ▼
  Gated Attention (Ilse et al.)
    ├── attention_weights → Localization Score (Hit@K)
    │
    └── weighted context vector
            │
            ▼
         MLP Classifier → Account Logit (F1/AUC)
```

## Loss Function

```
L = BCE(y, y_hat) + λc · ContrastiveLoss(z_pos, z_neg) + λe · EntropyReg(a)
```

- **BCE**: Binary cross-entropy for account classification
- **ContrastiveLoss**: Push phishing/benign context vectors apart (margin=1.0)
- **EntropyReg**: Encourage peaked attention for better localization

## New Files

| File | Purpose |
|------|---------|
| `scripts/verify_v3.py` | V3 model definition + training + evaluation (local CPU test) |
| `kaggle_unifiedtmil_v3.ipynb` | **Kaggle notebook** for full GPU training |

## Expected Results (GPU, 5 seeds, 10 epochs)

| Metric | v1 (GBM fusion) | v3 Target |
|--------|:---------------:|:---------:|
| ID-F1 | 0.744 ± 0.005 | **≥ 0.740** |
| X-AUC | 0.981 | **≥ 0.975** |
| Hard-AUC | 0.696 ± 0.148 | **≥ 0.750** |
| Hit@1 | 0.832 (GBM) | **≥ 0.650** (attention-only, end-to-end) |
| MRR | 0.880 | **≥ 0.750** |

> Hit@1 drop from 0.832 to ~0.65-0.75 is expected and architecturally justified. The paper claims end-to-end unification, not absolute localization SOTA.

## How to Run on Kaggle

1. Upload data to Kaggle:
   ```bash
   # From repo root:
   cd /workspace/Successkaboom
   zip -r /tmp/successkaboom_data.zip data/
   # Upload to Kaggle as a dataset
   ```

2. Open `kaggle_unifiedtmil_v3.ipynb` in Kaggle
3. Set `QUICK = False` for full experiment
4. Connect GPU accelerator (T4/P100)
5. Run all cells (~30-60 min for full 5 seeds × 10 epochs)
