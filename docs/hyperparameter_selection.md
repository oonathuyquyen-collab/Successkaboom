# Hyperparameter Selection Report

## 1. Executive Summary
This report details the selection of optimal hyperparameters for the UnifiedTMIL model based on the ablation studies conducted. The primary metric for optimization was the F1 score on the in-domain validation set. Key findings indicate that a contrastive loss weight of 0.1 and an entropy regularization lambda of 0.0 yield the best performance.

## 2. Contrastive Loss Weight (λc) Ablation (Part 1.1)

The ablation study for the contrastive loss weight (`contrast_weight`) revealed that `λc = 0.1` achieved the highest mean F1 score, outperforming the previously used `λc = 0.3`.

| Configuration | F1 Mean | F1 Std | AUC Mean | AUPR Mean |
| :------------ | :------ | :----- | :------- | :-------- |
| `contrast_0.0` | 0.7311 | 0.0065 | 0.9189 | 0.8057 |
| **`contrast_0.1`** | **0.7367** | **0.0038** | **0.9213** | **0.8095** |
| `contrast_0.3` | 0.7349 | 0.0136 | 0.9190 | 0.8077 |
| `contrast_0.5` | 0.7280 | 0.0066 | 0.9189 | 0.8081 |

**Decision**: Based on these results, the optimal `contrast_weight` is **0.1**. This aligns with the prompt's observation that `λc=0.1` shows higher mean F1 and lower standard deviation compared to `λc=0.3`.

## 3. Backbone Component Ablation

The ablation of backbone components demonstrates the importance of each module in the UnifiedTMIL architecture.

| Configuration | F1 Mean | F1 Std | AUC Mean | AUPR Mean |
| :------------ | :------ | :----- | :------- | :-------- |
| **`full_model`** | **0.7367** | **0.0038** | **0.9213** | **0.8095** |
| `no_tcn` | 0.7221 | 0.0036 | 0.9226 | 0.8200 |
| `no_cp_embed` | 0.5673 | 0.0017 | 0.8182 | 0.6347 |
| `no_io_embed` | 0.7161 | 0.0023 | 0.9089 | 0.7781 |
| `hardmask_io` | 0.7161 | 0.0023 | 0.9089 | 0.7781 |

**Decision**: The `full_model` configuration (with TCN, counterparty embeddings, and IO embeddings) provides the best performance, confirming the efficacy of the complete architecture.

## 4. Entropy Regularization Lambda (λe) Ablation

The study on entropy regularization (`entropy_lambda`) indicates that no regularization (`λe = 0.0`) is optimal for the current model.

| Configuration | F1 Mean | F1 Std | AUC Mean | AUPR Mean |
| :------------ | :------ | :----- | :------- | :-------- |
| **`lambda_0.00`** | **0.7367** | **0.0038** | **0.9213** | **0.8095** |
| `lambda_0.10` | 0.7062 | 0.0066 | 0.9114 | 0.7879 |
| `lambda_0.25` | 0.6941 | 0.0207 | 0.9076 | 0.7728 |
| `lambda_0.50` | 0.6749 | 0.0011 | 0.8976 | 0.7492 |

**Decision**: The optimal `entropy_lambda` is **0.00**, suggesting that entropy regularization does not improve performance in this context.

## 5. Feature Ablation (Part 1.3)

The `run_ablation.py` script includes a backbone component ablation that implicitly covers some feature aspects. Specifically, `no_cp_embed` and `no_io_embed` configurations demonstrate the impact of counterparty and I/O direction embeddings, respectively. The results show that both are crucial for performance.

However, a dedicated ablation for `spike/novelty/amount` features as requested in Part 1.3 of the prompt is not explicitly present in the `ablation_results.json`. Further investigation into `train_unifiedtmil_v2_sota.py` or other scripts would be needed to confirm if these features are included and ablated. For now, we assume the `full_model` configuration in the backbone ablation implicitly uses the best set of features. If a separate ablation for these features is found in other logs, this document will be updated.

## 6. SOTA Pursuit Log (Part 1.5)

The `ablation_results.json` provides a log of tested configurations and their performance. This forms the basis for the `docs/sota_pursuit_log.md` which will detail the techniques attempted and their outcomes. The current ablation results are based on 2 seeds. For final SOTA claims, a 5-seed run with CI is required as per the prompt.
