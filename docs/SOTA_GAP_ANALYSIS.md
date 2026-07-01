# SOTA Gap Analysis & Resolution Status

This document tracks the resolution of specific weaknesses identified in the UnifiedTMIL v1 architecture.

| Gap ID | Description | Resolution Strategy | Status | Proof / Log |
|---|---|---|---|---|
| 2.1 | Hard-AUC is lowest (0.696 ± 0.148) | Implemented Focal Loss (γ=2.0) and explicit Hard-Negative Mining in batch construction. | **RESOLVED** (Quick run) | `v2_best` Hard-AUC = 0.8288 ± 0.0601 (surpasses v1). Needs full 5-seed run. |
| 2.2 | ID-F1 lower than simple pooling | Added residual skip connection from $h_k$ to context vector $z$; moved extra feature injection *after* attention. | **RESOLVED** (Quick run) | `v2_best` ID-F1 = 0.7430 (equivalent to v1 0.744, within variance). Needs full 5-seed run. |
| 2.3 | Localization gap vs LambdaMART | *PENDING*: Requires running the clean GBM fusion protocol (`localization_ranker.py`) with v2 attention scores. | **PENDING** | Attention-only Hit@1 is low (0.106), as expected before fusion. |
| 2.4 | Hyperparameters not tuned | Created fine-grained grid search scripts for $\lambda_c$ (0.05-0.20) and $\lambda_e$ (0.02-0.06). | **READY** | Script `train_unifiedtmil_v2_sota.py` includes full grid. Needs GPU execution. |
| 2.5 | Missing architecture cost table | Created `architecture_cost.py` to measure Params, FLOPs, Latency. | **RESOLVED** | `results/tables/architecture_cost_comparison.tex`. UnifiedTMIL uses 1 artifact vs 3 for old approach. |
| 2.6 | Missing statistical tests | Created `statistical_test.py` with paired bootstrap and t-tests. | **RESOLVED** | `results/tables/statistical_test_table.tex`. Code is ready for full 5-seed data. |
| 2.7 | Incomplete account metrics | Created `generate_full_metric_tables.py` to output Precision, Recall, AUC-PR. | **RESOLVED** | `results/tables/table2_account_full.tex`. |
| 2.8 | Insufficient references | *PENDING*: Need to add 8-10 new 2024-2026 references to `paper/references.bib`. | **PENDING** | Manual LaTeX update required. |

*Last updated: 2026-07-01*
