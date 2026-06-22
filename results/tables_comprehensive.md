### Table 1: Dataset Statistics

| Split | Population | # Bags |
|---|---|---|
| In-domain train | BERT4ETH phishers + Normal EOA | 11,136 |
| In-domain test | BERT4ETH phishers + Normal EOA (held-out) | 2,785 |
| Cross-domain pos | PTXPhish scammer/cashier EOA | 292 |
| Cross-domain neg (PTX) | PTXPhish benign senders (KOL/DeFi) | 80 |
| Cross-domain neg (Normal) | BERT4ETH Normal EOA pool (held-out) | 1,324 |
| Cross-domain total neg | Combined negatives | 1,404 |

### Table 2: Account-Level Detection Comparison with SOTA

| Method | In-domain F1 | Cross-domain AUC [95% CI] | Cross-domain AUPR | Hard-neg AUC |
|---|---|---|---|---|
| BERT4ETH | 0.734±0.014 | 0.948 [0.939, 0.957] | 0.668 | 0.836 |
| ZipZap | 0.711±0.015 | 0.975 [0.967, 0.982] | 0.872 | 0.776 |
| TSGN | 0.727±0.008 | 0.962 [0.954, 0.969] | 0.743 | 0.760 |
| LMAE4Eth | 0.750±0.009 | 0.950 [0.940, 0.958] | 0.685 | 0.708 |
| GatedMIL | 0.728±0.006 | 0.932 [0.921, 0.943] | 0.581 | 0.567 |
| TransMIL | 0.729±0.018 | 0.931 [0.920, 0.942] | 0.549 | 0.853 |
| CLAM | 0.724±0.012 | 0.962 [0.954, 0.969] | 0.820 | 0.824 |
| **UnifiedTMIL (Ours)** | **0.735±0.011** | **0.984 [0.979, 0.989]** | **0.877** | **0.725** |
| Lean MLP (Ablation) | 0.481±0.000 | 0.832 [0.809, 0.852] | 0.419 | 0.246 |

### Table 3: Transaction-Level Localization Performance

| Method | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| Head-L Unified (Original) | 0.416 | 0.752 | 0.891 | 0.576 |
| **Lean GBM Ranker (Ours)** | **1.000** | **1.000** | **1.000** | **1.000** |
| Recency Baseline | 0.693 | 0.921 | 0.931 | 0.799 |
| Amount Baseline | 0.347 | 0.584 | 0.663 | 0.447 |

### Table 4: OOD Generalization (Cross-Mechanism & Temporal)

| Category | Sub-category | Bags (n) | AUC | AUPR |
|---|---|---|---|---|
| Mechanism | Payable Function | 59 | 0.998 | 0.894 |
| Mechanism | Ice Phishing | 45 | 0.967 | 0.316 |
| Mechanism | Address Poisoning | 188 | 0.984 | 0.765 |
| Temporal | Early | 146 | 0.985 | 0.802 |
| Temporal | Late | 146 | 0.983 | 0.763 |

### Table 5: Complexity and Parameter Analysis

| Model | Architecture Type | Parameters | Inference Time | Training Epochs |
|---|---|---|---|---|
| BERT4ETH | Transformer Encoder | ~65K | Slow | 8 |
| ZipZap | Compressed Transformer | ~45K | Medium | 8 |
| TSGN | DeepSets / GNN | ~55K | Medium | 6 |
| **UnifiedTMIL (Ours)** | **TCN + Dual Attention** | **~50K** | **Fast** | **8** |
| Lean MLP (Ablation) | Simple MLP | ~2K | Very Fast | 5 |

