# Dataset Verification Report (Successkaboom vs Readykaboom)

## 1. Executive Summary
A comprehensive bit-wise and content-level audit was performed between the datasets of the current repository (**Successkaboom**) and the source of truth (**Readykaboom**). The datasets are identical in content, and all discrepancies in performance metrics have been traced to pipeline leakage rather than data differences.

## 2. Dataset Statistics Verification

| Metric | Readykaboom | Successkaboom | Match |
| :--- | :--- | :--- | :--- |
| `ptx_bags.pkl` (total bags) | 532 | 532 | **Yes** |
| `ptx_bags.pkl` (positives) | 292 | 292 | **Yes** |
| `ptx_bags.pkl` (negatives) | 240 | 240 | **Yes** |
| `ptx_bags.pkl` (MD5 Hash) | `66ca1631a1...` | `66ca1631a1...` | **Yes** |
| `defi_hard_bags.pkl` | 160 bags | 160 bags | **Yes** |
| `normal_eoa_neg.pkl` | 1324 bags | 1324 bags | **Yes** |

## 3. Key Audit Findings
- **Data Integrity**: The raw data files and processed bags are identical across both repositories.
- **Relabeling Protocol**: The `gt_idx` and `hashes` within the bags match perfectly, confirming that contract-mediated relabeling (Seaport/NFT/internal) has been applied consistently.
- **Localization Bags**: Both repositories identify the same subset of **101 bags** with valid internal ground truth for transaction-level localization.

## 4. Discrepancy Analysis
The discrepancies in performance metrics (e.g., Hit@1=1.000 vs Hit@1=0.832) are **not** due to dataset differences. The root cause is:
1. **Data Leakage**: Global `cp_reputation` in the legacy localization pipeline.
2. **Reporting Bias**: Cherry-picking of best-seed results in account-level reporting in previous versions.

## 5. Conclusion
The dataset is **correctly built** and consistent with the reference repository. All subsequent performance improvements must come from technical enhancements (Section 1.5) rather than data manipulation.
