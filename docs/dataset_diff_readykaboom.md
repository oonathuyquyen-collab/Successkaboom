# Dataset Audit: Successkaboom vs Readykaboom

## Audit Summary
A comprehensive bit-wise and content-level audit was performed between the datasets of the current repository (**Successkaboom**) and the source of truth (**Readykaboom**).

| Metric | Readykaboom | Successkaboom | Match |
|---|---|---|---|
| `ptx_bags.pkl` (total bags) | 532 | 532 | ✅ |
| `ptx_bags.pkl` (positives) | 292 | 292 | ✅ |
| `ptx_bags.pkl` (negatives) | 240 | 240 | ✅ |
| `ptx_bags.pkl` (MD5 Hash) | `66ca1631a1eeb6a1b7fc8f57ba60ef5a` | `66ca1631a1eeb6a1b7fc8f57ba60ef5a` | ✅ |
| `step2_gt_map.json` | Identical | Identical | ✅ |
| `defi_hard_bags.pkl` | 160 bags | 160 bags | ✅ |
| `normal_eoa_neg.pkl` | 1324 bags | 1324 bags | ✅ |

## Findings
- **Data Integrity:** The raw data files and processed bags are identical across both repositories.
- **Relabeling Protocol:** The `gt_idx` and `hashes` within the bags match perfectly, confirming that the contract-mediated relabeling (Seaport/NFT/internal) has been applied consistently.
- **Localization Bags:** Both repositories identify the same subset of 101 bags with valid internal ground truth for transaction-level localization.

## Conclusion
The discrepancies in performance metrics (e.g., Hit@1=0.996 vs Hit@1=0.832) are **not** due to dataset differences. The root cause is a **data leakage** in the transaction-level evaluation pipeline in Successkaboom and **cherry-picking** of best-seed results in the account-level reporting.

**Action Plan:**
1. Fix the `cp_reputation` leakage in `enhanced_localization.py`.
2. Re-evaluate using a strict Leave-One-Out (LOO) protocol where features are computed *inside* each fold.
3. Report mean±std across 5 seeds for all metrics.
