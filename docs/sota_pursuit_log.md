# SOTA Pursuit Log

This document details the pursuit of State-of-the-Art (SOTA) results for the UnifiedTMIL model and clarifies the discrepancy between initial SOTA attempts and the final reported "clean" metrics in the paper.

## Initial SOTA Pursuit (unifiedtmil_v2_sota_results.json)

During initial experimentation, a version of UnifiedTMIL (referred to internally as `unifiedtmil_v2_sota`) achieved the following metrics:

- **Account-Level Detection:**
    - ID-F1: 0.744
    - X-AUC: 0.981
    - Hard-AUC: 0.696
- **Transaction-Level Localization:**
    - Hit@1: 0.802
    - MRR: 0.854

These results were initially considered for the paper. However, a subsequent audit revealed potential data leakage issues, particularly concerning the `cp_reputation` feature, which was globally available during training for these runs. This leakage led to an inflated performance, especially in localization tasks.

## Audited "Clean" Results (clean_results.json)

Following a thorough audit and mitigation of data leakage (as detailed in `docs/leakage_audit.md`), the UnifiedTMIL model was re-evaluated under a stricter, more robust protocol. The "clean" results, which are reported in the final paper, are as follows:

- **Account-Level Detection:**
    - ID-F1: $0.743\pm0.003$
    - X-AUC: $0.737$
    - Hard-AUC: $0.829\pm0.060$
- **Transaction-Level Localization:**
    - Hit@1: $0.106$
    - MRR: $0.267$

**Note:** The localization metrics for the clean results are significantly lower than the initial SOTA pursuit. This is primarily due to the strict exclusion of any features that could introduce positional bias or data leakage, ensuring a truly end-to-end and fair evaluation. The paper emphasizes architectural consistency and feature parsimony over inflated SOTA numbers achieved through potentially leaky features.

## Conclusion

The paper prioritizes the "clean" results from `clean_results.json` to ensure the integrity and reproducibility of the findings. While the initial SOTA pursuit yielded higher numbers, the audited results provide a more honest and reliable assessment of UnifiedTMIL's performance under rigorous conditions. The ablation study further supports the design choices made to achieve architectural consistency and interpretability.
