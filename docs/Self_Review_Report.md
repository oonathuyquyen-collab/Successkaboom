# Self-Review & Pre-Release Audit Report

## 1. Quality Scores (1-5)
- **Novelty (4/5):** The unified approach using a shared set of features for both account detection and transaction localization is a solid contribution, directly challenging the assumption that complex attention mechanisms are required for this domain.
- **Technical Soundness (5/5):** Rigorous methodology, including LOO evaluation for localization, strict separation of train/test for counterparty reputation, and proper use of bootstrap confidence intervals to address the small hard-negative pool.
- **Experiments (5/5):** Comprehensive evaluation across multiple seeds, clear ablation studies, and honest reporting of high-variance metrics.
- **Reproducibility (5/5):** A single `reproduce_all.py` script generates all results from raw data. All dependencies and configurations are documented.
- **Presentation (5/5):** The paper is formatted in IEEE standard, compiles cleanly, has professional figures, and maintains a clear, honest narrative.

## 2. Potential Reviewer Attacks & Mitigations
- **Attack:** "The dataset is too small (only 80 hard negatives) to claim SOTA."
  - **Mitigation:** We explicitly acknowledge this in the Limitations section and report bootstrap 95% CIs instead of just point estimates. The narrative honestly frames the results as "competitive while additionally localizing" rather than an absolute, unassailable SOTA.
- **Attack:** "The high Hit@1 (0.996) seems too good to be true, possible data leakage?"
  - **Mitigation:** We implemented a strict Leave-One-Out (LOO) protocol and removed the detection-cutoff artifact (last transaction) to ensure no leakage. The high performance stems from the highly stereotyped nature of the attacks and the strong recency prior, which we explicitly analyze and compare against.
- **Attack:** "Why not use a deeper graph neural network?"
  - **Mitigation:** Our ablation study directly answers this: learned attention and complex structures add no statistically significant marginal value over engineered features for localization ($p \approx 0.68$).

## 3. Pre-Release Checklist
- [x] Transaction-level kept intact + fully presented; numbers consistent.
- [x] Bolded cells reflect real, reproducible numbers; no losing cells are bolded.
- [x] Title/Abstract/Conclusion perfectly match the tables and do not over-claim.
- [x] Numbers consistent across text, tables, figures, code, README, and CSVs.
- [x] $\ge 5$ seeds + CI + p-values for all account metrics.
- [x] No "Lean SOTA", "Kaboom", "Manus" or comparisons to "Readykaboom".
- [x] References $\ge 25$, all real, no placeholders.
- [x] Figure 1 correct; PDF compiles cleanly with no font/overlap issues.
- [x] `reproduce_all.py` runs and generates the exact numbers reported.
- [x] No sensitive tokens, data, or huge files committed.
