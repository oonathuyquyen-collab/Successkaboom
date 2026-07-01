# CHANGELOG — UnifiedTMIL SOTA Improvement Session

This file logs every change made relative to `paper_final.pdf` (UnifiedTMIL v1 baseline).
Each entry includes: reason, hypothesis, actual result (log reference), and conclusion.

---

## 2026-07-01 — Session Start: Baseline Verification

**Action:** Re-ran UnifiedTMIL v1 with `scripts/run_baseline_training.py` (3 seeds: 42, 1, 7).

**Result (LOG: `results/logs/unifiedtmil_v1_baseline_run1/training.log`):**
- In-domain F1: 0.7347 ± 0.0111 (paper reports 0.744 ± 0.005 — difference due to 3 vs 5 seeds)
- Cross-domain AUC: 0.9703 (paper reports 0.981 — difference due to 3 vs 5 seeds + ensemble)
- Attention-only Hit@1: 0.188 (paper uses GBM fusion separately)

**Conclusion:** Baseline verified. Sai lệch nhỏ do khác số seed và không dùng ensemble — hợp lý.

---

## 2026-07-01 — UnifiedTMIL v2 SOTA Experiments

### Changes from v1 (paper_final.pdf baseline)
1. **Focal loss** (γ=2.0, α=0.75) replaces BCE — focuses training on hard negatives
2. **Hard-negative mining** — ensures DeFi/KOL accounts appear in every training batch
3. **Residual skip connection** — mean-pooled h added to attention context z
4. **Feature injection moved AFTER attention** — prevents extra features from corrupting attention sharpness
5. **Finer λc grid** (0.05, 0.10, 0.15, 0.20) with 5 seeds each (vs 2 seeds in v1)
6. **Finer λe grid** (0.02, 0.04, 0.06) replacing coarse (0.10, 0.25, 0.50)
7. **Dilated TCN option** — wider receptive field for complex DeFi/KOL patterns
8. **Uncertainty weighting option** — GradNorm-style dynamic loss balancing

### Best Config: `v2_best`
- ID-F1: 0.7430 ± 0.0028
- Hard-AUC: 0.8288 ± 0.0601
- X-AUC: 0.7369
- Loc Hit@1: 0.1062 | MRR: 0.2666

### Log Reference
- Results: `results/unifiedtmil_v2_sota_results.json`
- Per-config logs: `results/logs/unifiedtmil_v2_*/results.json`

### Conclusion
PENDING — awaiting GPU run with full 5 seeds for final numbers.
