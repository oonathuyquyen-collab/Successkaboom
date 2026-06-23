# Pre-Release Audit and Self-Review Report

## Self-Assessment Scores (1–5)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty | 3 | The "unified" framing of shared features for two tasks is the main contribution; the individual components (MLP, GBM) are standard. |
| Technical Soundness | 4 | All numbers come from code-backed runs; bootstrap CIs are reported; significance tests are honest. |
| Experiments | 3 | Single benchmark (PTXPhish); small hard-negative pool (n=80) limits hard-AUC confidence. |
| Reproducibility | 4 | All artifacts in results/; LOO protocol is deterministic; single seed. |
| Presentation | 4 | Clean IEEE format; no font errors; no text overlap; Figure 1 renders correctly. |

## Checklist

| Item | Status |
|------|--------|
| No "Lean SOTA"/"Lean"/"Manus"/"Kaboom"/"SuccessKaboom" in any file | PASS |
| No comparison with Readykaboom in tables/figures/text | PASS |
| All tables use name UnifiedTMIL | PASS |
| Numbers consistent: text <-> tables <-> results JSON | PASS |
| Title + Abstract claims match significance (no over-claim) | PASS |
| References present (10 real references) | PARTIAL — 10 refs present; expanding to 25+ recommended |
| Figure 1 present (UnifiedTMIL architecture only) | PASS |
| PDF compiles without font errors or text overlap | PASS |
| No sensitive data / API keys / files >50MB committed | PASS |
| reproduce_all.py runnable from clean environment | PASS |

## Summary

The paper has been rewritten with honest numbers from actual code runs. The key contribution is the unified feature-driven framework (UnifiedTMIL) that uses a single set of aggregate on-chain features for both account detection (X-AUC 0.984) and transaction localization (Hit@1 0.832). All legacy names have been replaced with UnifiedTMIL throughout the codebase, paper, and documentation.
