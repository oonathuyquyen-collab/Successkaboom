# Consistency Audit Report

## 1. Goal
Ensure all numbers, metrics, and claims in the final paper match the outputs of the executed code and logs.

## 2. Source of Truth
The canonical source of truth is the output of `unified_tmil/finish_pipeline.py`, which is saved in `results/comprehensive_results.json` and logged in `/tmp/finish_log.txt`.

## 3. Metric Checks

### 3.1. Account-Level Metrics (5 seeds)
- **ID-F1:**
  - Log: `0.7739 ± 0.0203`
  - Paper: `0.774 ± 0.020`
  - Status: **MATCH**
- **ID-AUC:**
  - Log: `0.9431 ± 0.0100`
  - Paper: `0.943 ± 0.010`
  - Status: **MATCH**
- **X-AUC:**
  - Log: `0.9974 ± 0.0035`
  - Paper: `0.997 ± 0.004`
  - Status: **MATCH**
- **Hard-AUC:**
  - Log: `0.6919 ± 0.0437`
  - Paper: `0.692 ± 0.044`
  - Status: **MATCH**

### 3.2. Transaction-Level Metrics (5 seeds)
- **Hit@1:**
  - Log: `0.3901 ± 0.0615`
  - Paper: `0.390 ± 0.062`
  - Status: **MATCH**
- **MRR:**
  - Log: `0.5483 ± 0.0488`
  - Paper: `0.548 ± 0.049`
  - Status: **MATCH**

### 3.3. Ensemble Metrics
- **ID-F1:**
  - Log: `0.8307`
  - Paper: `0.831`
  - Status: **MATCH**
- **Hard-AUC:**
  - Log: `0.6704`
  - Paper: `0.670`
  - Status: **MATCH**
- **Hit@1:**
  - Log: `0.4059`
  - Paper: `0.406`
  - Status: **MATCH**

### 3.4. Leakage Audit
- **Leaky Hit@1:**
  - Log: `0.465`
  - Paper: `0.465` (Mentioned in Section IV.E)
  - Status: **MATCH**
- **Clean Hit@1:**
  - Log: `0.455` (single seed), `0.390` (5 seeds)
  - Paper: `0.390` (Mentioned in Section IV.E as final clean result)
  - Status: **MATCH**

### 3.5. Dataset Statistics
- **Total Accounts:** 4,998 (Paper: Table I) -> MATCH
- **Clean GT:** 4,168 + 466 = 4,634 (92.7%) (Paper: Table I) -> MATCH
- **Train Bags:** 11,136 (Paper: Table I) -> MATCH
- **Test Bags:** 2,785 (Paper: Table I) -> MATCH

## 4. Conclusion
All metrics reported in the final LaTeX paper (`paper_final.tex`) strictly match the outputs of the automated pipeline. The inconsistencies present in the original repository (e.g., Hard-AUC 0.857 vs 0.696) have been resolved by standardizing on the 5-seed, leakage-controlled pipeline.
