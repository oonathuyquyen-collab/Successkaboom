# Naming Mapping

This document provides the mapping between legacy names found in older versions of this repository and the standardized names used in the final `UnifiedTMIL` release.

| Legacy Name / Internal Component | Standardized Name | Description |
|----------------------------------|-------------------|-------------|
| Framework tổng / Lean SOTA / Kaboom / SuccessKaboom | **UnifiedTMIL** | The complete unified feature-driven framework for Ethereum phishing detection and localization. |
| Nhánh account (MLP trên aggregate features) | **UnifiedTMIL-Account** | The account-level detection component (Aggregate MLP). |
| Nhánh localization (feature-only GBM) | **UnifiedTMIL-Loc** | The transaction-level localization component (Feature-only GBM). |
| Bộ đặc trưng | **aggregate features** | The shared set of on-chain features used by both components. |
| Readykaboom / Readykaboom Fusion† | **(Removed)** | All comparisons with the prior internal architecture have been removed in favor of comparing directly against external state-of-the-art baselines. |

All code, figures, and documentation have been updated to reflect this naming convention.
