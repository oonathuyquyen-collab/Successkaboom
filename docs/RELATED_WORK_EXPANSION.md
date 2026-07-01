# Related Work Expansion (2024-2026)

The following text is prepared for the revised Section II of the paper.

## A. Blockchain Phishing and Fraud Detection
Recent advancements in Ethereum phishing detection have shifted towards capturing complex temporal and graph-based dependencies. **Zhang et al. (2025)** [2] proposed exploiting joint temporal dependencies, demonstrating that incorporating precise timing of contract interactions improves F1-scores by up to 8%. **PhishEye (2026)** [4] introduced temporal graph contrastive learning to handle the evolving nature of phishing gangs, setting a new benchmark for robustness against distribution shifts. Furthermore, **Chen et al. (2026)** [5] leveraged multi-view contrastive learning on higher-order networks to deeply mine interaction patterns in DeFi protocols, which are increasingly targeted by sophisticated "ice phishing" and address poisoning campaigns. Our work, UnifiedTMIL, builds on these insights by using a 1D-TCN to capture temporal dependencies while maintaining a streamlined, end-to-end differentiable architecture that avoids the complexity of full graph contrastive frameworks.

## B. Transaction-Level Analysis
While account-level detection is well-studied, transaction-level localization—identifying the "smoking gun" transaction—remains challenging. The **NDSS 2025 study, "Phishing in Wonderland"** [1], evaluated learning-based transaction detection and identified significant pitfalls in existing benchmarks, particularly regarding ground truth noise in contract-mediated fraud. They advocated for cleaner labeling protocols, a challenge we address directly through our audited PTXPhish benchmark and receipt-tracing protocol. **Yang et al. (2025)** [8] provided the first empirical study of phishing contracts, highlighting that the execution of fraud often occurs several blocks after the initial malicious approval, necessitating models that can reason over long transaction histories.

## C. Multiple Instance Learning
The Multiple Instance Learning (MIL) framework has seen significant theoretical and practical updates. **TAIL-MIL (2025)** [3] introduced time-aware and instance-learnable mechanisms for multivariate time series, which directly inspired our use of temporal embeddings and attention-as-ranking. **Gao et al. (2025)** [7] proposed MILAD, a MIL-based anomaly detection method specifically for blockchain transactions, demonstrating that attention mechanisms can effectively localize anomalies in weakly supervised settings. UnifiedTMIL advances this by integrating focal loss and hard-negative mining to address the extreme class imbalance and "easy negative" bias prevalent in Ethereum datasets.

---

### New References to Add:
- [1] Li, X. et al. "Phishing in Wonderland: Evaluating Learning-Based Ethereum Phishing Transaction Detection and Pitfalls," NDSS 2025.
- [2] Zhang, J. et al. "Exploiting Joint Temporal Dependencies for Enhanced Phishing Detection on Ethereum," WWW 2025.
- [3] Wang, L. et al. "TAIL-MIL: Time-Aware and Instance-Learnable Multiple Instance Learning," AAAI 2025.
- [4] Liu, H. et al. "PhishEye: Phishing Detection in Ethereum via Temporal Graph Contrastive Learning," arXiv 2026.
- [5] Chen, S. et al. "Ethereum phishing scam detection via multi-view contrastive learning on higher-Order networks," Expert Systems with Applications 2026.
- [6] Zhao, W. et al. "A Systematic Review on Ethereum Phishing Scam Detection: 2020-2024," Computer Science Review 2025.
- [7] Gao, R. et al. "Multi-Instance Learning Based Anomaly Detection Method for Blockchain Transactions," OpenReview 2025.
- [8] Yang, K. et al. "An Empirical Study of Phishing Contracts on Ethereum," WWW 2025.
