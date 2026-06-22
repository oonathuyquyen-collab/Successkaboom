# PHẦN 1: Phân tích toàn bộ project Successkaboom

Dựa trên việc đọc toàn bộ source code và tài liệu của repository `Successkaboom`, dưới đây là phân tích toàn diện về kiến trúc và pipeline của hệ thống.

## 1. Kiến trúc tổng thể (Overall Architecture)
Hệ thống giải quyết bài toán phát hiện lừa đảo (phishing detection) trên Ethereum ở hai cấp độ đồng thời:
1. **Account-level (Cấp độ tài khoản):** Phân loại một tài khoản (EOA) có phải là scammer hay không dựa trên lịch sử giao dịch.
2. **Transaction-level (Cấp độ giao dịch):** Định vị (localize) chính xác các giao dịch lừa đảo cụ thể trong lịch sử của tài khoản đó.

Mô hình hiện tại (`UnifiedTMIL`) sử dụng phương pháp Multiple Instance Learning (MIL). Mỗi tài khoản được coi là một "bag" chứa các giao dịch ("instances"). Mô hình có kiến trúc "Unified Two-Head": một encoder dùng chung (shared encoder) và hai đầu ra (read-out heads) cho hai tác vụ trên, huấn luyện end-to-end trong một forward pass.

## 2. Data Pipeline & Feature Engineering
- **Dữ liệu thô:** Lịch sử giao dịch của các tài khoản được thu thập qua Etherscan API.
- **Bag Construction (`step3_build_bags.py`):** 
  - Lịch sử giao dịch được sắp xếp theo thời gian.
  - Lấy một cửa sổ cố định (sliding window) tối đa 100 giao dịch cuối cùng (L=100).
  - Đối với tập positive (scammer), cửa sổ được căn chỉnh sao cho kết thúc tại giao dịch lừa đảo cuối cùng (để đảm bảo có ground truth trong bag), sau đó loại bỏ giao dịch cuối cùng này (artifact removal) để tránh mô hình học "recency bias" quá mức.
- **Feature Engineering (per transaction):**
  - `input_ids`: ID của tài khoản đối tác (counterparty), được ánh xạ qua một vocabulary (V).
  - `input_io`: Hướng giao dịch (1 = Outbound, 2 = Inbound).
  - `input_amounts`: Giá trị giao dịch (đã log-transform: `log(1 + |amount|)`).
  - `delta_ts`: Thời gian trôi qua kể từ giao dịch trước đó (log-transform).

## 3. Dataset
- **In-domain (Train/Test):** Dữ liệu từ BERT4ETH (phishing bags và normal EOA bags). Không có ground-truth cấp độ giao dịch.
- **Cross-domain (Zero-shot Test):**
  - **Positives:** 292 tài khoản scammer từ PTXPhish (có ground-truth giao dịch).
  - **Negatives:** 80 tài khoản "hard negative" (DeFi/KOL active senders) từ PTXPhish + 1324 Normal EOAs held-out.

## 4. Model Architecture (`step7_unified.py`)
Kiến trúc `UnifiedTMIL` bao gồm:
1. **Shared Encoder:**
   - `cp_embed`: Embedding cho counterparty ID (kích thước `V x d`).
   - `io_embed`: Embedding cho hướng giao dịch (kích thước `3 x d`).
   - `hc_proj`: Linear projection cho các feature liên tục (amount, delta_ts).
   - `LayerNorm`
   - `TCN (Temporal Convolutional Network)`: 1D Convolution (kernel_size=3) để bắt thông tin ngữ cảnh thời gian.
2. **Head-C (Account Classification):**
   - Soft Gated-Attention: Tính trọng số attention $a^C$ cho mỗi giao dịch.
   - Bag pooling: Tổng có trọng số của các instance embeddings.
   - 2-Layer MLP: Phân loại tài khoản.
3. **Head-L (Transaction Localization):**
   - Outbound-Masked Gated-Attention: Tính trọng số attention $a^L$, mask (loại bỏ) các giao dịch inbound nếu bag có chứa giao dịch outbound.
   - Đầu ra là ranking của các giao dịch dựa trên $a^L$.

## 5. Loss Function, Optimizer & Training Strategy
- **Loss Function:** 
  $L = BCE(p^C, y) + \lambda \cdot BCE(p^L, y) + \beta \cdot L_{contrast}(p^C)$
  Trong đó $p^C$ và $p^L$ là dự đoán cấp độ tài khoản từ 2 head. $L_{contrast}$ là margin contrastive loss (đẩy xác suất trung bình của positive và negative bags cách nhau ít nhất 0.3).
- **Optimizer:** Adam (lr=1e-3).
- **Training Strategy:** Huấn luyện 8 epochs trên tập in-domain. Ensemble kết quả của 3 random seeds để giảm variance. Đánh giá zero-shot trên tập cross-domain PTXPhish.

## 6. Evaluation Pipeline
- **Account-level:** Đánh giá bằng Precision, Recall, F1, ROC-AUC, PR-AUC. Có phân tích chi tiết theo nguồn negative (hard vs easy) và phân tầng theo số lượng giao dịch (stratified).
- **Transaction-level:** Đánh giá bằng Hit@1, Hit@5, Hit@10, MRR. So sánh với các heuristic baselines (amount, recency, novelty, degree).
- **OOD Evaluation:** Đánh giá khả năng tổng quát hóa trên các cơ chế lừa đảo khác nhau (ice phishing, address poisoning, payable function) và theo thời gian.

## 7. Dependency Graph & Module Interaction
- `step1_audit.py` -> `step2_crawl.py` -> `step3_build_bags.py`: Chuẩn bị dữ liệu.
- `step7_unified.py`: Core training script. Import `collate` và `iterate` từ `step4_final.py`.
- `step4_final.py`: Đánh giá chi tiết (account & localization) cho mô hình đã train.
- `sota.py`: Huấn luyện và đánh giá các mô hình baseline (BERT4ETH, ZipZap, TSGN, LMAE4Eth, v.v.).
- `loc_ablate_final.py`: Phân tích chuyên sâu về localization bằng Gradient Boosting (GBM) trên các engineered features.
- Các script `step9_figures.py`, `step13_arch.py`: Sinh hình ảnh cho paper.

## Sơ đồ hệ thống (Mô tả văn bản)
```text
[Raw Tx Data] --> [Bag Construction (L=100, artifact removed)] --> [Feature Ext: IDs, IO, Amt, dT]
                                                                        |
                                                                        v
                                                         [Shared Encoder: Embeddings + TCN]
                                                                        |
                                            +---------------------------+---------------------------+
                                            |                                                       |
                                            v                                                       v
                             [Head-C: Soft Gated Attn]                               [Head-L: Masked Gated Attn]
                                            |                                                       |
                                            v                                                       v
                                 [Bag Vector Pooling]                                 [Attention Weights a_L]
                                            |                                                       |
                                            v                                                       v
                                 [MLP Classifier]                                       [Transaction Ranking]
                                            |                                                       |
                                            v                                                       v
                                  Account Prediction                                  Localization Prediction
```
