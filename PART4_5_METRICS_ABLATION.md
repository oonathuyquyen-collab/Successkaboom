# PHẦN 4: Cải thiện Metric

Mục tiêu của kiến trúc **Lean SOTA** không chỉ là giảm độ phức tạp mà còn phải vượt qua (hoặc ít nhất là duy trì) các metric của mô hình gốc trên cả hai cấp độ.

## 1. Cải thiện Account-level Metrics
**Các metric mục tiêu:** Precision, Recall, F1-score (in-domain), ROC-AUC, PR-AUC (cross-domain zero-shot).

**Chiến lược cải thiện:**
1. **Loại bỏ nhiễu từ Embedding:** Mô hình gốc sử dụng embedding cho counterparty ID. Do số lượng ID cực lớn và phân tán (sparse), mô hình dễ bị overfit trên tập in-domain và giảm khả năng tổng quát hóa (generalization) trên tập cross-domain. Bằng cách thay thế ID bằng `unique_cp_ratio` (tỉ lệ ID duy nhất), mô hình bắt được hành vi (behavior) thay vì ghi nhớ ID, giúp tăng Cross-domain AUC và giảm False Positives.
2. **Tập trung vào Amount và Time Statistics:** Lừa đảo thường có dấu hiệu bất thường về dòng tiền (amount) và tần suất (time). Việc trích xuất trực tiếp các đặc trưng thống kê (mean, std, max) giúp MLP học được các ranh giới quyết định (decision boundaries) rõ ràng hơn so với việc để TCN tự học từ raw sequence. Điều này cải thiện F1-score và Precision.
3. **Margin Contrast Loss:** Duy trì hàm loss này từ bản gốc để tiếp tục ép xác suất dự đoán của positive bags và hard-negative bags ra xa nhau, trực tiếp tăng ROC-AUC trên tập hard-negative (PTX-benign).

**Dự kiến kết quả:**
- Cross-domain AUC tổng hợp: Tăng nhẹ hoặc giữ nguyên (~0.990).
- Hard-negative AUC (PTX-benign): Cải thiện rõ rệt nhờ giảm overfit.
- Tăng độ ổn định (Robustness): Phương sai (variance) giữa các lần chạy sẽ giảm gần bằng 0.

## 2. Cải thiện Transaction-level Metrics
**Các metric mục tiêu:** Hit@1, Hit@5, Hit@10, MRR (Mean Reciprocal Rank).

**Chiến lược cải thiện:**
1. **Thay thế Attention bằng Explicit Ranker:** Attention weights trong mô hình gốc bị phân tán (diffused) cho nhiều giao dịch để phục vụ bài toán phân loại account, dẫn đến không chính xác (không faithful) khi dùng để định vị. Thay vào đó, dùng một Gradient Boosting Ranker được huấn luyện trực tiếp để phân biệt giao dịch lừa đảo vs bình thường sẽ tối ưu trực tiếp cho bài toán ranking, giúp tăng mạnh Hit@1 và MRR.
2. **Tận dụng Counterparty Reputation:** Tính toán `cp_reputation` (tỉ lệ một địa chỉ từng liên quan đến lừa đảo trong tập huấn luyện, dùng Leave-One-Out để tránh rò rỉ dữ liệu). Đây là tín hiệu cực mạnh (strong prior) giúp mô hình "chỉ điểm" ngay lập tức giao dịch khả nghi, giảm mạnh False Negatives trong việc định vị.
3. **Feature kết hợp (Interaction Features):** Các feature như `inbound_val` (inbound * amount) hay `zero_out` giúp bắt các pattern lừa đảo đặc thù (như Ice Phishing thường là outbound approval với amount=0, hoặc Payable Function thường là inbound với amount lớn). Các cây quyết định (Decision Trees) trong GBM rất giỏi trong việc kết hợp các feature này.

**Dự kiến kết quả:**
- Hit@1: Tăng vọt (dự kiến từ ~0.41 lên >0.80).
- Hit@10: Đạt mức bão hòa (>0.95).
- Giảm False Positives trong việc chỉ định nhầm các giao dịch lớn nhưng hợp lệ.

---

# PHẦN 5: Ablation Study

Để chứng minh tính hiệu quả của kiến trúc Lean SOTA và sự đóng góp của từng thành phần, một Ablation Study toàn diện sẽ được thiết kế.

## 1. Account-level Ablation (Stage 1)
Thiết kế các thí nghiệm loại bỏ (remove) từng nhóm feature hoặc module để xem ảnh hưởng tới F1 (in-domain) và AUC (cross-domain).

| Thí nghiệm | Mô tả | Mục đích chứng minh |
| --- | --- | --- |
| **Full Lean SOTA** | Sử dụng toàn bộ 8 aggregate features + Margin Loss | Baseline tốt nhất |
| **Remove Amount Stats** | Bỏ `mean_log_amt`, `std_log_amt`, `max_log_amt` | Chứng minh tầm quan trọng cốt lõi của thông tin dòng tiền |
| **Remove Time Stats** | Bỏ `mean_log_dt`, `std_log_dt` | Đánh giá ảnh hưởng của tần suất giao dịch |
| **Remove IO Ratio** | Bỏ `outbound_ratio` | Chứng minh hướng dòng tiền là tín hiệu phân loại mạnh |
| **Remove Unique CP Ratio** | Bỏ `unique_cp_ratio` | Đánh giá hành vi tương tác với nhiều/ít đối tác |
| **Remove Margin Loss** | Chỉ dùng BCE Loss | Chứng minh vai trò của Contrastive learning đối với hard-negatives |
| **Replace with RF** | Dùng Random Forest thay cho MLP | Chứng minh MLP phù hợp hơn cho các feature liên tục này |

## 2. Transaction-level Ablation (Stage 2)
Thiết kế các thí nghiệm loại bỏ từng nhóm feature trong tập 16 features của GBM.

| Thí nghiệm | Mô tả | Mục đích chứng minh |
| --- | --- | --- |
| **Full Lean SOTA Ranker** | 16 features, không dùng Attention | Baseline tốt nhất cho Localization |
| **Remove CP Reputation** | Bỏ `cp_reputation` | Chứng minh giá trị của lịch sử đối tác |
| **Remove Amount Features** | Bỏ `log_amt`, `amt_z`, `amt_rank` | Chứng minh lừa đảo thường liên quan đến lượng tiền bất thường |
| **Remove Position/Time** | Bỏ `position`, `dist_end_nl`, `log_dt` | Đánh giá mức độ ảnh hưởng của "recency bias" |
| **Remove Zero Outbound** | Bỏ `zero_out` | Đánh giá khả năng bắt Ice Phishing (Approve) |
| **Add Attention (Original)** | Thêm `attn` và `attn_rank` từ mô hình gốc vào tập feature | Chứng minh Attention thực sự là nhiễu (noise) và làm giảm performance |
| **Recency Only** | Chỉ xếp hạng dựa trên giao dịch gần nhất | Heuristic baseline |

## 3. Bảng phân tích độ phức tạp (Complexity Analysis)
Sau khi có kết quả, một bảng so sánh toàn diện sẽ được tạo ra để chứng minh "Lean" tốt hơn "Complex":

| Model | Precision | Recall | F1 | Complexity | Parameters | Training Time |
| --- | --- | --- | --- | --- | --- | --- |
| UnifiedTMIL (Original) | [val] | [val] | [val] | High (TCN+Attn) | ~50K | ~150s |
| Lean SOTA (Proposed) | [val] | [val] | [val] | Minimal (MLP) | ~2K | ~15s |

*(Lưu ý: Các giá trị [val] sẽ được điền chính xác trong Phần 10 (Sinh toàn bộ bảng) dựa trên kết quả chạy thực nghiệm).*
