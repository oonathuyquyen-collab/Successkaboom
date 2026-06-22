# PHẦN 2: Đánh giá kiến trúc hiện tại

Dựa trên việc phân tích source code và báo cáo `ANALYSIS_REPORT.md` (của Manus AI trước đó), dưới đây là đánh giá chi tiết về kiến trúc hiện tại của `UnifiedTMIL` để xác định những module thực sự cần thiết và những module dư thừa.

## 1. Những module thực sự cần thiết (Nên giữ)

| Module / Tính năng | Vì sao nên giữ | Mức độ ảnh hưởng tới Performance | Mức độ ảnh hưởng tới Complexity |
| --- | --- | --- | --- |
| **Engineered Features** (Amount stats, Time stats, Bag length, IO ratio, Unique CP ratio) | Đây là các tín hiệu cốt lõi quyết định việc phân loại tài khoản. Việc tổng hợp (aggregate) các đặc trưng này mang lại hiệu suất cực cao. | Rất cao (quyết định F1 và AUC) | Rất thấp (chỉ là các phép toán thống kê cơ bản) |
| **Margin Contrast Loss** | Giúp đẩy phân phối xác suất của positive và negative bags ra xa nhau, đặc biệt hiệu quả trong việc phân biệt với hard-negatives. | Cao (cải thiện đáng kể Hard-neg AUC) | Thấp (chỉ thêm một term vào loss function) |
| **IO Direction Feature** | Hướng giao dịch (inbound/outbound) là tín hiệu then chốt trong các vụ lừa đảo (ví dụ: tiền thường chảy ra từ victim vào scammer). | Cao | Thấp |
| **LOO Counterparty Reputation** (trong localization) | Đặc trưng quan trọng nhất để định vị giao dịch lừa đảo. Nếu một đối tác (counterparty) có lịch sử liên quan đến lừa đảo, khả năng giao dịch đó là lừa đảo rất cao. | Rất cao (quyết định Hit@K) | Trung bình (cần tính toán tần suất/tỉ lệ) |
| **Evaluation Protocol** | Cách đánh giá hiện tại (bootstrap CI, by-source breakdown, stratified AUC, artifact removal) cực kỳ chặt chẽ và chuẩn mực học thuật. | Rất cao (đảm bảo tính khoa học) | Không ảnh hưởng kiến trúc |

## 2. Những module dư thừa và làm tăng complexity (Nên bỏ)

| Module / Tính năng | Vì sao nên bỏ | Mức độ ảnh hưởng tới Performance | Mức độ ảnh hưởng tới Complexity |
| --- | --- | --- | --- |
| **TCN (Temporal Convolutional Network)** | Các đặc trưng tổng hợp (aggregate stats) đã đủ để nắm bắt thông tin tài khoản. TCN thêm tham số mà không mang lại cải thiện đáng kể so với MLP đơn giản. | Thấp (không làm tăng metric đáng kể) | Cao (tăng số lượng tham số và tính toán convolution) |
| **Head-L (Transaction Localization Head)** | `ANALYSIS_REPORT.md` đã chỉ ra rằng Attention weights không faithful cho localization. Các engineered features kết hợp với một ranker đơn giản (như GBM) cho kết quả tốt hơn nhiều (Hit@1: 0.845 vs 0.416). Việc cố gắng ép mô hình học chung (unified) làm giảm hiệu quả. | Âm (Làm giảm Hit@K so với dùng feature ranker) | Rất cao (Thêm một nhánh mạng nơ-ron, loss term, mask logic phức tạp) |
| **Counterparty Embedding (Vocabulary)** | Kích thước từ vựng (V) lớn làm tăng số lượng tham số khổng lồ (V x d). Hầu hết các counterparty là hiếm gặp (sparse), dẫn đến overfitting. Thay vào đó, dùng `unique_cp_ratio` hoặc `cp_reputation` hiệu quả hơn. | Thấp | Rất cao (Chiếm phần lớn số tham số của mô hình) |
| **3-Seed Ensemble** | Variance của các mô hình đơn giản (như MLP trên aggregate features) rất thấp. Chạy 3 seeds chỉ làm tăng thời gian training x3 mà không cải thiện kết quả đáng kể. | Thấp | Cao (Tăng thời gian training và inference) |
| **Attention Strengthening (Step 22)** | Các kỹ thuật entropy-sharpening penalty và instance-max consistency loss làm code cực kỳ phức tạp nhưng cuối cùng không beat được mô hình không dùng attention. | Âm | Rất cao |

## 3. Phân tích Code Smells & Code Lặp (Code Duplication)

- **Trùng lặp kiến trúc:** `step4_final.py` định nghĩa `IODirTMIL` trong khi `step7_unified.py` định nghĩa `UnifiedTMIL`. Cả hai gần như giống hệt nhau về mặt encoder. Cần gộp lại hoặc loại bỏ hoàn toàn để chuyển sang kiến trúc Lean mới.
- **Code chết / Code thí nghiệm:** Nhiều script (như `step22_headL_strong.py`, `step21_locwin2.py`, `step23_fusion_marginal.py`) chỉ là các thí nghiệm thất bại hoặc không cần thiết, làm rối repository.
- **Những phép xử lý có thể loại bỏ:** Việc tính toán mask phức tạp cho `Head-L` (outbound-masked) có thể bị loại bỏ hoàn toàn nếu ta bỏ `Head-L`.

## Tổng kết Đánh giá
Kiến trúc `UnifiedTMIL` hiện tại đang bị **Over-engineered**. Nó cố gắng giải quyết hai bài toán (Account-level và Transaction-level) bằng một mô hình deep learning duy nhất, dẫn đến việc phải sử dụng TCN, Attention phức tạp và các hàm loss gượng ép. 

**Hướng đi tối ưu (Lean Architecture):**
Tách bài toán thành 2 giai đoạn (pipeline) rõ ràng:
1. **Account-level:** Trích xuất các đặc trưng thống kê (Aggregate Features) từ bag và dùng một MLP nhỏ gọn để phân loại.
2. **Transaction-level:** (Chỉ chạy cho các tài khoản bị dự đoán là lừa đảo) Trích xuất các đặc trưng cho từng giao dịch (Engineered Features) và dùng một thuật toán ranking (như LightGBM/GBM) để định vị.

Cách tiếp cận này sẽ giảm 96% số lượng tham số, giảm 90% thời gian training, dễ dàng giải thích (interpretable) và mang lại kết quả (metric) cao hơn.
