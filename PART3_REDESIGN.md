# PHẦN 3: Thiết kế lại mô hình (Lean Architecture)

Dựa trên nguyên tắc Occam's Razor (ưu tiên giải pháp đơn giản nhất nhưng hiệu quả nhất), mô hình `UnifiedTMIL` cồng kềnh sẽ được thay thế bằng một **Two-Stage Pipeline** tối giản, gọi là **Lean SOTA**.

## Tiêu chí thiết kế
- **Lean Architecture:** Loại bỏ mọi thành phần Deep Learning không cần thiết (TCN, Attention, khổng lồ Embedding layer).
- **Minimal Complexity:** Dễ đọc, dễ code, dễ train, dễ reproduce.
- **Maximum Performance:** Tập trung vào các đặc trưng có tính phân loại cao nhất.
- **High Reproducibility:** Deterministic, không phụ thuộc nhiều vào random seed.

## Kiến trúc Lean SOTA đề xuất

Kiến trúc mới sẽ được chia làm 2 stage độc lập nhưng hoạt động nối tiếp nhau.

### Stage 1: Account-Level Detection (Phát hiện tài khoản)
Thay vì xử lý chuỗi giao dịch qua TCN, ta tổng hợp (aggregate) toàn bộ bag thành một vector đặc trưng có kích thước cố định (8 chiều), sau đó đưa qua một MLP rất nhỏ.

**1. Aggregate Feature Extractor:**
Với mỗi bag (tài khoản), trích xuất 8 đặc trưng thống kê:
1. `mean_log_amt`: Trung bình của log(1 + \|amount|)
2. `std_log_amt`: Độ lệch chuẩn của log(1 + \|amount|)
3. `max_log_amt`: Giá trị lớn nhất của log(1 + \|amount|)
4. `mean_log_dt`: Trung bình của log(1 + delta_time)
5. `std_log_dt`: Độ lệch chuẩn của log(1 + delta_time)
6. `bag_length`: Số lượng giao dịch trong bag
7. `outbound_ratio`: Tỉ lệ giao dịch gửi đi (outbound)
8. `unique_cp_ratio`: Tỉ lệ đối tác (counterparty) duy nhất trên tổng số giao dịch

**2. Lean Account MLP:**
Mô hình nơ-ron tối giản với khoảng 2,000 tham số (giảm 96% so với bản gốc).
```python
import torch.nn as nn

class LeanAccountMLP(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.mlp(x).squeeze(-1)
```

**3. Loss Function:**
Sử dụng Binary Cross Entropy (BCE) kết hợp với Margin Contrast Loss để phân tách tốt hơn giữa positive và hard-negative.
$Loss = BCE(p, y) + \beta \cdot \max(0, margin - (\bar{p}_{pos} - \bar{p}_{neg}))$
(Với $\beta = 0.3$, $margin = 0.3$)

### Stage 2: Transaction-Level Localization (Định vị giao dịch)
Nếu Stage 1 dự đoán tài khoản là lừa đảo (probability > threshold), Stage 2 sẽ chạy để tìm ra giao dịch lừa đảo cụ thể. Ta loại bỏ hoàn toàn Head-L Attention và thay bằng mô hình Ranking dựa trên Gradient Boosting (LightGBM/GBM).

**1. Transaction Feature Extractor:**
Với mỗi giao dịch trong bag, trích xuất 16 đặc trưng (không sử dụng attention weights):
1. `log_amt`: log(1 + \|amount|)
2. `amt_z`: Z-score của amount trong nội bộ bag
3. `amt_rank`: Xếp hạng của amount trong bag (0 đến 1)
4. `novelty`: 1 nếu counterparty này mới xuất hiện lần đầu trong bag, ngược lại 0
5. `inbound`: 1 nếu là giao dịch nhận, 0 nếu không
6. `outbound`: 1 nếu là giao dịch gửi, 0 nếu không
7. `position`: Vị trí tương đối của giao dịch trong bag (0 đến 1)
8. `zero_out`: 1 nếu là giao dịch outbound có giá trị 0 (thường là token approval), ngược lại 0
9. `amt_vs_cummax`: Tỉ lệ amount so với giá trị lớn nhất tích lũy từ đầu đến hiện tại
10. `is_runmax`: 1 nếu amount hiện tại lớn hơn hoặc bằng giá trị lớn nhất tích lũy
11. `local_z`: Z-score của amount so với cửa sổ cục bộ (±3 giao dịch lân cận)
12. `log_dt`: log(1 + delta_time)
13. `inbound_val`: `inbound` * `log_amt`
14. `dist_end_nl`: Khoảng cách phi tuyến tính tới cuối bag ($1 / (N - position)$)
15. `cp_freq`: Tần suất xuất hiện của counterparty trong bag
16. `cp_reputation`: Tỉ lệ ground-truth lịch sử của counterparty (tính bằng Leave-One-Out để tránh data leakage)

**2. Ranker Model:**
Sử dụng `GradientBoostingClassifier` hoặc `LightGBM` với cấu hình:
- `n_estimators`: 200
- `max_depth`: 3-5
- `learning_rate`: 0.05
Mô hình sẽ dự đoán xác suất (probability) cho từng giao dịch. Giao dịch nào có xác suất cao nhất sẽ được xếp hạng đầu (Rank 1).

## So sánh kiến trúc

| Tiêu chí | UnifiedTMIL (Bản gốc) | Lean SOTA (Bản mới) |
| --- | --- | --- |
| **Cách tiếp cận** | End-to-end Deep Learning | Two-stage: MLP + Tree-based Ranker |
| **Số lượng tham số** | ~50,000 | ~2,000 (giảm 96%) |
| **Feature extraction** | Deep Embeddings + TCN | Explicit Statistical Features |
| **Localization mechanism**| Masked Gated-Attention | Gradient Boosting Ranker |
| **Training Time** | Chậm (cần TCN, nhiều epoch, 3 seeds) | Rất nhanh (MLP nhỏ, 1 seed) |
| **Tính diễn giải (Interpretability)** | Thấp (Black-box attention) | Cao (Rõ ràng tầm quan trọng của từng feature) |

Kiến trúc này đảm bảo mọi module đều có lý do tồn tại rõ ràng, dễ dàng cài đặt và tối ưu hóa tối đa cho các metric mục tiêu.
