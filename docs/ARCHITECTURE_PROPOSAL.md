# Đề xuất Kiến trúc Unified Head (Phương án C - Joint Loss Function)

Dựa trên phân tích từ `ARCHITECTURE_CRITIQUE.md` và việc kiểm tra trực tiếp mã nguồn cũng như tập dữ liệu (dataset) của dự án Successkaboom, tài liệu này trình bày phương án tối ưu để hợp nhất kiến trúc thành một mô hình end-to-end duy nhất.

## 1. Cơ sở lựa chọn phương án

Quá trình kiểm tra dữ liệu thực tế (script `inspect_data.py`) cho thấy:
*   Tập dữ liệu huấn luyện (`train_bags.pkl`) **không có** nhãn mức giao dịch (transaction-level labels). Nhãn mức giao dịch (`gt_idx`) chỉ tồn tại trong tập đánh giá `ptx_bags.pkl`.
*   Do không có nhãn mức giao dịch trong tập huấn luyện, việc sử dụng các hàm loss giám sát trực tiếp (supervised ranking loss như ListMLE hay Pairwise loss) cho Head-L trong quá trình huấn luyện là **không khả thi**.

Do đó, phương án **Multi-task learning với Joint Loss (Phương án C)** kết hợp với cơ chế **Attention-as-Explanation** là lựa chọn duy nhất khả thi và mang tính đột phá về mặt học thuật.

## 2. Thiết kế Phương án C: Unified Head với Joint Loss

### 2.1. Ý tưởng cốt lõi
Chúng ta sẽ loại bỏ hoàn toàn các mô hình Gradient Boosting (LightGBM, XGBoost, LambdaMART) khỏi quá trình suy luận (inference). Mạng neural (IODirTMIL) sẽ được mở rộng để tự học cách phân loại tài khoản và định vị giao dịch đồng thời, chỉ dựa trên nhãn mức tài khoản (account-level label) trong quá trình huấn luyện.

### 2.2. Kiến trúc chi tiết
1.  **Backbone (Giữ nguyên):** Vẫn sử dụng BERT4ETH Embeddings và 1D-TCN để trích xuất đặc trưng chuỗi thời gian.
2.  **Unified Head:**
    *   **Gated Attention Network:** Tính toán trọng số $\alpha_k$ cho mỗi giao dịch.
    *   **Account Classification:** Context vector $z = \sum \alpha_k \cdot h_k$ đi qua một MLP để xuất ra xác suất tài khoản là phishing ($P_{account}$).
    *   **Transaction Localization (Attention-as-Ranking):** Sử dụng trực tiếp trọng số attention $\alpha_k$ làm điểm số (score) để xếp hạng mức độ khả nghi của giao dịch. Giao dịch có $\alpha_k$ cao nhất sẽ được dự đoán là giao dịch lừa đảo.
3.  **Engineered Features Injection (Quan trọng):**
    *   Theo phân tích trong bài báo hiện tại, attention thuần túy không đủ tốt. Các đặc trưng kỹ thuật (Engineered features như `cp_reputation`, `amount_spike`) đóng vai trò quyết định.
    *   *Cải tiến:* Thay vì đưa các đặc trưng này vào LambdaMART ở bước sau, chúng ta sẽ **nối (concatenate)** các đặc trưng kỹ thuật này vào biểu diễn $h_k$ *trước* khi đi qua mạng Gated Attention. Điều này ép mạng Attention phải xem xét các tín hiệu mạnh mẽ này khi tính toán $\alpha_k$.

### 2.3. Hàm Loss Kết hợp (Joint Loss)
Vì không có nhãn giao dịch, chúng ta sử dụng phương pháp học giám sát yếu (Weakly Supervised Learning) cho phần localization.

$\mathcal{L}_{total} = \mathcal{L}_{BCE} + \lambda \cdot \mathcal{L}_{attn\_reg}$

Trong đó:
*   **$\mathcal{L}_{BCE}$:** Binary Cross-Entropy loss chuẩn cho phân loại tài khoản.
*   **$\mathcal{L}_{attn\_reg}$:** Hàm điều chuẩn (Regularization loss) cho attention. Ví dụ:
    *   *Entropy Minimization:* Ép phân phối attention $\alpha_k$ trở nên sắc nét (sharp) hơn, tập trung vào một vài giao dịch cụ thể thay vì dàn trải đều (điều này phù hợp với bản chất lừa đảo thường chỉ nằm ở 1-2 giao dịch).
    *   *Instance-Max Consistency:* Đảm bảo rằng giao dịch có attention cao nhất phải chứa đủ thông tin để phân loại đúng tài khoản.

## 3. Lợi ích và Đánh đổi (Trade-offs)

### Lợi ích:
*   **End-to-End thật sự:** Toàn bộ hệ thống được huấn luyện bằng một hàm loss duy nhất, cho phép gradient chảy ngược từ quyết định phân loại về tận các đặc trưng giao dịch.
*   **Đơn giản hóa:** Loại bỏ hoàn toàn sự phụ thuộc vào các thư viện GBDT, giảm số lượng siêu tham số, dễ dàng triển khai (chỉ cần một model PyTorch duy nhất).
*   **Tính mới (Novelty) cao cho Q1/A*:** Đề xuất một phương pháp Weakly Supervised Transaction Localization thông qua việc nhúng đặc trưng kỹ thuật trực tiếp vào cơ chế Attention, chứng minh rằng mạng neural có thể tự định vị lừa đảo mà không cần nhãn giao dịch chi tiết.

### Đánh đổi:
*   **Hiệu năng Localization có thể giảm nhẹ:** LambdaMART (GBDT) nổi tiếng là rất mạnh trong bài toán xếp hạng dữ liệu dạng bảng. Việc dùng mạng neural thuần túy (với attention) có thể không đạt được Hit@1 cao như 0.832 của baseline cũ. Tuy nhiên, sự đánh đổi lấy tính "end-to-end" và tốc độ inference là hoàn toàn xứng đáng cho một bài báo học thuật.

## 4. Kế hoạch Cài đặt
1.  Tạo nhánh mới/thư mục mới `src/core/unified_model.py` để không ghi đè baseline cũ.
2.  Chỉnh sửa class `IODirTMIL` để nối (concatenate) các engineered features (sẽ cần viết hàm trích xuất các feature này thành tensor) vào $h_k$ trước lớp Attention.
3.  Thêm $\mathcal{L}_{attn\_reg}$ (Entropy loss) vào hàm `train_model`.
4.  Chạy thử nghiệm trên 3 seed và so sánh kết quả.
