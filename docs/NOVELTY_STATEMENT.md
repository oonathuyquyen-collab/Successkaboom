# Tuyên bố Tính Mới (Novelty Statement)

Dự án **UnifiedTMIL** mang đến những đóng góp mới mẻ và quan trọng sau đây cho lĩnh vực phát hiện gian lận trên blockchain, đặc biệt hướng tới các tạp chí chất lượng cao (Q1/A*):

## 1. Kiến trúc Hợp nhất "End-to-End" (Genuinely Unified Architecture)
Khác với các hệ thống lai (hybrid) hiện có thường tách biệt quá trình trích xuất đặc trưng bằng Deep Learning và quá trình phân loại/xếp hạng bằng Machine Learning cổ điển (như GBDT, LambdaMART), UnifiedTMIL đề xuất một kiến trúc mạng neural **hoàn toàn end-to-end**. 
Mô hình sử dụng một hàm loss kết hợp (Joint Loss) để đồng thời tối ưu hóa khả năng phân loại tài khoản lừa đảo (Account-level) và định vị giao dịch khả nghi (Transaction-level) trong một lần chạy tiến (single forward pass). Điều này giúp giảm thiểu độ phức tạp của hệ thống, loại bỏ nhu cầu bảo trì nhiều pipeline huấn luyện riêng biệt, và cho phép gradient từ bài toán phân loại tinh chỉnh trực tiếp các trọng số chú ý (attention weights).

## 2. Định vị Giao dịch Giám sát Yếu (Weakly Supervised Transaction Localization)
Trong bối cảnh dữ liệu blockchain thực tế rất khan hiếm nhãn ở mức độ giao dịch (transaction-level labels), UnifiedTMIL tiên phong áp dụng phương pháp học giám sát yếu. 
Thay vì cần một tập dữ liệu lớn các giao dịch lừa đảo đã được gán nhãn thủ công để huấn luyện mô hình xếp hạng (như LambdaMART), UnifiedTMIL tận dụng trực tiếp trọng số của mạng Gated Attention ($\alpha_k$) làm điểm số khả nghi (suspicious score). Bằng cách áp dụng các hàm điều chuẩn (như Entropy Minimization Loss), mô hình bị ép buộc phải tập trung sự chú ý vào một số ít giao dịch quyết định tính chất lừa đảo của toàn bộ tài khoản.

## 3. Tích hợp Đặc trưng Kỹ thuật vào Cơ chế Chú ý (Engineered Features Injection into Attention)
Nhận thức được hạn chế của cơ chế attention thuần túy trong việc giải thích (Attention is not Explanation) và định vị, UnifiedTMIL giới thiệu kỹ thuật nối (concatenate) các đặc trưng kỹ thuật mạnh mẽ (Engineered features như uy tín đối tác `cp_reputation`, đột biến giao dịch `amount_spike`) trực tiếp vào biểu diễn ẩn (hidden representation) *trước* khi đưa qua lớp Attention.
Sự kết hợp này mang lại "tốt nhất của hai thế giới": khả năng học biểu diễn chuỗi thời gian của TCN và sức mạnh phân biệt của các đặc trưng do chuyên gia thiết kế, giúp cơ chế attention trở nên sắc bén và chính xác hơn hẳn trong việc định vị giao dịch.

## 4. Giao thức Đánh giá Minh bạch (Honest Evaluation Protocol)
Bài báo đi kèm với một quy trình đánh giá cực kỳ khắt khe, bao gồm Leave-One-Out (LOO) cross-validation cho localization, loại bỏ các giao dịch cuối cùng để tránh rò rỉ dữ liệu (leakage/positional bias), và báo cáo khoảng tin cậy (Confidence Intervals) thông qua bootstrap thay vì chỉ đưa ra các con số điểm (point estimates) dễ bị ảnh hưởng bởi nhiễu (cherry-picking). Điều này thiết lập một tiêu chuẩn mới về độ tin cậy cho các nghiên cứu bảo mật blockchain trong tương lai.
