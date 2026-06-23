# Báo Cáo Mâu Thuẫn (Inconsistency Report)

## 1. Table V (0.996) vs Table VI (0.832) — Nghi vấn Leakage Chính

**Mâu thuẫn:** 
Trong bài báo, Table V công bố kết quả của mô hình `UnifiedTMIL` với Hit@1 đạt **0.996** (gần như hoàn hảo), Hit@5/10 đạt 1.000, và MRR đạt 0.998 trên tập $n=101$ với giao thức Leave-One-Out (LOO). Tuy nhiên, ngay trong Table VI, kết quả của mô hình SOTA content-aware fusion (cùng sử dụng giao thức LOO trên cùng tập dữ liệu) lại chỉ đạt Hit@1 là **0.832**. Sự chênh lệch khổng lồ này giữa hai bảng biểu cho thấy một trong hai kết quả đã bị rò rỉ dữ liệu (data leakage) hoặc được tính toán bằng các quy trình không nhất quán.

**Nguyên nhân:**
Qua kiểm tra mã nguồn, mâu thuẫn này xuất phát từ sự khác biệt trong quy trình đánh giá giữa hai tập lệnh:
- Table V được sinh ra từ quá trình đánh giá có chứa feature `cp_reputation` được tính toán sai lầm. Trong một số kịch bản, đặc biệt là khi tính toán danh tiếng của counterparty (cp), dữ liệu từ chính bag đang được kiểm tra đã bị rò rỉ vào tập huấn luyện (train set) hoặc trực tiếp vào việc tính toán đặc trưng. Điều này dẫn đến hiện tượng mô hình "nhìn thấy" đáp án trước, đẩy Hit@1 lên mức phi lý 0.996 (thậm chí là 1.0 trong `lean_loc_results.json` với mô hình `lean_gbm_full`).
- Table VI (SOTA content-aware fusion) lại là kết quả của một quá trình đánh giá trung thực hơn (honest evaluation) hoặc là quá trình ablation (trong `loc_fusion_marginal.json`), nơi mà leakage đã được kiểm soát tốt hơn, dẫn đến Hit@1 thực tế chỉ ở mức 0.832.

**Kế hoạch khắc phục (Fix plan):**
- **Đồng bộ hóa giao thức:** Cần phải đảm bảo cả hai bảng đều sử dụng chung một quy trình đánh giá (clean protocol).
- **Loại bỏ Leakage:** Sửa lỗi tính toán feature `cp_reputation` để đảm bảo tuân thủ nghiêm ngặt nguyên tắc Leave-One-Bag-Out (LOO). Dữ liệu của bag đang test tuyệt đối không được tham gia vào bất kỳ khâu nào của việc tính toán đặc trưng cho chính nó.
- **Cập nhật bài báo:** Thay thế kết quả 0.996 trong Table V bằng kết quả trung thực (dự kiến trong khoảng 0.83–0.88), đồng thời điều chỉnh lại các nhận định "near-perfect" trong văn bản.

---

## 2. CI Vô Lý: X-AUC = 0.992 nhưng CI = [0.972, 0.989]

**Mâu thuẫn:**
Trong phần tóm tắt (Abstract) và phần kết quả của bài báo, giá trị trung bình của X-AUC được báo cáo là **0.992**. Tuy nhiên, khoảng tin cậy 95% (Confidence Interval - CI) đi kèm lại là **[0.972, 0.989]**. Về mặt toán học, giá trị trung bình (mean) bắt buộc phải nằm bên trong khoảng tin cậy. Việc CI nằm hoàn toàn dưới điểm ước lượng là một lỗi toán học nghiêm trọng.

**Nguyên nhân:**
Lỗi này phát sinh từ việc báo cáo kết quả của một seed duy nhất (có thể là seed tốt nhất - cherry-picking) làm điểm ước lượng chính (0.992), trong khi khoảng tin cậy lại được tính toán dựa trên phương pháp cluster-aware bootstrap từ toàn bộ các seed hoặc từ một phân phối khác (trung bình thực tế của các seed thấp hơn 0.992). Điều này tạo ra sự bất hợp lý giữa con số đại diện và khoảng tin cậy của nó.

**Kế hoạch khắc phục (Fix plan):**
- **Tính toán lại trung bình và CI:** Chạy lại quá trình đánh giá bootstrap với các seed cố định. Sử dụng giá trị trung bình thực sự của các seed làm điểm ước lượng chính.
- **Đảm bảo tính hợp lệ toán học:** Khoảng tin cậy được báo cáo phải bao quanh giá trị trung bình mới được tính toán.
- **Cập nhật bài báo:** Thay thế số liệu X-AUC 0.992 bằng giá trị trung bình chuẩn xác cùng CI hợp lệ trong toàn bộ văn bản (Abstract, Table II, và Sec. IV).

---

## 3. Hard-AUC Headline 0.857 vs Mean 0.696±0.148

**Mâu thuẫn:**
Bài báo sử dụng con số **0.857** làm headline cho chỉ số Hard-AUC (trong Abstract và Table II). Tuy nhiên, kết quả thực tế từ nhiều lần chạy (multi-seed runs) cho thấy giá trị trung bình của Hard-AUC chỉ là **0.696±0.148**. Việc chọn một giá trị cao bất thường làm đại diện đi ngược lại với nguyên tắc đánh giá trung thực (honest evaluation) mà bài báo đang cố gắng xây dựng.

**Nguyên nhân:**
Con số 0.857 là kết quả của một seed đặc biệt tốt (cherry-picking) hoặc là kết quả của mô hình ensemble (được tối ưu hóa quá mức cho một tập test cụ thể), chứ không phải là hiệu suất kỳ vọng thực tế của mô hình trên nhiều lần chạy độc lập.

**Kế hoạch khắc phục (Fix plan):**
- **Sử dụng Mean±Std:** Thay thế hoàn toàn con số 0.857 bằng giá trị trung bình và độ lệch chuẩn thực tế (`0.696±0.148` hoặc giá trị mới sau khi chạy lại sạch) trong Abstract, Table II, và các phần thảo luận.
- **Minh bạch hóa:** Nếu tác giả vẫn muốn giữ lại con số 0.857 để minh họa cho tiềm năng tối đa (hoặc hiệu suất của mô hình ensemble tốt nhất), con số này chỉ nên được đặt trong footnote hoặc ghi chú rõ ràng, không được dùng làm headline chính.
- **Điều chỉnh văn phong:** Cập nhật lại các nhận định so sánh (ví dụ: việc "đánh bại BERT4ETH (0.836)") để phản ánh đúng thực tế dựa trên giá trị trung bình.
