# Ghi chú hiểu dự án Successkaboom

## Nguồn đã đọc
- `paper/UnifiedTMIL_paper_final.pdf`
- `upload/pasted_content.txt`

## Hiểu nhanh về dự án
Dự án nghiên cứu một mô hình **Unified TMIL** để phát hiện tài khoản lừa đảo Ethereum ở hai mức:
1. **Account-level detection**: phân loại tài khoản phishing hay không.
2. **Transaction-level localization**: trong một bag giao dịch vào, xác định giao dịch lừa đảo đích thực.

Bài báo mô tả một backbone BERT4ETH chia sẻ, với hai head:
- **Head-C** cho mục tiêu phân loại tài khoản.
- **Head-L** cho mục tiêu định vị giao dịch.

Đóng góp được bài báo tự nhận gồm:
- kiến trúc hợp nhất một forward pass;
- benchmark đã audit với contract-mediated relabeling;
- protocol đánh giá “honest” cho localization.

## Các con số nổi bật đọc từ PDF hiện tại
- Account-level: ID-F1 khoảng `0.801`, X-AUC khoảng `0.992`, Hard-AUC khoảng `0.857`.
- Transaction-level Table V: Hit@1 khoảng `0.996`, Hit@5 `1.000`, Hit@10 `1.000`, MRR khoảng `0.998`.
- Table VI nêu fused content-aware ranking ở mức thấp hơn đáng kể, Hit@1 khoảng `0.832`.

## Mâu thuẫn đã thấy ngay từ PDF và chỉ thị người dùng
1. **Table V vs Table VI mâu thuẫn lớn**: cùng được mô tả là LOO protocol, n=101, nhưng Table V gần hoàn hảo còn Table VI chỉ khoảng 0.832.
2. **CI của X-AUC không hợp lệ**: mean khoảng `0.992` nhưng CI trong bài lại được nêu thấp hơn mean.
3. **Hard-AUC headline có dấu hiệu cherry-pick**: headline `0.857` mâu thuẫn với chỉ thị rằng mean thật có thể là `0.696±0.148`.

## Hướng kiểm tra kỹ thuật cần làm tiếp theo
- Xác định pipeline sinh **Table V**: nhiều khả năng liên quan `src/step23_fusion_marginal.py`, `src/sota.py`, và các file tổng hợp bảng trong `results/`.
- Xác định pipeline sinh **Table VI**: khả năng nằm ở script ablation như `src/loc_ablate_final.py` hoặc file tổng hợp.
- So sánh logic:
  - LOO split theo bag;
  - candidate set;
  - loại bỏ final transaction;
  - feature `cp_reputation` / drainer reputation;
  - train/test split của LambdaMART;
  - tiêu chí GT match.
- Kiểm tra toàn bộ thư mục `results/` để truy nguồn các số đã công bố.
- Kiểm tra `scripts/reproduce_all.py` để hiểu quy trình tái lập hiện tại.

## Mục tiêu deliverable theo yêu cầu người dùng
- `docs/inconsistency_report.md`
- `docs/leakage_audit.md`
- `results/clean_results.json`
- `paper/paper_en.tex` đã cập nhật
- `unifiedTMILpaper.pdf`
- push toàn bộ lên GitHub bằng token đã cung cấp

## Ghi chú bảo mật thao tác
Token GitHub và khóa Etherscan được người dùng cung cấp để phục vụ clone/push và truy xuất dữ liệu. Không hard-code vào file dự án; chỉ dùng qua môi trường hoặc URL tạm thời khi cần.

## Kết luận giai đoạn đọc ban đầu
Repo đã clone thành công. PDF xác nhận đúng là dự án học thuật về Unified TMIL và cũng cho thấy ngay các điểm bất thường mà người dùng yêu cầu audit. Bước tiếp theo là đọc các file `README`, `scripts/reproduce_all.py`, `src/step23_fusion_marginal.py`, `src/sota.py`, `src/loc_ablate_final.py`, cùng các JSON trong `results/` để truy ra pipeline sinh từng bảng và xác định nguồn leakage.

