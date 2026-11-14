# Nhật ký nghiên cứu và giới hạn

## Bối cảnh dữ liệu

- FSC-147: 6.135 cặp positive toàn bộ; official TEST có 1.190 positive.
- Bộ negative cũ: 111 cặp đã xác minh, trải trên 96 ảnh và được dùng trong quá trình phát triển V4–V7.
- Bộ kết hợp official TEST: 1.190 positive + 111 negative = 1.301 cặp, ảnh có thể lặp với truy vấn khác nhau.
- V7.5: 2 lần đánh giá validation mỗi lần 240 ảnh, không trùng nhau; **chưa phải đánh giá pipeline V7 + V7.5 end-to-end trên negative mới**.

## Hành trình phát triển

- V4: Qwen3-VL xác minh candidate boxes từ CountGD++.
- V5: negative text và visual exemplar để loại confounder.
- V6: guarded router; còn nhiều false positives trên hard-negative.
- V7: dual-evidence router, giảm negative FPR xuống 12,61% trên 111 cặp cũ, nhưng positive MAE 18,24.
- V7.1: thử rescue evidence conflict; positive MAE cải thiện nhẹ, negative MAE tăng mạnh đến 12,27; bác bỏ.
- V7.2: targeted adaptive tiling ở các ca baseline đếm 0; đã sửa lỗi parse YES/NO; positive MAE 16,78.
- V7.3: xác minh crop kèm ngữ cảnh, positive MAE 16,29.
- V7.4: tiling 8×8 thí điểm, positive MAE 15,79; hiệu ứng chủ yếu từ một trường hợp markers cực đông.
- V7.5: nhánh multi-scale 4×4/8×8/12×12; đo riêng positive trên 480 validation ở hai đợt; 6 trường hợp dự đoán thay đổi.

## Tính khoa học

Các nhánh V7.2–V7.4 được hình thành từ phân tích lỗi TEST, nên các con số là exploratory. Không được gọi đó là 'unseen test'. V7.5 validation chỉ kiểm chứng nhánh multi-scale trên positive, không chứng minh giảm false positive ở truy vấn absent.

Bộ 160 cặp mới có 126 nhãn ABSENT do **một người** đánh giá. Khi xuất bản cần nêu quy trình và giới hạn xác minh. Không dùng ground truth để quyết định bật nhánh rescue/tiling khi chạy inference; GT chỉ dùng ở bước tính metric.

Để chuẩn bị paper cần chốt phương pháp, khóa protocol trước khi đánh giá tập mới, báo MAE positive, MAE negative, FPR, per-class, số ảnh thay đổi, runtime; so sánh công bằng trên cùng bộ dữ liệu.
