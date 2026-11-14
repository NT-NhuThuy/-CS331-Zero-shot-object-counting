# V7.2 — Phát hiện lỗi parser Qwen và đánh giá lại từ checkpoint

## Phát hiện và cách sửa

- Input được đối chiếu: `V7_2_Adaptive_Tiling_Exploratory_1301.zip` (1.301 dự đoán, 27 checkpoint tiling, 27 checkpoint Qwen). `V7_2_partial_checkpoint.zip` là checkpoint trung gian tương ứng; không phải các kết quả độc lập.
- Hàm `qwen_full_presence` trong notebook cũ chứa **ký tự backspace U+0008** trong regex: `r'<U+0008>(?:YES|NO|UNCERTAIN)<U+0008>'`, thay vì word boundary `r'\b(?:YES|NO|UNCERTAIN)\b'`.
- Do vậy, trường `full_presence` bị gán `UNCERTAIN` cho toàn bộ 27 trường hợp; mọi gate đều không được kích hoạt dù `raw_full` nhiều khi là `YES`.
- Bản đánh giá lại **không gọi Qwen hoặc CountGD++**: dùng nguyên dữ liệu thô `raw_full`, `raw_crops`, `tile_count` đã lưu; áp dụng **đúng quy tắc ban đầu** `full==YES`, >=4/6 crop trả YES, parse đầy đủ, >=4 detection. Không điều chỉnh ngưỡng dựa trên ground truth.
- Quy tắc kích hoạt gốc: `baseline_count == 0` và `v7_count == 0` (không dùng nhãn ground truth), nên trong 1.301 cặp bộ 27 đều là positive và 0 hard-negative kích hoạt.

## Kết quả đánh giá lại trên cùng tập 1.301

| Subset | N | V7 MAE | V7.2 bản lỗi | V7.2 sau parser fix |
|---|---:|---:|---:|---:|
| Positive | 1.190 | 18,2353 | 18,2353 | **16,7765** |
| Hard-negative | 111 | 4,3333 | 4,3333 | **4,3333** |
| All | 1.301 | 17,0492 | 17,0492 | **15,7148** |

- Positive MAE giảm **8.00% so với V7**; giảm từ 21.700 xuống 19.964 đơn vị lỗi tuyệt đối.
- 15/27 trường hợp được gate xác nhận; 15/1.301 dự đoán thay đổi; cả 15 đều thuộc positive; 12 trường hợp còn lại giữ nguyên.
- 41 → 26 ảnh positive dự đoán 0.
- Hard-negative MAE=4,3333 và FPR=14/111 (12,61%) **không đổi**, vì không có hard-negative nào kích hoạt tại tập kiểm tra hiện hành, không phải do visual gate đã được kiểm chứng thành công trên hard-negative mới.
- Bootstrap mô tả theo từng cặp positive (10.000 mẫu, seed 42) cho mức cải thiện MAE trung bình 95% percentile CI khoảng **0.560 đến 2.671**; đây không là CI xác nhận độ tổng quát hóa do TEST đã được tái sử dụng trong phát triển.

## Các trường hợp cải thiện nổi bật

| Ảnh | Target | Ground truth | V7 | V7.2 corrected | Giảm lỗi tuyệt đối |
|---|---|---:|---:|---:|---:|
| 6644.jpg | markers | 500 | 0 | 521 | 479 |
| 7167.jpg | markers | 217 | 0 | 235 | 199 |
| 3318.jpg | green peas | 185 | 0 | 191 | 179 |
| 6647.jpg | markers | 199 | 0 | 216 | 182 |
| 3311.jpg | green peas | 154 | 0 | 141 | 141 |

## Những điểm chưa giải quyết

- `1123.jpg` (3.701 markers) chỉ có 1/6 crop YES: V7.2 giữ 0 dù có 403 tiled detections.
- `7611.jpg` (2.560 legos) full Qwen trả NO và 0/6 crop YES: V7.2 giữ 0 dù 704 tiled detections.
- `6860.jpg` (512 legos) full Qwen YES nhưng 0/6 crop YES: giữ 0.
- Không dùng số liệu TEST này để tiếp tục tối ưu ngưỡng rồi công bố như test độc lập. Cần giữ nguyên V7.2 parser-corrected, đánh giá trên bộ positive/negative mới chưa sử dụng hoặc tập đánh giá giữ lại từ trước.

## Khả năng tái lập

1. Đầu vào: ZIP V7.2 gốc đã xuất (chứa 1.301 dự đoán và 27 Qwen log), **không** cần tải lại trọng số mô hình hoặc GPU.
2. Chạy `V7_2_Replay_NoGPU_1301.ipynb` đính kèm, hoặc đọc trực tiếp các CSV kết quả tái đánh giá.
3. `V7_2_Adaptive_Tiling_Kaggle_PARSER_FIXED.ipynb` là notebook suy luận GPU được sửa đúng regex **và** thêm cơ chế tái tính gate từ cache để tránh phải chạy lại nếu checkpoint đã tồn tại.
