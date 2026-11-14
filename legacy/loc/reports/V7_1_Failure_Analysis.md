# Kết quả đánh giá V7.1 trên 1.301 ảnh–truy vấn (EXPLORATORY)

**Nguồn:** dữ liệu trong `V7_1_EXPLORATORY_eval_1301.zip` do nhóm xuất từ Kaggle. Phép tính dùng trực tiếp `v7_count`, `v7_1_count`, `gt_count` trên cùng 1.301 hàng.

## Kết luận

**Không thay thế V7 bằng V7.1.** V7.1 có MAE positive nhỉnh hơn một chút nhưng tăng lớn MAE/FPR hard-negative và làm xấu MAE gộp. Các con số dưới đây **chỉ là nghiên cứu khám phá**, không phải xác nhận tổng quát hóa trên test độc lập.

| Tập | n | V7 MAE | V7.1 MAE | Nhận xét |
|---|---:|---:|---:|---|
| Positive FSC-147 TEST | 1.190 | 18,2353 | 18,0059 | Giảm 0,2294 MAE |
| Hard-negative | 111 | 4,3333 | 12,2703 | Tăng 7,9369 MAE |
| Chung | 1.301 | 17,0492 | 17,5165 | Tăng 0,4673 MAE |

Hard-negative FPR: V7 **14/111 (12,61%)**, V7.1 **18/111 (16,22%)**. Zero-on-positive: V7 **41/1190 (3,45%)**, V7.1 **40/1190 (3,36%)**. V7.1 cải thiện tổng sai số tuyệt đối positive **273**, nhưng làm xấu tổng sai số hard-negative **881**; hiệu ứng ròng làm xấu **608** đơn vị sai số tuyệt đối.

## Phân tích 14 dự đoán thay đổi

- **9 positive:** 7 cải thiện, 2 trở nên kém hơn; một ảnh `2180.jpg` (apples, GT=223) thay 0 → 154, giảm sai số từ 223 xuống 69.
- **5 hard-negative:** cả 5 đều đếm sai hơn. Trong đó có 4 truy vấn tăng 0 → số đếm lớn vì V7.1 lật ngược quyết định `NEG_TEXT_EXEMPLAR` khi V4 còn giữ nhiều bounding box.
- `STRONG_TARGET_AND_FIVE_KEEPS` có 8 positive và 1 negative thay đổi. Trường hợp negative `CAND__6282__r1__balls` tăng số đếm 31 → 75. Như vậy Qwen `STRONG` và >=5 KEEP boxes **không đủ đảm bảo target thực sự tồn tại**.

### Năm hard-negative bị tác động

| Truy vấn | Đếm V7 → V7.1 | Sai số tăng |
|---|---:|---:|
| `CAND__2219__r1__red_beans` | 0 → 302 | +302 |
| `HARD__6661__r3__legos` | 0 → 242 | +242 |
| `HARD__4062__r2__watches` | 0 → 158 | +158 |
| `HARD__4952__r3__coins` | 0 → 135 | +135 |
| `CAND__6282__r1__balls` | 31 → 75 | +44 |

## Cách diễn giải khoa học

- V7.1 minh họa **trade-off recall–specificity**: cứu lại một số positive bị loại nhầm nhưng cũng khôi phục các detection thuộc confounder ở hard-negative.
- Kết quả cho thấy **retention ratio hoặc số KEEP boxes từ V4 không phải bằng chứng hiện diện độc lập**. Không nên dùng chỉ số đó một mình để lật kết quả `NEG_TEXT_EXEMPLAR`.
- 111 hard-negative đã được dùng trong quá trình phát triển V7; quy tắc V7.1 cũng đã được kiểm tra trên TEST. Vì vậy không báo cáo các điều chỉnh sau quan sát này như một cải tiến xác nhận trên held-out TEST.

## Hướng xử lý tiếp theo để viết paper nhanh mà đúng phương pháp

1. **Giữ V7 gốc là phương pháp chính** cho bảng kết quả hiện tại. Báo cáo V7.1 là ablation không thành công / failure analysis. 
2. Với phiên bản kế tiếp, yêu cầu **xác nhận hiện diện đối tượng độc lập** (Qwen binary presence trên ảnh toàn cảnh với truy vấn nguyên văn, tốt hơn thêm cross-check trên các crop) trước khi rescue số đếm. Không dùng `pair_type`, `gt_count`, `source_class`, hoặc tên lớp ground-truth trong quyết định suy luận.
3. **Tạo hard-negative mới thật sự chưa từng xem** và chọn quy tắc trên development/validation tách biệt. Giữ nguyên quy tắc khi chạy final TEST; công bố cả positive MAE, negative MAE và FPR.
4. Lưu nhật ký từng ảnh: output CountGD++, V4 kept boxes, V5 negative evidence, V6 semantic target evidence, nhánh V7, và rescue. Cần thiết cho tái lập kết quả.

**Kết quả đáng tin hiện tại:** V7 MAE positive 18,24; MAE negative 4,33; FPR 12,61%; MAE chung 17,05 trên 1.301 cặp kết hợp, **với chú thích rằng hard-negative đã được sử dụng để phát triển phương pháp**.
