# Đối chiếu độc lập checkpoint V7.4 — 03/11/2026

## Phạm vi và kiểm tra tính toàn vẹn

- Nguồn: hai ZIP người dùng gửi (`V7_4_HighRes_Pilot_1301.zip`, `V7_4_HighRes_Partial_Checkpoint.zip`). Tính toán dựa trên dự đoán và ground truth có trong ZIP, **không chạy GPU**.
- 1.301 `group_id` duy nhất: 1.190 positive, 111 hard-negative; không trùng, không thiếu giá trị đếm.
- 6/6 ca 8×8 có trạng thái `ok` cả ở bước CountGD++ và Qwen. Trong đó 2/6 được chấp nhận; 4/6 giữ V7.3.
- Chỉ có 2 dự đoán thay đổi so với V7.3.

## Kết quả tái tính toán

| Phương pháp | Positive MAE | Hard-negative MAE | MAE chung |
|---|---:|---:|---:|
| CountGD++ | 17.7773 | 62.5495 | 21.5972 |
| V7 | 18.2353 | 4.3333 | 17.0492 |
| V7.2 | 16.7765 | 4.3333 | 15.7148 |
| V7.3 | 16.2950 | 4.3333 | 15.2744 |
| **V7.4** | **15.7899** | **4.3333** | **14.8125** |

- Positive MAE giảm 3.10% so với V7.3, 13.41% so với V7, 11.18% so với CountGD++.
- Hard-negative FPR **14/111 = 12,61%**, không đổi từ V7.3 vì V7.4 không sửa negative.
- Positive RMSE giảm từ 125.9244 xuống 113.4252.
- Positive dự đoán 0 vẫn **24/1.190**.

## Hai ca thay đổi

| Ảnh | Đối tượng | GT | V7.3 | V7.4 | Giảm sai số tuyệt đối |
|---|---|---:|---:|---:|---:|
| 1123.jpg | markers | 3701 | 403 | 996 | 593 |
| 7075.jpg | sheep | 195 | 170 | 178 | 8 |

Tổng sai số giảm 601 đối tượng. Riêng ảnh 1123 chiếm **98.67%** cải thiện V7.4 so với V7.3. Nếu bỏ ảnh này khỏi so sánh, lợi ích còn lại chỉ **8 đối tượng** trên 1.189 ảnh positive khác. Do đó không thể coi V7.4 đã cải thiện rộng khắp các nhóm ảnh.

## Kiểm tra 6 ca Qwen

- 1123: 403 → 996; 6/6 YES, tỷ lệ 2,4715; ACCEPT.
- 7075: 170 → 178; 6/6 YES, tỷ lệ 1,0471; ACCEPT.
- 3318: 191 → 183; 6/6 YES nhưng giảm số đếm, giữ 191.
- 6644: 521 → 503; 5/6 YES nhưng giảm số đếm, giữ 521.
- 6647: 216 → 187; 6/6 YES nhưng giảm số đếm, giữ 216.
- 7167: 235 → 197; 6/6 YES nhưng giảm số đếm, giữ 235.

## Hạn chế quan trọng và đề xuất vòng tiếp theo

1. Kết quả **exploratory**: các ảnh TEST đã nhiều lần tham gia lựa chọn và cải tiến V7.2–V7.4. Không báo như thành tích trên held-out test độc lập.
2. Hard-negative không đổi **do không đi qua nhánh tiling**, nên chưa có kiểm chứng an toàn khi mở rộng điều kiện kích hoạt sang negative mới.
3. Hai ca sai số lớn nhất sau V7.4 là `1123.jpg` (GT 3701, V7.4 996, AE 2705) và `7611.jpg` (GT 2560, V7.4 0, AE 2560). Không nên tự động nhân số đếm hay nâng các detection thành dự đoán cuối chỉ để khớp GT.
4. Nên chốt một quy tắc chọn ảnh **chỉ dựa trên prediction/evidence**, thử trên một tập phát triển riêng và so sánh multi-scale tiling với tiêu chí xác minh độc lập. Dành một tập positive mới và negative mới để đánh giá lần cuối, không điều chỉnh phương pháp dựa vào tập đó.

**Kết luận:** V7.4 có cải thiện đo được trên benchmark hiện tại, nhưng kết quả chủ yếu đến từ một ca cực đoan. Phương pháp có tiềm năng, song chưa đủ chứng minh tổng quát hóa.
