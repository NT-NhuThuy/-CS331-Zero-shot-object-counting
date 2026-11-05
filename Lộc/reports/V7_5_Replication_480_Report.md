# Đánh giá lặp lại V7.5 trên 480 ảnh FSC-147 validation

**Nguồn số liệu:** `V75_Blind_Validation_Pilot_240.zip` và `V75_REP_Blind_Validation_Pilot_240.zip`. Các số liệu được tính lại từ CSV dự đoán từng ảnh, không dùng số liệu sao chép từ log.

## Kiểm tra tính hợp lệ của dữ liệu

- Mỗi ZIP có **240/240** cặp ảnh–truy vấn positive; tổng **480 cặp không trùng `group_id`**.
- Tất cả 240 baseline của lần replication có trạng thái `ok`; 18 trường hợp kích hoạt multi-scale cũng có `status=ok`.
- Không thiếu nhãn hoặc số đếm; không có dự đoán âm; các giá trị absolute error được đối chiếu từ ground truth.
- SHA-256 của tệp dự đoán sealed trong replication khớp với bản ghi checksum.
- File `v75_frozen_protocol.json` và `v75rep_frozen_protocol.json` có cùng cấu hình suy luận (threshold 0.23, overlap 0.2, NMS IoU 0.5, baseline trigger ≤ 3, các grid 4/8/12, min_grid8 6, min_local_yes 5, max_scale_ratio 2.5, Qwen3-VL-4B-Instruct).

## Kết quả định lượng

| Tập | Số cặp | CountGD++ MAE | V7.5 MAE | Giảm tuyệt đối MAE | Giảm tương đối | Ảnh đổi dự đoán | Cải thiện / xấu đi |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lần 1: pilot 240 | 240 | 13.0917 | 7.9417 | 5.1500 | 39.34% | 4 | 4 / 0 |
| Lần 2: replication 240 | 240 | 12.1792 | 11.6792 | 0.5000 | 4.11% | 2 | 2 / 0 |
| **Gộp hai lần** | **480** | **12.6354** | **9.8104** | **2.8250** | **22.36%** | **6** | **6 / 0** |

Ở lần replication, `REPLICATION_A` (120 ảnh) không đổi (11.3333 → 11.3333). `REPLICATION_B` (120 ảnh) cải thiện 13.0250 → 12.0250.

**Lưu ý quan trọng:** 39,34% lần đầu **không tái xuất hiện** ở lần lặp lại. Lợi ích thực sự hiện diện trong tập mới, nhưng tỷ lệ kích hoạt/rescue thấp và hiệu quả phân bố không đều. Đừng báo kết quả gộp 22,36% như một lợi ích ổn định được kỳ vọng cho mọi tập dữ liệu.

## Sáu ảnh thực sự thay đổi

| Giai đoạn | Ảnh | Target | GT | Baseline | V7.5 | Số lỗi tuyệt đối giảm |
|---|---|---|---:|---:|---:|---:|
| Pilot | 3425.jpg | polka dots | 885 | 1 | 867 | 866 |
| Pilot | 3484.jpg | polka dots | 356 | 0 | 276 | 276 |
| Pilot | 5060.jpg | polka dots | 98 | 0 | 84 | 84 |
| Pilot | 734.jpg | ants | 20 | 0 | 10 | 10 |
| Replication | 4572.jpg | flamingos | 141 | 2 | 111 | 109 |
| Replication | 776.jpg | books | 15 | 3 | 14 | 11 |

Tổng giảm absolute error: **1.356** đối tượng trên 480 ảnh. Ảnh `3425.jpg` chiếm **63,9%** mức cải thiện gộp; trong lần replication, `4572.jpg` chiếm **90,8%** mức cải thiện. Nghĩa là hiệu quả đang tập trung vào một số ít ảnh đếm sai lớn.

## Audit cơ chế multi-scale ở lần replication

- 18/240 ảnh có CountGD++ dự đoán ≤ 3 nên kích hoạt xử lý multi-scale.
- 4/18 đến bước Qwen xác minh ảnh + crop; 2/18 được chấp nhận rescue.
- 2 trường hợp xác minh `YES` nhưng bị chặn bởi **scale-stability ratio**: `1397.jpg` (pills; 76 detection ở 8×8 nhưng chỉ 13 ở 12×12) và `4097.jpg` (54 → 13). Đây là các tín hiệu để phân tích, **không đủ lý do để hạ ngưỡng trực tiếp trên tập đã xem nhãn**.
- Số ảnh positive dự đoán bằng 0 ở lần replication: **15 trước và 15 sau**. Hai ảnh được cứu có baseline lần lượt 2 và 3, chứ không phải 0.
- Sai số lớn còn tồn tại: 1915.jpg bottle caps (GT 684, dự đoán 154, AE 530), 949.jpg birds (1092 vs 899, AE 193), 840.jpg books (637 vs 446, AE 191), 1397.jpg pills (166 vs 0, AE 166).

## Mức bất định

Dùng paired bootstrap theo ảnh (20.000 lần lấy mẫu với hoàn lại, seed 34021) cho **mức giảm MAE tuyệt đối**:

- Replication 240: **0.50**, CI bootstrap 95% xấp xỉ **[0.00, 1.45]**.
- 480 ảnh: **2.825**, CI bootstrap 95% xấp xỉ **[0.22, 7.26]**.

Các khoảng này mang tính mô tả, vì số ảnh đổi dự đoán rất ít và phân phối sai số lệch mạnh. **Không coi đây là bằng chứng đã chứng minh cải thiện ổn định có ý nghĩa thống kê.**

## Giới hạn và quyết định nghiên cứu

1. V7.5 ở đây là **nhánh adaptive multi-scale** được so với **CountGD++ trên positive validation**; chưa phải thử nghiệm đầy đủ V7.4/V7.5 trên cả presence-aware routing và hard-negative.
2. Hai tập 240 chỉ độc lập về **ảnh**, nhưng cùng đến từ FSC-147 validation; chưa phải kiểm tra khác miền dữ liệu. Sau khi đã mở nhãn và kết quả, chúng không còn là blind holdout cho những thay đổi tiếp theo.
3. Mốc TEST cũ V7.4 (positive MAE 15.79 và hard-negative MAE 4.33) thuộc **tập dữ liệu và pipeline khác**; không được so trực tiếp với MAE 9.81 của tập validation hiện tại như thể cùng một benchmark.

**Quyết định:** Đóng băng V7.5 hiện tại làm phương pháp ứng viên. Bước ưu tiên tiếp theo là lập bộ hard-negative mới được xác minh thủ công và kiểm tra rủi ro rescue khi target vắng mặt. Sau đó kiểm tra nhánh multi-scale trên một tập ảnh positive chưa dùng để phát triển. Không chỉnh quy tắc `max_scale_ratio` dựa trên ảnh `1397.jpg` và `4097.jpg` của replication đã mở nhãn.

### Các tệp kèm theo

- `v75_480_per_image.csv`: 480 dòng với dự đoán, ground truth và absolute error.
- `v75_480_changed_6.csv`: 6 ảnh bị thay đổi, để phân tích ca lỗi/ca thành công.
- `v75_480_recomputed_metrics.json`: thống kê tính lại và paired-bootstrap CI.
- `verify_v75_replication.py`: tính lại tất cả từ hai ZIP gốc, không cần GPU.
