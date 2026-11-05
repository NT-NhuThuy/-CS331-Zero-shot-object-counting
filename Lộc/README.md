# Lộc — CS331 Zero-shot / Open-vocabulary Object Counting

> Nhánh nghiên cứu sử dụng **CountGD++ + Qwen3-VL**, kiểm chứng hiện diện mục tiêu và cải thiện đếm theo đa tỷ lệ. Đây là tài liệu tiến độ nhóm, **chưa phải phương pháp cuối hoặc kết quả đã được xác nhận trên held-out độc lập**.

## Mục tiêu

1. Giảm false positive khi đối tượng được hỏi **không xuất hiện** trong ảnh (`hard-negative`, GT = 0).
2. Giảm đếm thiếu khi đối tượng xuất hiện nhiều, nhỏ, dày đặc (`adaptive multi-scale tiling`).
3. Xác minh ngữ nghĩa/đơn vị đếm bằng mô hình thị giác–ngôn ngữ, tránh đếm nhầm confounder.

## Thư mục

- `notebooks/01_original_pipeline/`: mã nguồn nghiên cứu V4–V7 do nhóm đã thực thi (đổi tên cho rõ chức năng; **không chỉnh source code**).
- `notebooks/02_joint_test/`: ghép FSC-147 + 111 hard-negative và suy luận TEST.
- `notebooks/03_refinements/`: ablation V7.1 và các phiên bản V7.2–V7.4.
- `notebooks/04_validation/`: V7.5 multi-scale trên hai bộ validation khác nhau.
- `notebooks/05_fresh_negative/`: chuẩn bị và kiểm thử safety trên hard-negative mới.
- `results/`: kết quả dự đoán từng ảnh, metrics, audit và protocol.
- `data/`: bảng nhãn và metadata **không chứa ảnh gốc FSC-147**.
- `reports/`, `docs/`, `tools/`: phân tích và công cụ phụ trợ.

**Các notebook đã được xóa output lưu trong file để repository gọn hơn; nội dung code và Markdown được giữ nguyên.**

## Trạng thái thí nghiệm (06/11/2026)

| Phương pháp / tập | Số mẫu | MAE positive ↓ | MAE negative ↓ | Ghi chú |
|---|---:|---:|---:|---|
| CountGD++ trên tập TEST kết hợp | 1.190 positive + 111 negative | 17,78 | 62,55 | Baseline; positive/negative đánh giá riêng |
| V7 | 1.190 + 111 | 18,24 | 4,33 | FPR negative 12,61% |
| V7.1 | 1.190 + 111 | 18,01 | 12,27 | **Không được chấp nhận**: negative xấu đi |
| V7.2 sửa parser | 1.190 + 111 | 16,78 | 4,33 | Exploratory; nhánh tiling không kích hoạt trên negative cũ |
| V7.3 | 1.190 + 111 | 16,29 | 4,33 | Exploratory |
| V7.4 | 1.190 + 111 | 15,79 | 4,33 | Exploratory; cải thiện phần lớn do ảnh 1123.jpg |
| V7.5 validation lần 1 | 240 positive | 7,94 | — | CountGD++ cùng tập: 13,09 |
| V7.5 validation lặp lại | 240 positive mới | 11,68 | — | CountGD++ cùng tập: 12,18 |
| V7.5 validation tổng | 480 positive | 9,81 | — | CountGD++ cùng tập: 12,64; chỉ 6 ảnh thay đổi |

**Không lấy các dòng V7.5 (validation positive) để so sánh trực tiếp với bảng V7.4 (TEST 1.190 + 111).** Các tập, mục tiêu và cấu hình khác nhau. V7.5 chưa chứng minh safety trên hard-negative mới.

## Bộ hard-negative mới

`data/fresh_negative_160_manual_review.csv`: 160 cặp được **một** người xem và gán nhãn; 126 `ABSENT` và 34 `PRESENT`. `data/fresh_negative_126_absent.csv` chứa 126 cặp đề xuất dùng cho phép kiểm thử. Chưa có kết quả suy luận safety trên các cặp này trong bản bàn giao. Không mô tả chúng như nhãn đã được hai người xác nhận.

## Tái lập và giới hạn

- Yêu cầu dữ liệu/ảnh FSC-147 và trọng số mô hình do người chạy chuẩn bị riêng; repo không phân phối ảnh gốc, weights, API token hay checkpoint GPU nặng.
- Kaggle cần GPU/Internet cho notebook inference; notebook đánh giá/đối chiếu có thể chạy CPU.
- V4–V7 và V7.2–V7.4 được phát triển sau khi quan sát các tập TEST/hard-negative tương ứng: **các số trên mang tính exploratory**, không phải kết quả tổng quát hóa độc lập.
- V7.5 multi-scale trên 480 positive đã lặp lại nhưng mức tăng chủ yếu tập trung ở ít ảnh; bước tiếp theo là chạy safety test trên 126 negative mới.
- Xem `docs/THI_NGHIEM_VA_GIOI_HAN.md` và `docs/CACH_CHAY.md`.
