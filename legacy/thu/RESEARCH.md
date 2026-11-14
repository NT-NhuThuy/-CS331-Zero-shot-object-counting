# Phương pháp nghiên cứu

## Bài toán
CountGD++ có thể đếm trùng, bỏ sót đối tượng hoặc đếm cả khi đối tượng được truy vấn không tồn tại.

## Phương pháp
- **V4:** điều chỉnh đóng góp của các proposal để cải thiện HOW MANY.
- **Joint V3.2:** kết hợp tín hiệu cấu trúc và semantic reliability để quyết định correction.
- **A3:** kiểm tra sự hiện diện của đối tượng (WHAT). Nếu từ chối, đầu ra được đặt bằng 0.

## Dataset
- FSC147 TEST: 1190 positive pairs, 3 exemplars.
- Hard negatives: 111 absent pairs, 96 ảnh FSC147, GT=0.
- Unified: 1301 pairs.

## Kết quả
| Dataset | CountGD++ MAE | Joint V3.2+A3 MAE |
|---|---:|---:|
| FSC147 TEST | 11.6504 | 10.1488 |
| Absent111 | 62.5225 | 46.2394 |
| Unified | 15.9908 | 13.2280 |

## Giới hạn
111 negatives được hard-mine từ lỗi CountGD++; TEST đã được quan sát. Đây là đánh giá post-hoc, không phải blind benchmark. Positive và absent pairs khác nhau về số exemplar.
