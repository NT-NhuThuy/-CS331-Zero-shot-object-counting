# Chạy V7.5 Fresh Hard-Negative Safety với 1 người gán nhãn

## Bộ dữ liệu
- 160 cặp được kiểm tra thủ công bởi một người duy nhất.
- 126 `ABSENT` là tập hard-negative test; ground truth đếm là 0.
- 34 `PRESENT` không dùng để tính negative MAE/FPR.
- Không có reviewer thứ hai. Cột `adjudicated_presence` chỉ là cột tương thích kỹ thuật, sao chép đúng quyết định người đánh giá thứ nhất, KHÔNG phải nhãn đồng thuận.

## Kaggle
1. Import `NCKH_V75_FreshNeg_02_Safety_SINGLE_REVIEWER.ipynb`.
2. Add Input dataset FSC147 ảnh `hihihi5` và `V75_FreshNeg_Single_Reviewer_Inputs.zip`.
3. Bật GPU + Internet. Để nguyên các ngưỡng cố định của V7.5.
4. Nhấn Run All một lần; tải file `V75_FreshNeg_Safety_Results.zip` trong `/kaggle/working`.
5. Gửi file kết quả lại cuộc trò chuyện để kiểm tra FPR, MAE, số false positive phát sinh do tiling.

## Kiểm tra nhanh
Cell 02 phải in:
`SINGLE reviewer ABSENT: 126 | PRESENT excluded: 34 | UNCERTAIN excluded: 0`.
Nếu không đúng, dừng chạy và kiểm tra input. Không dùng bảng template chưa gán nhãn.

## Diễn giải kết quả
Notebook so sánh **CountGD++ baseline** với **nhánh rescue multi-scale V7.5** trên cặp negative, KHÔNG phải phép đo FPR end-to-end của V7 router. Khi viết báo cáo: ghi rõ nhãn từ **single annotator**, không có independent agreement/adjudication. Không dùng kết quả này để chỉnh ngưỡng rồi báo cáo lại như test độc lập.
