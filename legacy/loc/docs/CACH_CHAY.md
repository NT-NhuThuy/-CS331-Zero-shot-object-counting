# Hướng dẫn chạy (Kaggle)

1. Thêm FSC-147 images/annotations vào Kaggle Input. Không có ảnh FSC-147 trong repository này.
2. Chạy các notebook theo trình tự phù hợp trong `notebooks/`. Các phiên bản đầu dùng cache V4/V5/V6, không phải tất cả notebook có thể khởi động độc lập chỉ với ảnh.
3. Đối với V7.5, xem `notebooks/04_validation/`. Mỗi bộ validation có manifest và cấu hình frozen riêng. Không trộn kết quả lần 1 và lần 2 khi resume checkpoint.
4. Đối với fresh negative safety, dùng `notebooks/05_fresh_negative/V7_5_FreshNegative_Safety_126.ipynb` và `data/fresh_negative_126_absent.csv`, hoặc input bundle được tạo theo notebook. Chỉ dùng nhãn ABSENT; 34 PRESENT không phải negative.
5. Lưu output từ Kaggle và đặt CSV/JSON trong `results/` với tên phiên bản rõ ràng, không ghi đè dữ liệu đo cũ.
6. Có thể dùng `python tools/verify_results.py` để kiểm tra số liệu quan trọng từ kết quả kèm theo.

**Tương thích môi trường:** Notebook nghiên cứu từng gặp lỗi Transformers/Kaggle về `tensorflow_text`, `smart_apply` và API `additional_chat_templates`. File V7.4 đưa lên là bản đã thêm hướng xử lý lỗi. Chạy smoke test trước khi chạy dài, và cố định môi trường/phiên bản trong một thí nghiệm.
