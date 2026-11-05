# Đưa thư mục Lộc lên GitHub của nhóm

Repository: https://github.com/NT-NhuThuy/-CS331-Zero-shot-object-counting

Tài khoản GitHub đã kết nối tại thời điểm chuẩn bị gói: `coutM1nh4nh` có quyền READ nhưng không có WRITE. Do đó thư mục này được đóng gói để tải lên bằng tài khoản có quyền ghi hoặc sau khi chủ repository mời collaborator.

**Cách đơn giản với Git:**

```bash
git clone https://github.com/NT-NhuThuy/-CS331-Zero-shot-object-counting.git
cd -- -CS331-Zero-shot-object-counting
# Giải nén file ZIP, đặt thư mục Lộc/ ngay bên cạnh README.md và Thu/
git checkout -b loc/research-progress-v7-v75
git add 'Lộc/'
git commit -m 'research: add Loc notebooks and experiment results (V4-V7.5)'
git push -u origin loc/research-progress-v7-v75
```

Sau đó tạo Pull Request từ nhánh trên vào `main` để nhóm review. Nếu bạn không có quyền push repo của nhóm, hãy xin chủ repo cấp Write, hoặc fork repo rồi tạo PR từ fork. Không push Kaggle/Hugging Face API tokens, checkpoint GPU nặng, mô hình weights hay ảnh FSC-147 gốc.
