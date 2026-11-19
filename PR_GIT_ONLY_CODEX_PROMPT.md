# Chỉ dùng Codex cho thao tác Git (không yêu cầu Codex viết lại T01)

Nguồn artifact đã được ChatGPT tạo và kiểm thử riêng. Kiểm tra checkout local của repo
`NT-NhuThuy/-CS331-Zero-shot-object-counting`, nhánh tích hợp đích `research/unified-integration-verified`.

1. Mở Git clone đang dùng, tạo branch mới `feat/loc-t01-g0-review-20261119` từ base đã fetch/verified; KHÔNG force-push.
2. Bản code trong gói overlay T01 đã được chuẩn bị để COPY vào repo (người dùng/ChatGPT làm bước copy). Codex chỉ kiểm tra Git status/diff:
   `git status --short`, `git diff --check`, `git diff --stat`, `git diff -- src/countlab/integrations/countgd_official.py`.
3. Nếu adapter đang có trong repo khác SHA-256 `b0890cd15a5e46d055ad6680a0a29bdaf2aae9c73e125bb9cbf9892c69a87773`, DỪNG để người dùng review, không tự overwrite.
4. Đừng commit weights/BERT, tar V14 74MB, local venv, dữ liệu PRIVATE. Gói R10 GPU evidence ~1.8MB gồm JSON/NPZ trace không có checkpoint.
5. Khi người dùng kiểm tra và đồng ý với diff/tests, mới commit các file T01, push branch; tạo Draft PR có title `T01/G0: frozen pseudo3 FP32 GPU evidence + HOLD parity review` hướng đến `research/unified-integration-verified`. Mô tả trạng thái HOLD và V14 count-only 31/32.
6. Không thay `protocol_sha256`, threshold, parity tolerance, reviewer approval hay lịch sử Git để biến HOLD thành PASS.

**Chưa có commit/push/PR nào được thực hiện bởi gói này.**
