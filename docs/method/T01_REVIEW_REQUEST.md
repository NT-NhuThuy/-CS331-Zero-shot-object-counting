# Yêu cầu chốt T01 trước deadline 20/11

Owner: Lộc. Bằng chứng đã có: R10 real GPU FP32 32/32, two-pass pseudo3; CPU raw replay 32/32; 38 candidates, 64 calls, source/checkpoint/protocol SHA đầy đủ.

**Xin trưởng nhóm xác nhận bằng văn bản:**

1. **G0 protocol:** có chấp nhận `CGD_OFFICIAL_QUERY_ONLY_PSEUDO3_V1` — FP32, 0→pseudo3→pass2, `box_threshold=0.23` chỉ trong profile này, `text_threshold=0`, caption `{query} .`, resize 800/1333, seed42, checkpoint `b89395d7...`, source `2ecbc433...` hay phải thực thi official AMP profile khác?
2. **Reference/labels được duyệt:** cung cấp đúng 32 ID (hoặc roster G0 độc lập đã duyệt), SHA ảnh/query, strata present/verified_absent có provenance review, và raw GPU predictions theo cùng protocol (native indices, original float-pixel boxes, scores). V14 archived AMP có count-only 31/32; không đáp ứng strict parity R10 FP32.
3. **Quy tắc parity pre-registered:** coverage 100%, count/native indices exact, box atol 1e-4 pixel, score atol 1e-6. Có cần tiêu chuẩn strict hơn không? Chốt **trước khi chạy/reference mở**.

Tài liệu và code đã chuẩn bị để chạy comparison, nhưng không tự tạo chữ ký hoặc chứng nhận parity. Nếu chưa duyệt/thiếu reference, T01 nộp trạng thái `HOLD_WITH_REASON`, không đánh dấu DONE.
