# UNICOUNT v2 — T01 G0 submission bundle (19/11/2026)

**Kết quả hiện tại: T01 technical work mostly complete, strict G0 parity HOLD; NOT DONE.**

Đây là bộ artifact và source review-ready, tổng hợp từ R1 hardening snapshot và R10 Kaggle. Chưa commit/push vào GitHub. Không thay user approvals hoặc giả lập GPU inference mới.

## Files chính

- `configs/method/protocol_manifest.json`: manifest R10 chính xác, **candidate**, chưa team approval.
- `src/countlab/integrations/countgd_official.py`: official bridge đã được hash-lock, content SHA khớp R10; các file `src/countlab/` khác là R1 scaffold cần cho kiểm thử độc lập. Chỉ đưa thay đổi so với base Git thực tế vào PR.
- `src/countlab/integrations/t01_parity.py`: parity gate mới, kiểm tra đúng 32 SHA+query, native indices, score/boxes và strata riêng. Không chấp nhận V14 count-only như matched reference.
- `docs/method/parity_report.md`: báo cáo nghiệm thu trung thực, bao gồm mismatch V14.
- `docs/method/parity_logs/`: R10 GPU evidence, raw parser replay, V14 exploratory, SHA locks.
- `tools/t01/audit_r10_raw_cpu_replay.py`: CPU raw→parser replay PASS (không dùng GT).
- `tools/t01/reproduce_v14_count_diagnostic.py`: tái tạo 31/32 count-only so V14 AMP (GT labels đọc sau freeze).
- `tests/test_t01_parity.py`: 16 unit tests, positive/negative fixtures **simulated only**.
- `tests/test_countgd_official.py`: 34 tests adapter/fixtures từ R1.

## Chạy test CPU từ thư mục gốc repo (có Python + pytest, NumPy)

```bash
PYTHONPATH=src python -m pytest -q tests/test_countgd_official.py tests/test_t01_parity.py
PYTHONPATH=src python tools/t01/audit_r10_raw_cpu_replay.py \
    --r10-zip docs/method/parity_logs/UNICOUNT_T01_R10_GPU_EVIDENCE.zip \
    --image-dimensions docs/method/parity_logs/r10_image_dimensions_verified.json \
    --out /tmp/t01_r10_raw_replay_new.json
```

## Khi được leader cung cấp independent matching reference + strata + approval

`reference_predictions.jsonl` có **32 rows**: `image_sha256`, `query`, `candidates` chứa `native_query_index`, `box_xyxy`, `confidence` (không chứa GT). `reference_manifest.json`: `protocol_sha256`, `source_type=INDEPENDENT_GPU_REFERENCE_RAW_OR_NATIVE`, `source_commit`, `run_id` (riêng), `reference_predictions_sha256`. `strata.jsonl` riêng chứa SHA/query, `status=present|verified_absent`, `verification_source` có thật. `team_approval.json` chứa bốn bool `protocol_approved`, `roster_approved`, `reference_approved`, `tolerances_preapproved`, người duyệt và timestamp UTC thật. **Không có các file này trong bundle, vì chưa có bằng chứng hợp lệ; không tự dựng bằng giá trị demo.**

```bash
PYTHONPATH=src python src/countlab/integrations/t01_parity.py \
  --r10-zip docs/method/parity_logs/UNICOUNT_T01_R10_GPU_EVIDENCE.zip \
  --reference-jsonl /path/to/reference_predictions.jsonl \
  --reference-manifest /path/to/reference_manifest.json \
  --strata-jsonl /path/to/strata.jsonl \
  --team-approval /path/to/team_approval.json \
  --out /tmp/t01_matched_parity_new.json
```

Nếu thiếu reference/approval, chạy riêng `--r10-zip ... --out /tmp/...` sẽ trả trạng thái HOLD, exit 2. **Không nới `atol` sau khi biết mismatch.** Human approval và provenance tính độc lập vẫn cần kiểm tra bằng người; chương trình không thể chứng thực ai đã ký.

## Tích hợp Git (Codex chỉ làm Git)

Tải/copy các **file mới và phiên bản không trùng** vào repo branch của bạn. Trước khi commit, `git diff --check`, `git diff -- src/countlab/integrations/countgd_official.py` để xác nhận không vô tình ghi đè adapter đang có, rồi chạy tests. Không tự commit binary checkpoint/BERT hoặc `V14_P02_parity_bundle.tar.gz` (74 MB). `UNICOUNT_T01_R10_GPU_EVIDENCE.zip` ~1.8 MB chỉ gồm trace/NPZ và không chứa weights.

Tạo **Draft PR** nhằm review T01 HOLD, không gọi là DONE khi G0 chưa PASS. Target branch `research/unified-integration-verified`. Những commit/PR thật chỉ được xác nhận sau khi Git báo thành công.
