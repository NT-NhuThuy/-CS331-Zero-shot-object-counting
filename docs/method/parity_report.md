# T01 / G0 — Parity Report (19/11/2026)

**Owner:** Lộc · **Deadline:** 20/11/2026
**Nghiệm thu:** **HOLD — NOT DONE**. Không tự nâng thành `PARITY_PASS`.

## 1. Protocol candidate / source lock

- Adapter: `CGD_OFFICIAL_QUERY_ONLY_PSEUDO3_V1`, text-only, không exemplar thủ công.
- Two forward passes: pass 1 không exemplar; lấy tối đa 3 pseudo-exemplars nội sinh; pass 2 dùng pseudo-exemplars đó.
- Checkpoint SHA-256: `b89395d79711c9b9f13f3ef80cd2e528386fb1d70453f99443be4eccab33592f`.
- Source revision: `2ecbc4331d6730d017f52a4a97bbcbae7d5d714a`, source Git blob IDs trong manifest.
- Protocol SHA-256: `121786020bead2e87d58e9dd6896d022ba2068d58feecfbbc4026113b57f497e`.
- Caption `{query} .`, resize short/max 800/1333; initial exemplar 0, pseudo3, `box_threshold=0.23` **CHỈ thuộc profile R10**, text threshold 0.0, no NMS, seed 42, FP32, no crop, no TT-Norm.
- CUDA 12.8 / Torch 2.9.1+cu128 / T4; checkpoint, BERT files, extension and runtime frozen in `configs/method/protocol_manifest.json`.
- Đây là protocol candidate đã thực thi R10; chưa phải protocol G0 được nhóm duyệt.

## 2. Thực nghiệm Kaggle R10

- **32/32 exposed diagnostic image–query pairs** chạy thành công, **64 detector calls** (hai lượt trên mỗi cặp).
- 32 raw NPZ (first/second boxes/probabilities/input IDs), 32 trace records, 38 candidates, stable native indices/IDs và boxes original pixels; tổng inference latency trong trace 59.024 s.
- 24/32 cặp có 0 candidate; 8/32 cặp có candidates. Đây là predictions, **không phải accuracy**.
- SHA-256 từng raw NPZ, candidate/trace coverage, immutable protocol khóa đúng.
- **CPU raw replay** trên 32 raw NPZ: 64 stage canonical hashes, 32 count, 38 candidate IDs/indices, scores/boxes **PASS**. Đây là phép replay parser, **không phải external G0 parity**. Xem `parity_logs/r10_internal_parser_replay.json`.

## 3. Đối chiếu V14 lưu trữ — diagnostic, KHÔNG phải parity PASS

Từ `V14_P02_parity_bundle.tar.gz`, đối chiếu *sau khi đã freeze R10*: đúng **32/32 ảnh khớp từng SHA-256 byte** và query/caption. Strata nằm ở `pool/evaluator_manifest.json` của V14, tách khỏi inference R10.

| Stratum theo V14 archive | Cặp | Count khớp R10 vs V14 | Ghi chú |
|---|---:|---:|---|
| `verified_absent` | 17 | **17/17** | V14 label chưa chứng minh double-review của G0 |
| `present` | 15 | **14/15** | Lệch 1 count |
| **Tổng** | **32** | **31/32** | **Không đạt equality 100%** |

- Mismatch duy nhất: `image_id=113`, query **`cone`**, R10 FP32 = **18**, V14 AMP = **19**.
- V14 dùng AMP=true, R10 FP32: operating point chưa match.
- V14 chỉ có `predictions.json` boxes đã hậu xử lý (tọa độ nguyên); thiếu raw native indices/second-pass float boxes và binding runtime/source dirty V14. Không thể kết luận strict box/score/index parity.
- Các trường hợp present count=0 ở cả hai bản **không chứng minh phát hiện đúng**, chỉ count khớp. Không tune threshold .23 sau khi thấy mismatch.
- Phép đối chiếu này là exploratory, KHÔNG được dùng làm kết quả final hay phê duyệt T01.

## 4. DoD đối chiếu đúng Excel

| Tiêu chí | Evidence | Status |
|---|---|---|
| Checkpoint/prompt/pseudo/preproc/parser/threshold/source SHA | `protocol_manifest.json`, R10 run | **CANDIDATE FROZEN / PENDING TEAM APPROVAL** |
| GPU two-pass pseudo3 thực tế | R10 32/32 raw/provenance | **PASS (diagnostic)** |
| Raw parser reproducibility | 32/32, 38/38 | **PASS (internal only)** |
| Present matched independent parity | Chưa có reference đúng profile FP32 và signed verification | **HOLD / NOT_MEASURED** |
| Absent matched independent parity | Chưa có reference đúng profile FP32 và signed verification | **HOLD / NOT_MEASURED** |
| Historical V14 parity | AMP vs FP32, 31/32 count match, thiếu native tensors | **HOLD** |
| Git PR → `research/unified-integration-verified` | Package local, chưa commit/push | **NOT SUBMITTED** |

## 5. Để đổi HOLD thành DONE đúng quy định

1. Trưởng nhóm phê duyệt **operating point cụ thể** (FP32 R10 hoặc profile khác). Nếu chọn profile khác phải tạo run mới có version, không thay đổi lock R10.
2. Chốt **roster present + verified-absent**, provenance riêng, approval trước khi mở reference; không dùng `FINAL_SEALED` để debug.
3. Chạy **independent GPU reference** cùng checksum ảnh/query, checkpoint, preprocessing, pseudo selector, precision, parser và protocol. Export candidate raw/native index, original float-pixel boxes, scores; không chỉ count integer.
4. Dùng `src/countlab/integrations/t01_parity.py` so sánh 100% cặp, theo native index (không IoU hay row order), `box_atol=1e-4` pixel, `score_atol=1e-6`; không chỉnh tolerance hậu nghiệm. Lưu full failure cases.
5. Sau khi có bảng PASS present/absent + phê duyệt nhóm, cập nhật trạng thái G0 bằng một bản manifest mới có version và PR vào branch tích hợp.

**Cảnh báo nghiên cứu:** không dùng chính bản sao R10 làm `independent reference`; test fixtures chỉ thử logic. R10 không đọc GT, T04 và các task sau chưa được đo.
