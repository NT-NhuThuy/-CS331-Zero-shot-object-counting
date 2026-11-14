# FQIE implementation contract v1

> **Bidirectional addendum (2026-11-16):** the current method verifies frozen baseline proposals and rescue candidates. `UNKNOWN` retains the baseline estimate. A baseline proposal may be removed only when semantic=`NO` and whole-instance=`NO`; a rescue may be added only after semantic=`YES`, whole-instance=`YES`, and no baseline match at IoU 0.5. GT remains forbidden before reconciliation and artifact freeze.

Contract này phải được freeze trước khi triển khai Safe-IIC hoặc tạo test benchmark.

## Baseline contract

- checkpoint: frozen CountGD++ checkpoint, hash bắt buộc;
- query-only inference, không đọc GT;
- `num_exemplars=0`, `num_pseudo_exemplars=3` là operating point lịch sử V7/V8;
- mọi thay đổi config tạo experiment version mới;
- input image/query manifest và output prediction đều hash-lock.

## Evidence states

Mỗi candidate có các trạng thái độc lập:

- `semantic`: `YES | NO | UNKNOWN`;
- `whole_instance`: `YES | NO | UNKNOWN`;
- `identity`: `SAME | DIFFERENT | UNKNOWN`;
- `decision`: `ACCEPTED | REJECTED | UNKNOWN`.

Không được suy diễn `UNKNOWN → NO` hoặc `UNKNOWN → ACCEPTED`.

## Operational definitions

- `zero_evidence`: frozen baseline có đúng 0 prediction sau cùng một post-processing policy;
- `low_evidence`: chỉ được dùng nếu threshold được đăng ký trước trên validation split; không dùng tùy ý trên test;
- `semantic_confuser`: region được annotation là không thỏa query nhưng có category/appearance gần target;
- `part_whole`: region chỉ là bộ phận/texture/background của target whole object;
- `absent`: source image không có instance thỏa query specification;
- `ambiguous`: evidence không đủ để xác định accepted/rejected hoặc same/different.

## Candidate contract

Mỗi candidate phải lưu:

- `candidate_id`, `query_id`;
- `box_xyxy` trong original-image coordinates;
- `source_image_id`, `view_id`, `view_provenance`;
- `proposal_score`, `source_model`;
- `semantic`, `whole_instance`, `identity` states;
- `evidence_refs` và artifact hashes;
- `decision`.

## Count contract

```json
{
  "count": 3,
  "count_lower_bound": 3,
  "count_upper_bound": 4,
  "count_status": "partial",
  "accepted_instances": [],
  "rejected_regions": [],
  "unknown_regions": [],
  "evidence_trace": [],
  "query_abstained": false
}
```

`count_status` là một trong `exact | partial | abstained`.

- `exact`: lower = upper = count và không còn unresolved component;
- `partial`: count = lower bound, còn unresolved candidates/components;
- `abstained`: hệ thống từ chối scalar count vì evidence không đủ; lower/upper vẫn phải audit được.

Evidence trace là structured provenance, không phải free-form explanation.
