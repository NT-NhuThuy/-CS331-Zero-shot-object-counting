# P01-D — Historical Experiment Data-Lineage Recovery

Ngày audit: 2026-11-17  
Mục tiêu: kiểm kê provenance lịch sử cho `E00–E10`, `V13`, `V14` mà không suy
diễn split/source-disjoint từ filename, numeric image ID hoặc query text.

## Kết quả

**P01-D inventory PASS; G0 source-disjoint vẫn BLOCKED.**

- 13 experiment IDs được kiểm kê: E00–E10, V13, V14.
- E00–E10 và V13: không tìm thấy manifest, task index, per-query output hoặc
  source-image lineage trong scope `ZOC/`, `ZOC_research/` (nếu có),
  `UNICOUNT_PRIVATE/`.
- V14: phục hồi được 54 per-query entries từ original bidirectional output,
  imported candidate manifests và imported per-query results.
- Verified source-image entries: **0**.
- Missing-provenance entries: **54** (toàn bộ V14): có image ID/query/artifact
  evidence, nhưng không có source image path/hash hoặc bytes task-index gốc để
  chứng minh source identity.

Manifest chia sẻ cho Thư:

[`manifests/historical_experiment_lineage_audit_v1.json`](../../manifests/historical_experiment_lineage_audit_v1.json)

Nó không nhúng GT, ảnh private, image bytes, hay checkpoint. Các `artifact_path`
đều là đường dẫn logic tương đối workspace; source image path chỉ ghi nếu có
bằng chứng, hiện không có entry nào.

## Phạm vi và phương pháp

Audit quét read-only các root được giao. Với V14, nó dùng:

- `UNICOUNT_PRIVATE/outputs/v14_original_cpu_repro_20261114/bidirectional_output.json`
- `UNICOUNT_PRIVATE/outputs/v14_unicount_import_20261114/manifests/`
- `UNICOUNT_PRIVATE/outputs/v14_unicount_import_20261114/results/`

Mỗi dòng lineage có schema:

```text
experiment_id, run_id, image_id, image_sha256, query_id, query_text,
dataset, source_group, artifact_path, evidence_status
```

Khi raw source image file tồn tại, auditor phải hash bytes (`SHA-256`) và điền
`image_sha256`. Không có hash thì `source_group` phải là `null`; không được gán
`source-disjoint`.

## V14 evidence còn lại

V14 reports lưu query identity, image ID và query text. Ví dụ query `1` ghi
`image_id=210708`, `query_text="elephant ."`, đồng thời có original output,
imported manifest và imported result. Tuy nhiên:

- task-index chỉ còn claimed SHA
  `d35b92f975efbf4c3077196b35575aef9aa47e8a8c57b7992acc40e27592b5c6`;
  bytes/index path không còn;
- unary decision source chỉ còn claimed SHA
  `3ee584aa07ea3aeed69a705ea2e117e3a0d55ca64bef69825fef1512af4b48ca`;
  bytes không còn;
- không có source image path/hash trong V14 retained manifests/results.

Do đó status đúng là `ARTIFACT_QUERY_VERIFIED_SOURCE_UNVERIFIED`, không phải
`source-disjoint`, không phải "same dataset" theo nghĩa có thể audit.

## Cross-check với P03 của Thư

P03 cần cung cấp bảng gồm ít nhất:

```text
image_sha256, source_group, dataset, source_path_or_stable_source_id
```

Quy tắc join duy nhất hợp lệ là `image_sha256`. Join theo image filename, image
numeric ID, class/query text, output count, hoặc artifact directory đều bị cấm.
Vì current manifest có 0 `image_sha256`, hiện không có join P03 nào được phép;
không có kết luận positive/negative về overlap hay source-disjoint.

Khi lịch sử task-index/source paths được khôi phục, chạy lại:

```bash
cd /path/to/workspace-parent
PYTHONPATH=UNICOUNT/src python3 -m countlab.integrations.historical_experiment_lineage_audit \
  --workspace "$PWD" --private "$PWD/UNICOUNT_PRIVATE" \
  --out UNICOUNT/manifests/historical_experiment_lineage_audit_v2.json
```

Lưu ý: v1 chỉ cố ý phục hồi file set hiện còn; version sau chỉ được thêm hash
qua source path/bytes được chứng minh, không sửa tay `source_group`.

## PASS criterion mapping

| Criterion | Status |
|---|---|
| Mọi experiment ID được kiểm kê | PASS — 13/13 có record minh bạch |
| Mỗi entry có evidence status | PASS — V14 54 rows; IDs không tìm thấy có experiment status |
| Manifest chia sẻ được cho P03 | PASS — JSON GT-free, không private image/checkpoint |
| Source hash cross-check với P03 | BLOCKED — 0 source hashes recovered |
| G0 source-disjoint | BLOCKED — lịch sử data/split usage chưa đủ |

P01-D PASS ở đây chỉ là pass cho **inventory/audit discipline**. Nó không là
G0 PASS và không cho phép dùng V14 hay bất kỳ experiment historical nào làm
bằng chứng independent split/source-disjoint.

## Reproducibility and review limitation (2026-11-17)

The accompanying Python auditor reconstructs **V14** lineage from the specified
retained output/manifests/results. For **E00–E10 and V13**, the `NOT_LOCATED_IN_SCOPED_ROOTS`
status is an inventory statement supplied with this handoff: the published Python
script does **not** independently scan the `ZOC/` and `ZOC_research/` roots for
those experiment IDs. That absence claim requires retained search logs or a
separate auditable inventory before it is treated as independently reproduced.

The 54 V14 records contain legacy image IDs and query labels, **not** verified
source-image identities. Do not join them to P03 using numeric IDs or query text;
there is no legitimate source-hash join with the current evidence.
