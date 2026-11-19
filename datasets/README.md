# UNICOUNT datasets — provenance-controlled metadata

This directory is the **single entrypoint for dataset metadata and split policy**, not a location to upload raw FSC147/LVIS images, checkpoints or query-time GT.

## Datasets in scope

| ID | Historical status | Pairs | Role |
|---|---|---:|---|
| `fsc147_extended_1301` | EXPOSED_REFERENCE | 1,301 = 1,190 present + 111 absent | Diagnose errors and reproduce historical experiments; **not a new independent test** |
| `loc_fresh160` | EXPOSED_REFERENCE / review pending | 160 = 126 one-reviewer absent + 34 one-reviewer present | Audit labeling and lineage before any proposed reuse |
| `loc_fresh126` | Same source as subset of fresh160 | 126 absent | **Not an independent evaluation set** |

Actual versioned files are currently preserved under `legacy/thu/...` and `legacy/loc/...`; the registry records **references** so no dataset is copied and accidentally counted twice. The historical FSC cohort has 96 distinct images among the 111 absent image-query pairs. Labels `absent` and `count=0` apply to a **specific `(source_image, target_query)` pair**, never to the entire image.

## Important policy

1. Keep source images, model files and raw derived tensors outside Git in `datasets/_local/` or a private storage volume. Preserve SHA256, original acquisition/version and download instructions in tracked metadata.
2. `datasets/registry.json` lists accessible historical artifacts; it does not certify annotation correctness or independent splits.
3. Dataset labels/GT live in separate evaluation inputs and **must never enter** `countlab_candidate_pool_v1` inference JSON.
4. Historical FSC147 TEST/absent111 and previously observed V14/PrACo cohorts are exposed. P03 G0 is HOLD pending approved TRAIN/DEV/CAL/FINAL_SEALED source-disjoint allocations.
5. Counts/presence for negative pairs must be confirmed with sufficient annotation or independent human review. `source_class != query` is **not** evidence of absence. The legacy `annotation_absent111_empty.json` is a target-query-specific legacy artifact, **not** proof an image contains no objects.
6. Avoid interpreting aggregate MAE for 1,301 pairs as generalization. The 111 absent pairs were hard-mined from CountGD++ failures; original present/absent exemplar protocols also differ (3 vs 0).

## New pair format

Use the `schemas/pair_record_v2.schema.json` contract for **dataset storage/evaluation only**. `source_group_id` groups re-encodings/crops and matches so image/query variants never split across TRAIN/DEV/CAL/FINAL_SEALED. Inferential input still uses the existing countlab schema, without labels.

See `splits/README.md` for the approval process.
