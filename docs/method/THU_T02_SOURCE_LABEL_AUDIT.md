# Thư T02 — Source / Label / Split Eligibility Audit

**Owner:** Thư. **Scope:** G0/G2. **Status of code:** ready for CPU execution; no new independent dataset is certified.

## Purpose

Extend the already-merged P03 source-identity audit without rerunning the old research. Produce one historical evaluation-only ledger, one deterministic *EXPOSED_REFERENCE* pilot, and an explicit HOLD for missing clean data. This tool does **not** perform any image inference or verification of whether a target is truly absent.

## Inputs (existing files from the repository)

- `legacy/thu/generated/fsc147_presence_v4/test_pairs_1301.csv` — 1,190 positive and 111 historical reported negatives.
- `legacy/loc/data/verified_hard_negatives_111.csv` — must match **the same** 111 by pair ID and image-query.
- `legacy/loc/data/fresh_negative_160_manual_review.csv` — single-review diagnostic cohort.
- `legacy/loc/data/fresh_negative_126_absent.csv` — exact 126-record absent subset, **not** independent.
- Optional source-hash P03 ledger and its corresponding `p03_source_identity_report.json` from the user's research machine. These two files must be supplied together; their SHA and integrity status are checked. The large ledger is not checked into Git.
- Optional new proposed clean split roster; this receives metadata/leakage checks, but does not automatically receive approval.

## Run

From the repo root (Python environment with `pip install -e '.[test]'`):

```bash
python -m countlab.thu_t02 \
  --fsc legacy/thu/generated/fsc147_presence_v4/test_pairs_1301.csv \
  --loc111 legacy/loc/data/verified_hard_negatives_111.csv \
  --fresh160 legacy/loc/data/fresh_negative_160_manual_review.csv \
  --fresh126 legacy/loc/data/fresh_negative_126_absent.csv \
  --out-dir reports/thu_t02/run_001

python -m pytest -q tests/test_thu_t02.py
```

For **verified byte-level grouping** on the user's dataset machine, append:

```bash
  --source-ledger /path/to/p03_image_byte_hashes.csv \
  --source-identity-report /path/to/p03_source_identity_report.json
```

Do not put two separate shell blocks together accidentally; append these options to the first command before running. The output directory must not already exist.

## Outputs and interpretation

| Output | Purpose |
|---|---|
| `t02_label_ledger_GT_ONLY.csv` | Dataset audit and GT/label provenance. **Never** pass this to CountGD++/Qwen/solver. |
| `t02_pilot_labels_GT_ONLY.csv` | Labels and mapping from an **opaque** query ID to the historical pair ID, only used by a separate evaluator after freezing predictions. |
| `t02_pilot_inference_inputs_GT_FREE.json` | Opaque, label-neutral query IDs, image path, query and source identity; does not contain presence/count labels or historical `POS__`/`HARD__` IDs. |
| `t02_audit_report.json` | SHA256 hashes, overlap proof, label/provenance counts, and HOLD blockers. |

Historical absent111 are **HISTORICAL_REPORTED_ABSENT**, not independently re-reviewed. Fresh160 entries retain **SINGLE_REVIEW_PRESENT/ABSENT** or **UNVERIFIED**; neither becomes sealed data. Fresh160 lacks a trusted original-image identity mapping here and remains ineligible for pilot grouping until source verification.

The pilot uses a deterministic selection of up to 16 historical absent and 16 historical present pairs, with unique *declared* FSC image groups or attested P03 byte hashes. It is a technical integration pilot, **not a valid unbiased evaluation**. Same-image present/absent records are never simultaneously selected.

## Conditions still blocking G0/G2

- Need independently auditable source lineage for new TRAIN/DEV/CAL/FINAL_SEALED, including re-encoded/cropped image families.
- Need verified absent query annotations for the new cohort; historical single review and inferred `source_class != target` do **not** prove absence.
- Need clean split registry and disjointness checks to pass on the actual data.
- Lộc T01/T04 must freeze CountGD++ official pseudo3 and provide real candidate pool. This T02 audit does not satisfy these tasks.

**Expected:** `technical_status=PASS` may coexist with `g0_data_decision=HOLD` and `g2_decision=HOLD`. Do not rename HOLD to GO merely because the script or tests pass.

### Windows PowerShell

Sau khi activate môi trường Python trong thư mục repo:

```powershell
python -m pip install -e ".[test]"
python -m pytest -q tests/test_thu_t02.py
python -m countlab.thu_t02 --fsc legacy/thu/generated/fsc147_presence_v4/test_pairs_1301.csv --loc111 legacy/loc/data/verified_hard_negatives_111.csv --fresh160 legacy/loc/data/fresh_negative_160_manual_review.csv --fresh126 legacy/loc/data/fresh_negative_126_absent.csv --out-dir reports/thu_t02/run_001
```

Có thể chạy trước mà chưa có image checkpoint vì T02 chỉ đọc CSV và metadata. Nếu `reports/thu_t02/run_001` đã tồn tại, chọn thư mục `run_002` mới để giữ provenance. Trên server có file ledger P03, thêm cặp `--source-ledger` và `--source-identity-report` để nâng source hash sang nhóm byte-exact được chứng thực.
