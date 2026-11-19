# T05 — Thư: exact-bound V4/A3 diagnostic features on Lộc's frozen CountGD++ pool

**Status:** Code/contract implementation, CPU synthetic verification. Real T04 -> V4/A3 alignment **BLOCKED** pending independently supported original-run crosswalk and feature NPZ. **G0 remains HOLD.**

## Motivation

Lộc's official query-only pseudo3 pool uses `cgd-pseudo3:<hash>` IDs, `native_query_index` and original-image coordinates. Historic Thư V4/A3 `.npz` is keyed by `(node_graph_id, raw_query_index)` **from a different research artifact and potentially a different detector operating point**. `native_query_index` is **not** the same thing as V4/A3 `raw_query_index`. Joining by image filename, IoU, ordinal position or coincident integers would invent evidence. It is not acceptable for T05/T12.

The existing `countlab.integrations.thu_features.attach_features` handles a *trusted exact-key* join but did not itself establish that candidate IDs originated in the same image/query/run as those feature rows. This T05 wrapper adds the missing scope locks and validates the exact original-pixel candidate identity.

## Stage 1: emit an exact crosswalk request from a real frozen pool

On branch `feat/thu-t05` (based on `research/unified-integration-verified` at/after `12ba852`):

```powershell
python -m countlab.integrations.thu_t05_exact --mode request --pool fixtures/countgd_official_gpu_smoke.json --out-dir reports/thu_t05/request_smoke_001
```

It creates `binding_request.json`, with **exact** candidate IDs, original-pixel boxes, `native_query_index`, image/query/protocol identifiers and canonical pool SHA256. The two *frozen feature node keys are deliberately null*. The result prints `BLOCKED_CROSSWALK_NOT_PROVIDED`, **not an error of the CountGD++ detector**. On the historical recorded GPU smoke the pool contains 16 proposals, but this command **does not rerun GPU**.

The canonical SHA is over parsed JSON values and is portable across Windows LF/CRLF newlines. Raw SHA256 is still computed on each evidence/NPZ/run file's actual bytes; never change a stored SHA to make a test pass.

## Stage 2: C2/C1 handoff: produce the real crosswalk, not an estimated map

For each candidate, C2/C1 must show **documented source-run provenance** that maps a particular original `candidate_id` onto one exact frozen feature `node_graph_id,raw_query_index` in the supplied NPZ. The following three actual files are required:

1. The frozen NPZ containing **exactly** `node_graph_id`, `raw_query_index`, `v4_contribution`, `a3_score`. This is a *diagnostic* representation, not the Joint V3.2 scalar predictor or a retrained live V4/A3 model.
2. A source run manifest JSON with `image_sha256`, `query_text`, `protocol_id` matching the frozen pool, plus independently reviewable checkpoint, code revision and data origin information. Lộc's `fixtures/countgd_official_gpu_smoke_summary.json` contains these three scope fields for that recorded smoke, but **does not establish a V4/A3 crosswalk**.
3. A crosswalk evidence JSON (schema `thu_t05_crosswalk_evidence_v1`) from a reproducible feature-exporter run. This JSON must contain `source`, `candidate_identity_sha256`, `feature_npz_sha256`, and `candidate_rows`, where those values exactly match a new T05 binding JSON. The *provenance narrative and actual raw mapping construction* must be independently auditable; a syntactically correct crosswalk file does not prove two different detector runs are equivalent.

Prepare a `thu_t05_exact_binding_v1` binding JSON with these fields:

```json
{
  "schema": "thu_t05_exact_binding_v1",
  "source_cohort_status": "EXPOSED_REFERENCE",
  "source": "<copy the source object from binding_request.json>",
  "candidate_pool_sha256": "<copy from binding_request.json>",
  "candidate_identity_sha256": "<copy from binding_request.json>",
  "feature_npz_sha256": "<actual sha256 of NPZ bytes>",
  "crosswalk_evidence_sha256": "<actual sha256 of crosswalk evidence JSON bytes>",
  "source_run_manifest_sha256": "<actual sha256 of source run manifest bytes>",
  "candidate_rows": "<copy ALL candidate_rows from request; fill node_graph_id and raw_query_index ONLY from verified original-run export>"
}
```

The angle-bracket strings above are **descriptive placeholders**, not a directly valid binding. In real JSON the `source` object and `candidate_rows` array must be actual objects/arrays. All rows must cover exactly the frozen candidate set with original IDs, roles, view IDs, `native_query_index`, and `box_xyxy` unchanged; all mapped frozen node keys must be distinct. Never fill `raw_query_index` from `native_query_index` by assumption.

**Do not bind FSC147 legacy V4/A3 NPZ to Lộc's pseudo3 output if there is no proof of the exact same feature-producing detector source/run.** If it cannot be proven, the correct deliverable is a new GT-free V4/A3 feature export on Lộc's frozen original candidate pool — not relabeling old features.

## Stage 3: exact technical join (only if the mapping really exists)

```powershell
python -m countlab.integrations.thu_t05_exact --mode bind --pool path/to/frozen_pool.json --binding path/to/thu_t05_binding.json --feature-npz path/to/frozen_v4_a3_nodes.npz --crosswalk-evidence path/to/crosswalk_evidence.json --run-manifest path/to/run_manifest.json --out-dir reports/thu_t05/bound_001
```

Outputs are create-only:

- `frozen_candidates_t05.json`: original `baseline`, `rescue`, `identity_edges`, confidence, evidence and candidate IDs unchanged; adds `metadata.thu_frozen_features.per_candidate` and `metadata.thu_t05_exact_binding`.
- `t05_binding_report.json`: file and mapping hashes, candidate count, technical join status, and explicit `G0=HOLD`.

No `risk_hint` is generated, no verification budget or selected calls are changed, and no count estimate is modified. **T08** must later train/calibrate a GT-blind risk controller on approved CAL/TRAIN split and freeze policy before tuning; **T12** remains an offline paired evaluation and must not use exposed smoke as clean DEV.

## Tests

```powershell
python -m pytest -q tests/test_thu_t05_exact.py
python -m pytest -q tests/
```

The 27 T05 tests cover exact join, immutable original proposals, source/query/box/candidate namespace checks, NPZ and evidence hashes, duplicate/missing node mappings, GT leaking, run-manifest scope, zero-pool request, create-only CLI and duplicate JSON keys. They use a **synthetic** NPZ and do not establish real GPU/CountGD++↔V4/A3 compatibility.

## Open obligations and ownership

- **C2 / Lộc:** freeze complete original candidate pool + image SHA, raw run/checkpoint/protocol manifest. If a V4/A3 feature producer runs on the exact same candidate IDs, provide actual raw index/feature-export provenance and evidence.
- **C1 / Thư:** prove the source run matches, produce or audit original-run feature export, bind exact, and later provide calibrated query/presence risk (including zero-candidate queries) only with approved CAL/TRAIN evidence.
- **C3 / Thúy:** consume only original candidate namespace and recorded evidence references; keep 8 exposed queries and smoke in diagnostic scope; do not transfer them to q31/R10 or clean DEV.

**Never interpret a PASS_TECHNICAL_DIAGNOSTIC_ONLY as measured AZA, Present MAE, FZR+ improvement, runtime saving, model training or clean DEV certification.**
