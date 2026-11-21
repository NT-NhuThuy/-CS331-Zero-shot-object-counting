# T06 G1 — Pilot graph replay, frozen-evidence CPU contract v1

**Owner:** Thúy. **Task:** T06 G1. **Release scope:** additive local software acceptance, not signed real pilot. Strict input/solver from T03 (`t03_contract_v1.py`) and T09 (`t09_delivery_v1.py`) remain unchanged. Nothing in this implementation runs CountGD++/Qwen or reads GT.

## 1. Source and mode

`tools/thuy/t06_pilot_replay_v1.py --inputs ... --config ... --out-dir ...`

JSONL input has one `countlab_t06_pilot_input_row_v1` record per image-query:

| Field | Contract |
|---|---|
| `schema` | `countlab_t06_pilot_input_row_v1` |
| `manifest` | Exact T03 `countlab_candidate_pool_v1`; baseline/rescue boxes use original-image pixel coordinates and globally unique candidate IDs. Strict GT leak checks. |
| `observed_events` | Exact T09 event list: `REGION_SEMANTIC`, `REGION_WHOLE`, `PAIR_IDENTITY` with unique evidence refs. Missing evidence remains UNKNOWN. |
| `feature_join` | `{"status":"UNAVAILABLE"}` OR `{"status":"EXACT_ID_JOIN_DECLARED_UNVERIFIED","feature_source_sha256":"<sha256>","candidate_map":{...}}`. Map MUST cover precisely all IDs; each maps to nonnegative integer `node_graph_id` and `raw_query_index`, without duplicate pair keys. This is a **declared exact-ID check only**. It does not authenticate the NPZ or apply/calibrate V4/A3 scores. |
| `provenance` | `{"mode":"SYNTHETIC_CONTRACT"}` OR `{"mode":"EXTERNAL_HANDOFF_UNVERIFIED","source_image_sha256":"<sha256>","t04_pool_sha256":"<sha256>","t07_events_sha256":"<sha256>"}`. Hash syntax is checked, bytes of external files are not independently available or verified in this runner. |

All objects must be GT-blind. No silent fallback from invalid external data to synthetic mode. If any input row, event, candidate, duplicated image/query, feature ID join or config is invalid, the **entire roster fails before creating outputs**. Existing output directory is refused. Records are never silently omitted.

## 2. Solver and graph semantics

T06 calls existing `run_t09_delivery(manifest, events, fallback_count=manifest.baseline_count, config=solver_config)`. The same T03/T09 solver is not modified or retrained. The graph trace records every baseline/rescue `candidate_id`, `view_id`, original-pixel `box_xyxy`, actually observed unary verdicts and identity refs, positive physical-instance witnesses, blockers, input/solver hashes, and T05 feature join status. Hidden precomputed input labels never enter the observed evidence channel.

`decision=K` and `count>0` are **conditional on the observed candidate pool and correctness of upstream verifier events**, not certified whole-scene counts. If uncertain/conflicting/unobserved, `decision=ABSTAIN`, `count=null`, and `evaluation_fallback_count` remains the original baseline count. `observed_pool_decision=ZERO` only describes the bounded observed candidate set after dual NO, never scene absence. A missing `DIFFERENT` does not imply two physical objects; IoU does not prove SAME or DIFFERENT. No count correction is authorized by T06.

## 3. Files created by CLI

| File | Contents |
|---|---|
| `pilot_predictions.jsonl` | Exactly one schema-checked prediction per input pair: strict decision/count/fallback, blockers, node/instance counts, source hashes and explicit `handoff_status`. |
| `graph_trace.jsonl` | Exactly one full graph/evidence trace per pair; candidate and view identifiers with actual observed evidence only. |
| `pilot_replay_report.md` | Counts, SHA256 locks, non-certification and real-handoff HOLD. |
| `pilot_replay_lock.json` | Input/config/output/report SHA256 and pair count; `real_integration_signed_off=false`. |

The bundled files starting `T06_SYNTHETIC_...` are **illustrative frozen contract samples**, constructed using T09 test cases, and MUST NEVER be presented as the team's official `pilot_predictions.jsonl` or benchmark results.

## 4. Remaining requirements before REAL T06 G1 acceptance

1. **C2 T04/T01:** canonical pseudo3 candidate pool and pilot roster, full source/checkpoint/prompt hashes, source-image ID, stable candidate IDs, original-pixel coordinate transforms, no lost pairs; official pseudo3 parity PASS.
2. **C1 T05:** independently verifiable feature source bytes and exact candidate-ID to `(node_graph_id,raw_query_index)` mapping, real V4/A3 records or explicit unsupported statuses. This wrapper only audits declared mapping, **does not verify actual T05 source or use features for routing**.
3. **C2 T07:** raw observed semantic/whole/pair events with real revision/prompt/source hashes, unknown/timeout handling, no hallucinated DIFFERENT. The T09 event payload has no cost metadata; separate C2 source log must be audited outside it.
4. Run on approved canonical roster, verify hashes against original handoff files, compare expected and produced `(image_id,query_id)` pairs, check actual failures, preserve GT separation. A real cross-team integration signoff requires independent artifacts/CI.
5. **G2/G3/T14** science: do not claim MAE, AZA/FZR+ or matched GPU cost from CPU replay. The synthetic fixtures prove code behavior, not physical truth or predictive quality.

## 5. Reproduce on Windows

From the repository root, with Python/pytest installed:

```powershell
$env:PYTHONPATH = 'src'
py -m pytest -q tests/test_thuy_t06_pilot_replay_v1.py
$Run = Join-Path $env:TEMP ("UNICOUNT_T06_" + [guid]::NewGuid().ToString("N"))
py tools/thuy/t06_pilot_replay_v1.py --inputs docs/method/T06_SYNTHETIC_PILOT_INPUTS.jsonl --config configs/method/t06_pilot_replay_config.json --out-dir $Run
py -m pytest -q
```

Do not commit files from `$Run` as official pilot results. Git/GitHub actions occur **on Windows local only**. Linux server receives/returns code and artifacts only via ZIP/upload; no Git on server.
