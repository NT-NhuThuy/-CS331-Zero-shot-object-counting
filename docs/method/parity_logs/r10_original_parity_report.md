# T01 / G0 — P02 FP32 official pseudo3 GPU diagnostic

- GPU run: **T01_REAL_PSEUDO3_DIAGNOSTIC_32_PASS_PARITY_HOLD**
- Completed pairs: **32/32**
- Protocol: initial0 → pseudo3 → second forward, no human exemplar, no NMS, FP32.
- P02 cohort: **32 previously exposed image-query pairs**, GT-free inputs; NOT team-approved G0 present/absent parity.
- Present parity: **NOT_MEASURED** (independent verified labels + reference missing).
- Absent parity: **NOT_MEASURED** (independent verified labels + reference missing).
- Historical V14 equivalence: **HOLD** (source, AMP and raw binding differ).
- Official protocol frozen for this diagnostic only; team G0 approval still required.
- No GT labels read; no threshold tuning or T04 recovery performed.

## Evidence
- `protocol_manifest.json` candidate lock and source SHA.
- `roster_gt_free.json` and `per_pair_gpu_trace.jsonl` for exact pair coverage.
- `raw/*.npz` first/second-stage raw boxes and token probabilities, with SHA per pair.
- `candidates_diagnostic.jsonl` stable IDs, boxes, scores.
- `gpu_run_manifest.json` and `t01-worker.log` failures preserved.

## Gate to clear T01
Obtain team-approved frozen protocol and independently verified present/absent roster; produce matched reference raw predictions and exact per-pair count/native box/index parity. Do not relabel P02 32-pair diagnostic or R9 synthetic smoke as parity PASS.
