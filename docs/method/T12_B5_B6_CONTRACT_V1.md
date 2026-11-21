# T12 · B5 NMS / complete-link & B6 one-pass fixed schedule

**Owner:** Thúy · **Task:** T12 G3 · **Status:** CPU contract delivery PASS on exported source; official T04/T05/T07 cohort and measured B7 cost HOLD.

## What this implements

- **B5_NMS:** existing `countlab.method.baselines.run_baseline_arm(…, "B5")` on the event-grounded projection; dual-rejected proposals are filtered and the rest use score NMS with a locked IoU threshold. This is the existing B5 *all observed evidence* arm, not a new physical-identity proof.
- **B5_COMPLETE_LINK:** existing `run_baseline_arm(…, "B8")`, exposed as the T12 complete-link geometric/edge comparator. No unobserved pairwise evidence is invented. If its graph arbitration is unresolved, inspect `unresolved_candidate_ids`. The naming is scoped to this T12 comparator; B8 remains the repository's original arm.
- **B6_FIXED:** action slots are computed once from the frozen T03 candidate pool **before** reading any event. C1's static risk/confidence priority defines candidate order; the fixed policy selects `REGION_SEMANTIC` then `REGION_WHOLE` for each candidate, followed by top geometry-priority `PAIR_IDENTITY` slots. Geometry **only prioritizes** possible identity verification; it never creates SAME or DIFFERENT. Observed events from the C2 evidence bank are projected onto selected slots. Missing evidence stays UNKNOWN with **no refill**. C3's existing T09 `run_t09_delivery` produces the strict ZERO/K/ABSTAIN / fallback output.

Only the new `t12_fixed_baselines_v1.py` module defines this orchestration; **existing** `baselines.py`, `router.py`, `joint_solver_v2.py`, `t09_delivery_v1.py` and frozen C3 arbitration **are unchanged**.

## Input / output

Input JSONL: one object per image/query with exactly `schema=countlab_t12_evidence_bank_row_v1`, `manifest` (T03 candidate pool, including original-pixel boxes and IDs), `observed_events` (T09 event format). The bank is a complete offline *observation opportunity inventory*. One event/action is supported; conflicting retries need a new protocol and are rejected, not silently under-billed.

Output: one `countlab_t12_b5_b6_matched_v1` row per source image/query with three arms **sharing the identical scheduled event subset**; includes action-by-action selected/missing trace, input/plan hashes, scheduled evidence slots, observed refs, separate strict and fallback counts. The runner refuses an existing output directory and duplicate image/query pairs.

`--out-dir` produces `B5_B6_predictions.jsonl` and `replay_lock.json`. The bundled `B5_B6_predictions_SYNTHETIC.jsonl` has **six synthetic contract cases only**, not real benchmark predictions.

## Fair comparison / limitations

1. B5, B6 use identical original candidate pool, immutable T03 ID namespace and **identical actually available selected evidence**. A hidden original `manifest.evidence` or `identity_edges` value cannot enter the result unless an action is explicitly selected and its event returned. No GT in inference.
2. `verification_budget_calls` measures **planned individual evidence-action slots**, not tokens, actual Qwen calls, milliseconds or GPU cost. The offline bank may be incomplete; empty slots do not imply a real call was executed. Output intentionally records `actual_model_calls=null`, `measured_gpu_seconds=null` and `matched_to_b7_measured_cost=false`.
3. **B7 active feedback is not part of this frozen source export**. B6 is a fixed-schedule C1→C2-event→C3 replay using the available shared APIs; identical model/prompt/backend and matched GPU-cost comparison to B7 cannot yet be claimed. Requires T07/T10/T11 handoff and locked shared roster.
4. `B5_NMS` may emit zero for an observed candidate set; **that is not certified scene absence**. B6 T09 may ABSTAIN with a nonzero original fallback. Do not overwrite strict `count=null` with forced fallback in papers or user-facing confidence claims.
5. Source-disjoint DEV/CAL/FINAL, verified absent labels, exact physical identity annotations, paired MAE/AZA/FZR+ and C2 billed-cost logs remain missing. **No accuracy gain, real-world efficacy or research contribution is claimed** by this delivery.
6. Do not treat geometry overlap as physical instance identity. C2 Qwen negative identity evidence failed E3/E4 diagnostics; such results cannot be used as certified pairwise DIFFERENT without independent revalidation.

## Reproduction on Windows local

At repo root (PowerShell):

```powershell
$env:PYTHONPATH = 'src'
py -m pytest -q tests/test_thuy_t12_fixed_baselines_v1.py
py tools/thuy/t12_b5_b6_replay_v1.py --inputs docs/method/T12_B5_B6_SYNTHETIC_INPUTS.jsonl --out-dir "$env:TEMP\unicount_t12_synthetic_run"
py -m pytest -q
```

If `$env:TEMP\unicount_t12_synthetic_run` exists, choose a fresh output directory. The runner is create-only and never overwrites results.

## Handoff requirements before real T12 acceptance

- **C1/T02:** approved source-disjoint roster and split hashes.
- **C2/T01+T04:** canonical CountGD++ pseudo3 manifest/pool with full provenance.
- **C1/T05, T08:** candidate-ID-matched signals and approved router ranking interface.
- **C2/T07+T10:** event/action API, model and prompt revision, actual Qwen success/UNKNOWN logs, tokens, latency and GPU cost.
- **C1/T11:** B7 active feedback runtime with same allowed action universe, model/prompt, starting pool and measured-cost budget.
- **T12:** after all above, run matched pilot on the approved roster, verify budget fairness and store real predictions/trace. Until then integration and scientific evaluation are **HOLD**.
