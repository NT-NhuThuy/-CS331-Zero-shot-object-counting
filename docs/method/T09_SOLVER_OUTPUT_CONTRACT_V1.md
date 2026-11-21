# T09 G3 — Joint instance solver, controlled evidence and dual decision scope

**Delivery date:** 2026-11-21. **Owner:** Thúy. **Artifact:** additive T09 handoff on Windows branch `feat/thuy-v2-t03-t06`, starting from commit `e7b905a`.

## Exact modules

- Existing, unchanged source of truth: `src/countlab/method/joint_solver_v2.py` (`solve_joint_v2`), `t09_event_grounded_solver.py`, `t09_constraint_bounds.py`.
- New opt-in boundary: `src/countlab/method/t09_delivery_v1.py` (`run_t09_delivery`). It validates T03 input, passes through the original joint solver, checks witnesses/fallback invariants, and makes every candidate's actually observed references visible.
- Frozen strict config: `configs/method/solver_config.json`, exactly the four keys expected by `validate_solver_config`. No tuning on exposed FSC samples.
- Trace fixtures: `docs/method/solver_trace_examples.jsonl`, 13 synthetic CPU cases with both inputs and the actual replayed output.
- Deterministic command: `tools/thuy/t09_validate_v1.py`. It supports `--examples` replay or `--manifest --events --fallback --out` for one query.

## Input boundary

`manifest` must satisfy `countlab_candidate_pool_v1` via `validate_candidate_pool` from T03: baseline/rescue each carry unique IDs and original-image `box_xyxy`; `baseline_count` equals the number of baseline proposals; no GT/annotation leakage and no unexpected schema keys.

`observed_events` are an exact-key list of `REGION_SEMANTIC` (YES/NO/UNKNOWN), `REGION_WHOLE` (YES/NO/UNKNOWN), or `PAIR_IDENTITY` (SAME/DIFFERENT/UNKNOWN). Every event refers to existing candidate IDs and has a unique nonempty `evidence_ref`. **Prefilled verifier answers in the manifest are ignored for T09 counting**: only actually delivered observed events are used. Contradictory informative retries cause abstention; UNKNOWN cannot act as negative or as DIFFERENT. The input fallback is separately supplied and frozen by the caller.

## Two non-interchangeable decisions

`decision` is the **strict** solver outcome: `K` if a positive conditional observed-candidate count has physical-instance witnesses, otherwise `ABSTAIN` and `count=null` with unchanged `evaluation_fallback_count`.

`observed_pool_decision` is **not scene certification**: `K` for the same positive conditional result; `ZERO` if and only if there is at least one observed candidate, no strict blocking issues, and **every** candidate has observed semantic `NO` and whole-instance `NO`; `ABSTAIN` otherwise. On ZERO, `observed_pool_count=0` while `decision=ABSTAIN`, `count=null`, and fallback stays unchanged. Empty pool **never** gives observed-pool ZERO.

A claim that the entire image contains zero target objects requires an independent scene-absence argument. **Neither T09 nor this wrapper provides that argument.** Fields `scene_zero_certified=false`, `operational_count_correction_authorized=false` and `observed_pool_zero_not_scene_certified=true` are fixed. No action execution, Qwen, CountGD or GT read occurs.

## Identity and conflict semantics

- Two boxes are **not** automatically different because coordinates differ; two overlapping boxes are **not** automatically SAME.
- `SAME` components can be deduplicated for positive conditional K only when the solver has required direct SAME evidence, including all pairs in a multi-member clique. A transitive chain alone is insufficient.
- A positive conditional K with multiple instances needs observed `DIFFERENT` separation, positive semantic+whole anchors, and no unresolved or contradictory group.
- An observed SAME/DIFFERENT conflict, contradictory unary, incomplete SAME clique, or exhausted search resource leads to ABSTAIN/frozen fallback; no unsafe count promotion.
- `candidate_trace[]` gives role, view, source IDs, actually observed unary/identity evidence refs, conflict verdicts, and strict instance membership. `strict_result.physical_instances[]` contains witness refs and a source-visible hash. The event-reference strings are **not cryptographically authenticated**; the solver assumes a truthful upstream verifier.

## Acceptance examples and restrictions

The 13 frozen synthetic examples cover: empty pool, missing unary, dual-NO not scene ZERO, singleton positive, two explicit DIFFERENT instances, SAME dedup, contradictory pair, contradictory unary, two positive objects with missing DIFFERENT, incomplete SAME clique, precomputed hidden answers masked, UNKNOWN-then-YES retry, rescue-only positive. Tests additionally cover unknown pair, invalid event/GT fields, duplicate IDs, config limits, deterministic ordering, candidate-level evidence provenance and CLI replay.

The existing `c3_dual_channel.py` **operational guarded-identity path is not upgraded or certified by this handoff**. A separately tested unary-veto variant exists but is not silently substituted for the frozen A4 arbitration. No evidence was obtained here that Qwen's DIFFERENT is reliable: prior E3–E4 diagnostic observations found unsafe false DIFFERENT. Consequently the data-dependent C2/T07 evidence handoff, matched B5/B6 comparison and team integration remain HOLD. These tests prove software contract behavior on synthetic input, not counting MAE, zero-scene truth, or A* contribution.
