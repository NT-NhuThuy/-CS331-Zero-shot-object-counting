# FQIE multi-reference consistency — handoff report

**Prepared:** 2026-11-10 (Asia/Ho_Chi_Minh)  
**Active experiment:** `fqie_multiref_multiscale_rescue_20261110_v2`  
**Decision:** freeze **multi-reference standard** as the validated precision-filter result.  Reject the current fixed multi-scale rescue formulation.  Do not tune either on any cohort whose GT has now been opened.

---

## 1. Research question and method

The task is target-aware counting on LVIS validation image/query pairs.  The p0 Count Anything baseline emits points.  This work investigates whether multiple in-image OWLv2 proposals, after semantic certification, can provide a **query-specific visual-consistency boundary** that removes false p0 points without a globally tuned similarity cutoff.

### Standard multi-reference rule

For each query:

1. Generate diverse OWLv2 reference proposals; crop them.
2. Build a GT-free hard-confuser bank from nearest LVIS text labels.
3. Certify a reference only if its `target - best_confuser` margin is strictly positive in all three views: original, center-90%, horizontal flip.
4. If at least two references are certified, compute their pair affinities.  The query consistency floor is the minimum pair affinity.
5. For a p0 point crop, remove it only if all conditions hold:
   - the query has at least two certified references;
   - the point loses target-vs-confuser in at least one fixed view (`minimum_margin <= 0`);
   - its maximum similarity to any certified reference over all cross-views is below the consistency floor.

Queries without two certified references abstain: no p0 point is changed.

This is a **precision filter**.  It neither adds proposals nor claims recall recovery.

### Rejected multi-scale rescue rule

For a point that standard removes, additional 12.5% and 37.5% local crops were scored.  The point was rescued only if at least one alternate crop had both:

- positive target-vs-confuser margin in all three views (`minimum_margin > 0`); and
- maximum reference similarity at least the **same frozen query consistency floor**.

No new point/proposal was added.  This rule was pre-locked before GT evaluation.

---

## 2. Cohorts and results

### A. Prior multi-reference development cohort (v1; positive-only)

Path: `../fqie_multi_reference_consistency_20261109_v1/`

Protocol:

- 100 positive LVIS query/image pairs, source-image-disjoint from confirm300.
- OWLv2 top-3 proposal/reference pipeline and candidate selection were frozen before GT.
- 44/100 queries had at least two certified references.
- 75/428 p0 candidates were removed.

Post-lock result:

| Method | MAE | RMSE | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| p0 | 2.24 | 6.61 | 197 | 231 | 55 |
| multi-reference verifier | **1.85** | **5.45** | 178 | **175** | 74 |

Interpretation: removed 56 FP but also 19 TP; MAE fell 17.4%.  This established a promising precision-oriented signal but did **not** test verified-absent safety.

Primary report: `../fqie_multi_reference_consistency_20261109_v1/RESULT.md`

### B. First 50-present/50-absent cohort (v1 rescue setup; do not use for further tuning)

Path: `../fqie_multiref_multiscale_rescue_20261110_v1/`

It froze 293 proposals and produced only 29/100 multi-reference-eligible queries.  The input also contained `query_status` (`present`/`verified_absent`) even though annotations were empty.  The inference code did not consume that field, but it violates a strict blind-evaluation standard once humans/tools inspect the input.  Therefore this cohort is historical/auditable only; do **not** promote or retune on it.

Its important reference-stage artifact:

- `reference_pair_scores_freeze.json`: 29 queries with >=2 references, 57 pairs.

### C. Fresh blind confirmation cohort (v2; final current result)

Path: `.` (this directory)

The public inference input has no `query_status`; status exists only in evaluator-only metadata.  It is source-image-disjoint from confirm300, the v1 positive cohort, and v1 rescue cohort.

Composition was selected from LVIS annotations but inference received no annotations/GT:

- 50 present pairs
- 50 explicitly verified-absent pairs
- 100 source-distinct images

#### Reference-stage lock (before GT)

- Exactly five diverse OWLv2 proposals/query after NMS, 500/500 total.
- 500 reference crops; none rejected.
- 175/500 references certified.
- 48/100 queries had >=2 certified references.
- 232 reference-pair affinities frozen.
- Pair-affinity max-cross-view mean 0.854237; min 0.510836; max 1.0.

Coverage by evaluator-only status:

| Status | Queries | Certified refs | Queries with >=2 refs |
|---|---:|---:|---:|
| present | 50 | 123 | 31 |
| verified-absent | 50 | 52 | 17 |

Important implication: a certified reference does **not** establish target presence.  17 verified-absent queries still had >=2 certified references.

#### Candidate-stage lock (before GT)

- p0 frozen output: 100 query records, 316 points.
- Three windows/point: scale 0.125, 0.25 (base), 0.375.
- 948 crops scored on GPU across the fixed three views.
- 185/316 points were eligible for multi-reference voting.
- Standard removed 38 points.
- Rescue restored 2; rescue arm therefore removed 36.

#### Post-lock evaluation (GT opened only here)

| Arm | MAE | RMSE | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| p0 | 2.23 | 4.544 | 102 | 214 | 19 |
| standard | **1.99** | **4.362** | 96 | **182** | 25 |
| rescue | 2.01 | 4.383 | 96 | 184 | 25 |

Standard versus p0:

- MAE `2.23 -> 1.99`: reduction 0.24 / **10.8%**.
- FP `214 -> 182`: 32 fewer FP.
- TP `102 -> 96`: 6 TP lost.
- FN `19 -> 25`: 6 additional FN.

Breakdown:

| Arm | Present MAE | Present TP/FP/FN | Absent FP mass | Absent-active queries |
|---|---:|---:|---:|---:|
| p0 | 1.44 | 102 / 63 / 19 | 151 | 45/50 |
| standard | **1.38** | 96 / 52 / 25 | **130** | **41/50** |
| rescue | 1.40 | 96 / 53 / 25 | 131 | 41/50 |

Thus standard replicated a useful FP/count-error gain on a fresh source-disjoint mixed present/absent cohort.  It is the selected arm.

#### Rescue audit

Both rescued points were false positives; rescue recovered no TP:

1. `query_id=4`, `tablecloth`, present, point index 1, rescued at 0.375.  Standard: count 2, TP 1, FP 1.  Rescue: count 3, TP 1, FP 2.
2. `query_id=58`, `hose`, verified-absent, point index 0, rescued at 0.375.  Standard: 7 FP.  Rescue: 8 FP.

The fixed-scale rescue formulation is rejected.  Do not alter its scale/margin/floor thresholds on this opened cohort.

---

## 3. Important implementation/protocol corrections already made

### Blind input

`prepare_fresh_input.py` creates:

- public `input/inference_coco.json` — no annotations and no `query_status`;
- private `input/evaluator_manifest.json` — maps query IDs to present/verified-absent for evaluator only.

The Count Anything adapter requires a bookkeeping status field.  `prepare_p0_input.py` creates a p0-only shadow input where every `query_status` is the constant string `hidden`; this value is not a GT label and is not provided to the model.

### Top-5 semantics

The initial v1 rescue run claimed top-5 but yielded <=3 actual proposals/query due detector threshold/NMS behavior.  v2 uses postprocess threshold `0.0`, then NMS IoU 0.6, and fail-fast if fewer than exactly five diverse proposals remain.  v2 froze 500 proposals for 100 queries.

### Tokenizer issue

The `ValueError: Unable to create tensor ... padding/truncation` came from forcing **SigLIP2** to the slow tokenizer during the confuser batch stage.  It was fixed by removing that override.  OWLv2’s successful frozen v2 run used `use_fast=False`; current `owl_top5_gpu.py` is synchronized to that actual frozen setting.  The warning about future `use_fast=True` default is informational, not an error for the locked run.

### Partial crop race (resolved)

During first multiscale crop preparation, an execution-session timeout/race yielded an incomplete temporary crop directory.  It was deleted along with its inconsistent manifest/lock and regenerated cleanly.  Final valid state is exactly 948 crop files and a matching manifest hash.

---

## 4. Key files to read

Read these in approximately this order when resuming.

### Executive evidence

1. `HANDOFF_REPORT.md` (this file)
2. `evaluation.json` — authoritative v2 metrics and per-query p0/standard/rescue results.
3. `candidate_arm_lock.json` — final pre-GT selection summary and ledger hash.
4. `candidate_arm_ledger.json` — every standard/rescue decision and evidence.
5. `../fqie_multi_reference_consistency_20261109_v1/RESULT.md` — original positive-only result.

### v2 protocol/input/reference artifacts

6. `prepare_fresh_input.py`
7. `input/lock.json`
8. `input/inference_coco.json` — public blind input.
9. `input/evaluator_manifest.json` — evaluator-only labels; **do not use in inference/selection**.
10. `owl_top5_gpu.py` and `owl_topk_freeze.json`
11. `prepare_reference_crops.py`, `build_confusers_gpu.py`, `score_reference_crops_gpu.py`, `select_references.py`, `score_reference_pairs.py`
12. `certified_references_lock.json`, `reference_pair_scores_freeze.json`

### v2 candidate/arm implementation

13. `prepare_p0_input.py`, `p0_adapter.py`, `freeze.json`
14. `prepare_multiscale_candidates.py`, `multiscale_candidate_manifest_lock.json`
15. `score_multiscale_candidates.py`, `multiscale_candidate_scores_freeze.json`
16. `select_standard_and_rescue.py`
17. `evaluate_arms.py`

### Runners

- `RUN_P0_PREPARE.sh`
- `RUN_P0_GPU.sh`
- `RUN_PREPARE_MULTISCALE_CANDIDATES.sh`
- `RUN_SCORE_MULTISCALE_GPU.sh`
- `RUN_LOCK_ARMS.sh`
- `RUN_EVALUATE_ARMS.sh`

Reference-stage runners (already completed in v2):

- `RUN_PREPARE_FRESH_INPUT.sh`
- `RUN_TOP5_GPU.sh`
- `RUN_REFERENCE_PIPELINE_CPU.sh` or resume variant `RUN_REFERENCE_AFTER_CROPS_CPU.sh`

---

## 5. Final v2 provenance/checkpoint values

These are the values to recognize when auditing the completed v2 run.

| Artifact | SHA-256 / key value |
|---|---|
| p0 predictions | `6d89ca418ecc5690970707b3a02e8ecb0580aa6aef2f6af6ed22005f5c0b28f5` |
| p0 lock | `5596b5fb8e35f4ebeae4b915f3d15321d3a29f2d51ea5fc299f2c082650ce9db` |
| certified-reference lock | `f92a62d7c2e778c0ea38a11f5deaea653f6a0628d79f59bfc6893bb51e7383f6` |
| pair scores | `5b956cd6be5f9a0849f1b51763d28ae943a72dd93d5891d72d8a79447b8f0cb4` |
| candidate crop manifest | `5c3caccc91f5154e8f67caeaf9e6caa9a71b6e9aa49139a603e6932ab3381dc6` |
| candidate score file | `93e8d5721fcdd04ed78e9d7ee0cbc5170c8ea3500a7593b64b8f5f71d3450184` |
| final selection ledger | `450dc6fa619d21d8adf9125c93fd558e1f9b016afffb0fe8bdd583423b227ff1` |
| final selection lock | `f3aeeedad950adb3c43e2b83c83c292573c8c0fa816291b89e5a6232463b32ea` |
| final evaluation | `2f195067a188c807ab823e40b13a55845505f310dde2ea496e8b6ef891daed3e` |

---

## 6. What is complete versus what must not be done

### Complete

- v1 discovery result and v2 fresh confirmation result.
- A blind (at inference time) present/absent v2 protocol.
- Exact top-5 reference generation.
- Standard multi-reference verifier, pre-GT lock, and evaluator-only measurement.
- Fixed multi-scale rescue test, pre-GT lock, and negative result.

### Do not do

- Do **not** tune scales, margins, similarity floors, number of reference proposals, NMS threshold, or reference certification using either v1 or v2 now that GT is visible.
- Do **not** use `input/evaluator_manifest.json` in any inference or selection code.
- Do **not** overwrite frozen artifacts in this directory; scripts intentionally refuse overwrite.
- Do **not** reinterpret the rescue outcome as evidence that scale never matters.  It only rejects the specified fixed-crop rescue rule.

---

## 7. Recommended next research branch

The standard branch is mature enough to freeze as:

> **Multi-reference consistency precision filter v1** — validated on a source-disjoint positive-only cohort and a fresh mixed present/verified-absent cohort.

The main remaining limitation is **query-level presence/safety**, not candidate crop scale:

- 17/50 verified-absent queries had >=2 certified references.
- Standard improved absent safety but still left 41/50 absent queries active and 130 absent predicted points.

The highest-value next branch is **query-presence gating** on a fresh, source-disjoint, hidden-GT cohort:

1. Create a new cohort disjoint from all three previous cohorts.
2. Build a query-level GT-free presence witness using only proposal/reference evidence.
3. Pre-register/lock the rule before reading its GT.
4. Compare p0, standard filter, and standard + presence gate.
5. Report present recall/count error and verified-absent active-rate/FP mass separately.

An alternative, lower-priority branch is robust reference clustering: identify and exclude inconsistent/outlier certified references before setting a floor.  It may improve both coverage and absent safety, but must also be evaluated on a new cohort.

Avoid a direct continuation of fixed multi-scale crop rescue: it yielded 0 TP recovery and +2 FP on v2.

---

## 8. Minimal resumption prompt

Use this in a new conversation after supplying this report:

> Read `ZOC/target_aware_counting/output/fqie_multiref_multiscale_rescue_20261110_v2/HANDOFF_REPORT.md` completely.  Preserve every frozen artifact.  Treat v2 standard multi-reference as the selected precision-filter baseline and v2 multi-scale rescue as rejected.  Develop a new, source-disjoint, GT-hidden query-presence-gating experiment; do not tune on prior cohorts.

