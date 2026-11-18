# T1 method progress — 2026-11-17

```yaml
owner: thuy
current_branch: feat/thuy-v14-import
commit: 2028aae (T0 base; T1 work is currently uncommitted)
pr_url: NOT_OPENED
base_branch_verified: false
p01_c: AUDIT_COMPLETE_HOLD
p01_d: INVENTORY_PASS_SOURCE_HASH_0
p05_a: CPU_TESTED
p02_candidate_ids: UNCONFIRMED
p03_split: HOLD
fresh_dev_instance_gt: NOT_VERIFIED
mechanism_headroom: SYNTHETIC_ONLY_WAITING_P02_P03
strong_controls: CPU_REPLAY_IMPLEMENTED
hypothesis_preregistered: true
solver: EXPERIMENTAL_JOINT_RULES_ONLY_NOT_GLOBAL_SOLVER
p05_b: HOLD
p09: HOLD
actual_gpu_calls: 0
scientific_claim_status: NONE
```

## Implemented in the current uncommitted worktree

- `joint_arbitration.py`: experimental baseline/rescue decisions using observed
  SAME/DIFFERENT/UNKNOWN; v1 default unchanged.
- `baselines.py`: B0–B9 replay controls.
- `t1_comparison.py` and `joint_diagnostics.py`: same-input replay and v1 delta
  trace. A delta is not an improvement claim.
- T1 problem, gap map, preregistration, formal constraints, falsification and
  baseline specs were drafted before empirical selection.
- This pass adds a create-only replay CLI/config and an offline-only identity
  metric evaluator. GT is joined only after all arms have run.
- Novelty threat and headroom status are recorded in separate reports.

## Commands

Focused CPU suite:

```bash
cd UNICOUNT
.venv/bin/python -m pytest -q tests/test_thuy_joint_arbitration.py tests/test_thuy_joint_diagnostics.py tests/test_thuy_t1_baselines.py tests/test_thuy_t1_comparison.py tests/test_thuy_t1_scenarios.py tests/test_thuy_t1_replay_cli.py
```

Create-only replay for one frozen manifest:

```bash
PYTHONPATH=src .venv/bin/python -m countlab.method.t1_replay \
  --manifest /path/to/frozen_gt_blind_candidate_manifest.json \
  --config configs/experiments/t1_joint_replay_v1.json \
  --out-dir /path/to/new_t1_replay_run
```

With reviewed instance labels, pass `--annotations` as the separate offline
evaluator input. It must be candidate-level `countlab_t1_instance_truth_v1`
and match the same image/query; it is never available to replay or config.

Create-only batch decision summary (no labels accepted):

```bash
PYTHONPATH=src .venv/bin/python -m countlab.method.t1_batch_replay \
  --index /path/to/manifest_index.json \
  --config configs/experiments/t1_joint_replay_v1.json \
  --out-dir /path/to/new_t1_batch_replay
```

`manifest_index.json` must use schema `countlab_t1_manifest_index_v1` and
contain unique, relative `manifest_paths`. The batch artifact hashes every
manifest and reports B6-vs-joint fallback-count deltas, changed-pair count,
unresolved/abstained pairs, planned pairwise calls, and action/reason totals.
It is a GT-blind decision summary, **not** an accuracy result or arm selector.

## Tested vs not tested

- Implemented: methods, B0–B9, joint diagnostics, create-only runner/evaluator.
- CPU tested: synthetic contracts/property scenarios only; no real dataset.
- Real-data validated: no.
- Paper claim allowed: none.
- P03 PR #6 integration remains unverified because the available remote did
  not advertise `origin/main`; no merge/push was performed.

## Next decision gate

Run the above replay on a P02-frozen candidate/evidence pool only after P03
approves source-disjoint DEV and evaluation labels. First measure candidate
recall ceiling and identity headroom. If instance annotations or observed
pairwise evidence are unavailable, record `NOT_MEASURABLE` and stop before
solver escalation. P05-B/P09 remain HOLD.

## Incremental control fix (2026-11-17)

B4 now masks unary evidence under the requested verification budget, while B5
keeps all frozen unary evidence available as the all-verify control. Their
trace records `budget_masked` versus `all_precomputed` explicitly. This makes
the budget ablation identifiable without changing the default v1 pipeline.
Focused CPU result after the change: 23 tests passed.

## Partition-aware offline audit (2026-11-17)

The offline evaluator now receives the reviewed candidate universe as well as
the predicted instance partition. It reports false merge, false split, missed
same-instance pairs caused by omitted candidates, and observed-instance
coverage. NMS suppression remains a selection action; it is never silently
relabelled as an identity `SAME` claim. Focused CPU result: 24 tests passed.

## Relation-blind pairwise allocation (2026-11-17)

The experimental joint layer now has a separate pairwise budget. It ranks
baseline--rescue collision pairs above rescue-only duplicate pairs from unary
states, role and geometry, before it can see any relation answer. Only edges
whose pairs were allocated are visible to arbitration; unallocated edges stay
`UNKNOWN`. This creates a falsifiable efficiency hypothesis: at an equal
pairwise budget, a decision-impact policy should resolve more count-changing
collisions than a generic risk order. Synthetic CPU result: 26 tests passed.

## Matched allocation controls (2026-11-17)

Frozen-evidence replay now emits both the impact-ranked joint policy and a
geometry-ranked pairwise control under the same unary budget, pairwise budget,
candidate pool and visible edge rule. The offline identity audit evaluates both
outputs after inference. This is the required comparator for the T1 efficiency
hypothesis; it has no real-data result yet. Synthetic CPU result: 27 tests
passed.

The matched replay also emits a risk-ranked control and a relation-blind
allocation summary (`collision_pairs`, `duplicate_pairs`, planned total), so
budget use can be audited before any labels are joined.

## Batch replay integration (2026-11-17)

`pipeline.run_method` now exposes `arbitration="sequential_v1"` (default) and
`arbitration="joint_t1"` (experimental). The first preserves the locked
sequential path; the second invokes joint baseline--rescue arbitration without
silently changing v1. `t1_batch_replay` creates a locked, hashed GT-blind batch
artifact over frozen manifests. The replay path and offline annotation join are
covered by CPU regression tests; this remains synthetic/toy infrastructure
until P02/P03 inputs are approved.

Offline batch metrics are evaluated separately:

```bash
PYTHONPATH=src .venv/bin/python -m countlab.method.t1_batch_evaluate \
  --index /path/to/manifest_index.json \
  --annotations-index /path/to/annotations_index.json \
  --config configs/experiments/t1_joint_replay_v1.json \
  --out /path/to/t1_offline_evaluation.json
```

The annotation index must use `countlab_t1_annotation_index_v1` and map each
`image_id/query_id` to exactly one reviewed annotation file containing both
`ground_truth_count` and `truth_by_candidate`. Evaluation is offline only and
does not select a policy or winner.
