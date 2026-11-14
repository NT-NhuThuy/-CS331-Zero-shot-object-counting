# UNICOUNT — one unified CountGD++ research method (CS331)

**Research working name only**. One pipeline with three module owners, not three disconnected notebook-based methods. The code in this branch runs **GT-blind CPU replay on frozen proposals/evidence**; GPU source/model adapters and any new A* novelty are future tasks. This repo is not a benchmark-verified SOTA method.

## One method, one executable

```text
image, category query
  → frozen CountGD++ baseline boxes
  → Thư: risk-aware verification budget / routing
  → Lộc: multi-view candidate recovery + per-instance verifier
  → Thúy: identity-consistent bidirectional arbitration
  → count + accepted/rejected/unknown + trace
```

**Start here:** [`docs/method/METHOD_V1.md`](docs/method/METHOD_V1.md) (scientific formulation), [`docs/method/REPO_PLAYBOOK.md`](docs/method/REPO_PLAYBOOK.md) (who implements what), [`docs/method/INTERFACE_CONTRACT.md`](docs/method/INTERFACE_CONTRACT.md) (exact I/O), [`docs/method/MILESTONES.md`](docs/method/MILESTONES.md) (A* go/no-go gates).

## Try the unified method's CPU replay

```bash
python -m pip install -e '.[test]'
countlab-method --input fixtures/unified_toy.json --config configs/method/unicount_v1.json --out toy_result.json
python -m pytest -q tests/
```

The toy fixture expects **2 chairs** according to synthetic labels supplied only here as example reasoning, not a model prediction: baseline count 2, one explicit false detection removed, two multi-view rescue proposals merged into one physical instance, final point count 2. The replay uses frozen synthetic YES/NO + SAME inputs, so this is a **contract smoke test, not accuracy evidence**.

To import actual old FQIE task records into the shared schema (still no GPU):

```bash
countlab-import-fqie --index /path/to/bidirectional_task_index.json \
  --unary /path/to/identity_unary_decisions.jsonl --out-dir local_fqie_manifests
```

Then `countlab-batch --inputs local_fqie_manifests --outputs local_unified_results --config configs/method/unicount_v1.json` runs the same decision method on every converted pair.

Generated outputs are create-only. Outputs are **observed-candidate**, not guaranteed scene-wide bounds. The evaluation/GT path remains isolated (`countlab-audit` on already frozen prediction CSVs).

## Repository map

```text
src/countlab/method/{schema,router,verifier,identity,pipeline,cli}.py  # ONE method
src/countlab/adapters/fqie_legacy.py                           # legacy → shared I/O
fixtures/                                                     # deterministic CPU examples
tests/                                                        # shared method unit tests
configs/regimes/                                               # separate legitimate protocols
docs/method/                                                   # shared method + owners + milestones
experiments/                                                   # versioned future runs
paper/                                                         # claims with evidence level
legacy/{thu,loc,thuy}/                                         # original work preserved unchanged
.github/workflows/ci.yml                                       # tests
```

Historical evidence and the first broad comparative audit are preserved under [`docs/A_STAR_RESEARCH_REPORT.md`](docs/A_STAR_RESEARCH_REPORT.md). The older [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) should now defer to `docs/method/METHOD_V1.md` when specifications differ.

## Critical boundaries

- **V14** reports improvement from 1.7611 to 1.1000 MAE, but its raw frozen inference outputs were not included in this mirror; do not call the new unified CPU runner a reproduction of that result.
- **Thư** has trained Joint V3.2 on exploratory FSC data; its direct scalar count head is a separately supervised baseline until tied to candidate decisions properly.
- **Lộc** V7.6 global Qwen veto is not the same as a region-level semantic/whole verifier and worsened positive MAE on PrACo despite excellent mismatch suppression.
- V14 and FSC-147 TEST are exposed. Need NEW calibration and untouched final tests for a newly designed combined method.
- No claim of first-ever novelty, general SOTA or conference acceptance until matched research baselines and external benchmarks support it.

For Git collaboration, have the repository owner push `research/unified-method-v1` after reviewing the supplied bundle. Then everyone clones and branches from it using the playbook.
