# Team Git playbook — ONE method, three component owners

## Team convention

Branch from `research/unified-integration-verified` (until approved into `main`). **Never each develop separate alternative full pipelines.** Each PR modifies a well-defined stage under `src/countlab/` with fixtures and tests; `legacy/` is archival and read-only.

| Person | Module to own now | Primary deliverable | Must pass before merge |
|---|---|---|---|
| Thư | `src/countlab/method/router.py`; `src/countlab/integrations/thu_features.py` | GT-blind risk scores and adaptive verification allocation, DEV-trained only | Strong uncertainty/margin baseline, matched calls and source-disjoint calibration; no train/test contamination |
| Lộc | `src/countlab/method/verifier.py`; `src/countlab/integrations/recovery.py` and `qwen_unary.py` | Native CountGD++ multi-view recovery and frozen per-instance Qwen evidence | Original coordinates, exact model prompts, revision/SHA, per-instance unary fixture, runtime and strong false-rejection analysis |
| Thúy | `src/countlab/method/identity.py`; `pipeline.py`; `adapters/fqie_legacy.py` | Safe association and bidirectional arbitration, validated against GT region identity | False merge/split vs NMS and naive Union-Find, over-pruning report, full output invariants |
| All | `tests/`, `configs/regimes`, `experiments/`, `paper/claim_ledger.csv` | Repeatable experiments, independent evaluation, falsifiable paper story | Reviewer-ready reproducibility / novelty / paired statistics |

## First-time setup, Windows PowerShell or Linux

```bash
git clone https://github.com/NT-NhuThuy/-CS331-Zero-shot-object-counting.git cs331-unicount
cd cs331-unicount
git fetch origin research/unified-integration-verified
git switch --track origin/research/unified-integration-verified
python -m pip install -e ".[test]"
python -m pytest -q tests/
countlab-method --input fixtures/unified_toy.json --config configs/method/unicount_v1.json --out local_toy_output.json
```

This works **after** someone pushes `research/unified-integration-verified` to `origin`. Read `docs/TEAM_START_HERE.md` for the bundle publishing workflow. The prepared bundle is local only; nothing was pushed to GitHub by the assistant.

## Work on the exact same method

```bash
git switch research/unified-integration-verified
git pull --ff-only origin research/unified-integration-verified
git switch -c feat/thu-split-audit # Thư; others use feat/loc-countgd-adapter or feat/thuy-v14-import
git add src/countlab tests docs/method
git commit -m "feat(router): integrate frozen risk features with tests"
git push -u origin feat/thu-split-audit
```

Open PR into **`research/unified-integration-verified`**, not immediately into `main`; require independent review. Never rewrite historical results. Use `experiments/runs/<immutable_id>` for new frozen artifacts (untracked files local by default).

## Single integration protocol

1. Agree on `countlab_candidate_pool_v1` input and `countlab_unified_method_output_v1` output; no owner changes the schema without updating tests + approval from other owners.
2. Each owner makes an adapter that produces EXACT SAME candidate/evidence types; don't add separate counting arithmetic after `pipeline.py`.
3. New features default OFF until parity + ablation on independent DEV. Prevent any model training on test/query GT accessible through inference path.
4. Merge after CPU CI, frozen GPU parity where required, source-disjoint audit, runtime budget and a failure-gallery report.
5. Freeze a candidate release tag; compare only then on untouched FINAL_SEALED; write null/negative findings, no SOTA claim without matched public benchmarks.

## Immediately actionable PR list

See [`PRIORITIZED_TASKS.md`](PRIORITIZED_TASKS.md) for P01–P12 with precise owners, branches, dependencies and go/no-go conditions. Start P01–P03 concurrently; P04–P07 once frozen parity and split audits are ready. Do not start model-dependent paper claims before independent evaluations.
