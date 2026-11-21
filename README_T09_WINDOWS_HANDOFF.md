# T09 Windows source handoff (additive, no Linux Git)

This ZIP adds a strict trace/validation boundary to the existing `joint_solver_v2.py`; it does not replace the frozen solver, A4 arbitration, T03 validator, or T12 comparator. All event-evidence test cases are synthetic. The original Windows branch was `feat/thuy-v2-t03-t06`, with T03 HEAD `e7b905a` at upload time.

## Apply on Windows

1. Confirm current branch and a clean working tree with `git status --short` and `git branch --show-current`.
2. Verify SHA256 of the current three core files against `T09_HANDOFF_MANIFEST.json` (`base_source_sha256`). If any differ, **HOLD**, review the actual branch before applying. Do not overwrite them.
3. Download `UNICOUNT_C3_T09_LOCAL_HANDOFF_20261121.zip` into Downloads, check its filename, and check target paths for collisions before extracting into the repo root.
4. Set `PYTHONPATH=src`; run `py -m pytest -q tests/test_thuy_t09_delivery_v1.py`, `py tools/thuy/t09_validate_v1.py --examples docs/method/solver_trace_examples.jsonl`, then `py -m pytest -q`.
5. Review `git diff --check` and the 9 added files. Only commit on Windows after local tests pass. A PR into `research/unified-integration-verified` requires the leader's G3 acceptance, and integration dependencies remain HOLD.

## One-query CPU replay

Prepare two existing JSON files: an unmodified T03 manifest and a list of genuinely observed events (not hidden precomputed answers). For example, in Windows PowerShell:

```powershell
$env:PYTHONPATH = 'src'
py tools/thuy/t09_validate_v1.py --config configs/method/solver_config.json --manifest path\to\manifest.json --events path\to\observed_events.json --fallback 1 --out path\to\new_t09_output.json
```

The CLI uses exclusive file creation; it will refuse to overwrite existing output. `--examples` compares all 13 expected outputs exactly with current solver execution. It never contacts a GPU.

## Hand-off limits

- Strict `K` is conditional on evidence; strict `ABSTAIN` retains the frozen comparator count. Observed-pool `ZERO` is **not** a claim of scene absence and is never used to rewrite operational count here.
- Real Qwen/CountGD evidence (T07) can be wrong; T09's unit tests do not independently validate its visual truth.
- No model, dataset, GT, server-generated logs, Git history or large artifacts are contained in this package.
- `T09_HANDOFF_MANIFEST.json` lists every delivered file and its SHA256 except itself.
