# UNICOUNT v2 — Thúy T12 Windows-only patch

Source exported from Windows HEAD `3da8aa2` (`feat/thuy-v2-t03-t06`).

Task T12 G3: **B5 NMS/complete-link + B6 no-feedback fixed schedule** and create-only GT-free batch replay. This is **CPU contract delivery**. Actual matched Qwen/GPU-cost tests and canonical roster integration remain **HOLD** until shared C1/C2 handoffs.

Files are ADDITIVE. Do not overwrite existing project files. Git/GitHub only on Windows. Never run Git on Linux server.

Windows instructions:

1. Check `git status --short` and last commit. It should be `3da8aa2` before applying.
2. Unzip this handoff package at `D:\EXERCISE\CS331\UNICOUNT_THUY_V2`. Check for filename conflicts first.
3. Run `py -m pytest -q tests/test_thuy_t12_fixed_baselines_v1.py` and then `py -m pytest -q` with `PYTHONPATH=src`.
4. Run the supplied CLI on the six synthetic fixture rows in a **fresh** output directory. Review `B5_B6_predictions.jsonl`, `replay_lock.json` and per-arm action traces.
5. Only after tests pass, stage the files and commit on Windows; no pushes before the team's integration/PR review.

See `docs/method/T12_B5_B6_CONTRACT_V1.md` for scientific limitations, schedule and dependencies. `docs/method/B5_B6_predictions_SYNTHETIC.jsonl` is deliberately named to prevent accidental use as real validation results.
