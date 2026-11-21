# T06 G1 Windows-only local handoff

This ZIP is **a patch of new files**, not a replacement repository. Verified against user-provided `UNICOUNT_T06_LOCAL_AUDIT.zip` from HEAD `d313721`. No existing source file is overwritten; `joint_solver_v2.py` and other frozen modules remain untouched.

T06 targets **graph replay for every canonical pilot query**, but the current archive does **not** include signed C2 T04 candidate pool, C1 T05 features, or C2 T07 observed events. Thus only the **synthetic CPU delivery** can pass; real G1 pilot remains HOLD.

On Windows PowerShell at `D:\EXERCISE\CS331\UNICOUNT_THUY_V2`:

1. Verify HEAD is `d313721` (or reconcile later changes), and `git status --short` is clean.
2. Verify download ZIP SHA256 against the value supplied in the conversation.
3. Check that all patch paths are absent, then expand ZIP into repo root.
4. Run with `$env:PYTHONPATH = 'src'`:
   - `py -m pytest -q tests/test_thuy_t06_pilot_replay_v1.py`
   - `py tools/thuy/t06_pilot_replay_v1.py --inputs docs/method/T06_SYNTHETIC_PILOT_INPUTS.jsonl --config configs/method/t06_pilot_replay_config.json --out-dir $Run` (set `$Run` to a fresh temp directory)
   - `py -m pytest -q`
5. Confirm all tests PASS; review `T06_LOCAL_ACCEPTANCE_REPORT.md`, then stage only new T06 patch files and commit in Windows Git. PR/CI and real T04/T05/T07 handoff are separate gates.

Full usage and meaning: `docs/method/T06_PILOT_GRAPH_REPLAY_CONTRACT_V1.md`. The ZIP ships 9 synthesized *contract* pairs and their frozen traces as `T06_SYNTHETIC_*`; it does not contain official benchmark predictions or claim count gains.
