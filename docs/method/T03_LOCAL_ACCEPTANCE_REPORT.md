# T03 local offline acceptance report — 2026-11-21

## Evaluated artifact and scope

- Base source supplied by user: `UNICOUNT_V2_3fb58ccedb69.tar` (offline extraction under an isolated scratch directory).
- Existing `src/countlab/method/schema.py` SHA256: `41f57a0c4eb0890ac09cf5cc35e75ce488953a1cb4a545ed9615f64af5947dda`.
- Dependency closure supplied by user: `UNICOUNT_C3_FULL_DEPS_20261120.zip`, whose frozen `schema.py` has exactly the same hash.
- New files are additive; no existing `src/countlab/method/*.py`, tests, router, solver, baseline or server files were modified in the deliverable ZIP.
- Software contract scope: GT-blind input shape, ternary evidence, pairwise IDs, no-candidate, duplicates, output ZERO/K/ABSTAIN semantics and fallback. Not official candidate-pool integration, benchmark effectiveness, parity, or clinical/scientific certification.

## Executed checks

| Test run on the supplied offline source snapshot | Result |
|---|---|
| Existing `tests/test_unified_method.py`, `tests/test_thuy_joint_arbitration.py` | **31 passed** |
| New `tests/test_thuy_t03_contract_v1.py` + existing 31 regression tests | **91 passed** (60 newly introduced) |
| Complete offline repository test suite with new test included: `PYTHONPATH=src python -m pytest -q` | **172 passed** |
| T03 fixture generator | Six real CPU method outputs generated with baseline `run_method`/`run_joint_arbitration` |

Python: CPython 3.13 (analysis container); no Qwen, detector checkpoint, GPU or GT file needed. The tests do not prove current Windows worktree CI is already green.

## Acceptance gates

- **G0 software contract tests (offline snapshot): PASS.** Example suite checks ZERO with empty pool, UNKNOWN baseline fallback, dual-NO rejection, conflict ABSTAIN, joint SAME coalescence, joint DIFFERENT separation.
- **G0 Windows branch smoke/PR CI: HOLD** until user applies this ZIP to current local branch and confirms no name/path conflicts and tests PASS there.
- **G0 integration with official T04/T05/T07 producers: HOLD** until those producers adopt the strict boundary and their real manifests pass, with allowed-field changes agreed by all owners.
- No publication claim, G2 data signoff, or T09 physical-instance correctness certification is implied.

## Source control boundary

Git/GitHub may be used **only from user's Windows checkout**, not from Linux inference server. This ZIP contains only small source/config/docs/tests; no giant binary data, checkpoints or source server output is bundled. Later server runtime artifacts must be explicitly copied to Windows before Git review and tracked by manifest+hash if needed.
