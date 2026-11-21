# T06 G1 local handoff acceptance — 2026-11-21

**Source audited:** user-exported Windows `git archive HEAD d313721`, SHA256 `9398c523e6d27cecd0d37e78401ee30cd4f7032a44f0221b873e74b8ae809523`. No Git operations were performed on Linux/server. This patch is additive to HEAD.

| Acceptance item | Status | Evidence |
|---|---|---|
| Batch T03 candidate + T09 event parsing | PASS (synthetic CPU) | `t06_pilot_replay_v1.py`, new tests |
| Every pilot input yields one output/trace or entire invalid roster fails | PASS (synthetic CPU) | 9/9 example pairs, 40 T06 tests and duplicate-roster rejection |
| Same-object identity dedup; no identity-by-IoU | PASS (synthetic CPU) | T09 SAME fixture, missing SAME turns into ABSTAIN |
| Missing unary/pair evidence remains UNKNOWN | PASS | Unknown and empty-pool tests |
| GT excluded and no model calls | PASS (contract) | Nested GT leak and event reference validation |
| Candidate/view/box/evidence provenance, deterministic SHA | PASS (synthetic CPU) | `graph_trace`, lock, repeated bytes identical |
| T04 canonical C2 pseudo3 pool and matched pilot roster | **HOLD** | No team-signed handoff in ZIP |
| T05 real V4/A3 features source, exact-ID independently verified | **HOLD** | Only declared mapping contract, no authenticated C1 feature data |
| T07 real Qwen event logs and revisions | **HOLD** | Only synthetic events, no real C2 observed evidence |
| Official pilot graph/predictions, external schema signoff | **HOLD** | Synthetic files explicitly labeled, no real G1 benchmark |
| Matched B6/B7 real costs, MAE, scene absence, scientific claim | **HOLD** | Out of scope of local T06 replay |

**Snapshot verification:** 40/40 T06 acceptance tests and 349/349 complete repository tests PASS.

Source-snapshot tests were run with `PYTHONPATH=src` to support subprocess imports of uninstalled local `countlab`. Windows must rerun the full suite and validate the exact patch in its current branch before committing.
