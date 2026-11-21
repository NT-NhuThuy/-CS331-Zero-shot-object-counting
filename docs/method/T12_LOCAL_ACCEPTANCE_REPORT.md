# T12 local acceptance — 2026-11-21

**Source:** `UNICOUNT_T12_LOCAL_AUDIT.zip`, 222460 bytes, SHA256 `183b0376e58d2f92b987d9ff0a15198efa159b31312149550cf70f5af1ed14c0`. Origin Windows commit `3da8aa2`, branch `feat/thuy-v2-t03-t06` (reported by owner). This is a subset Git archive, **not** a full checkout.

| Requirement from leader's 15-task sheet | Evidence | State |
|---|---|---|
| `B5_config.json`, NMS/complete-link | File + reuse of existing B5 and B8 controls | PASS offline |
| `B6_config.json`, one-pass fixed schedule | New GT-blind frozen schedule, no outcome-dependent retask | PASS offline |
| Same input / roster / identical observed evidence | One shared input manifest and action-selected event projection across all three arms | PASS offline |
| Same C1/C2/C3 modules as B7 | C1 static risk, C2 offline event format and T09 C3 reused; *B7 active runtime not available* | PARTIAL / HOLD integration |
| Schedule / budget trace | Per-slot schedule, observed and missing refs, SHA | PASS offline |
| Predictions / smoke tests | 6 synthetic pairs, create-only JSONL, new T12 acceptance pytest | PASS synthetic |
| Real `B5_B6_predictions.jsonl` for canonical roster | Missing T04/T05/T07 frozen handoff | HOLD |
| Measured GPU / token cost matching B7 | C2 cost logs and C1 B7 not available | HOLD |
| Pairwise DEV outcome / scientific superiority | No approved eval split or metrics | HOLD |

**Observed offline smoke test:** 6/6 synthetic scenario rows generated and checked. New T12 contract tests 26/26 PASS before final package lock. Combined exported-source suite: **306 PASS, 3 FAIL due only to omitted `fixtures/unified_toy.json` and `manifests/historical_experiment_lineage_audit_v1.json`** in the deliberately partial Git archive. These were not edited by this patch; run full `py -m pytest -q` on the actual Windows checkout for authoritative regression.

**No files from `src/countlab/method/` existing code were overwritten.** Original strict T09 and other frozen modules remain unchanged. Git, push and PR actions belong only on Windows local.

**Scope:** task delivery, not A* research, MAE improvement, or certified physical truth.
