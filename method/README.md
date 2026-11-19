# UNICOUNT v2 — one method, three scientific contributions

**Single research question:** Can an open-world counter decide correct zero counts without suppressing real targets, while accurately counting present instances using active evidence and identity-consistent reasoning under constrained compute?

## Single entrypoint for design

- [`PIPELINE.md`](PIPELINE.md) — GitHub-rendered Mermaid architecture and evidence loop.
- [`METHOD_V2.md`](METHOD_V2.md) — scientific hypothesis, states and policy.
- [`TEAM_TASKS.md`](TEAM_TASKS.md) — next deliverables and owners.
- [`DECISIONS_20261117.md`](DECISIONS_20261117.md) — decisions and questions requiring joint sign-off.
- [`PROTOCOL.md`](PROTOCOL.md) — fair present/absent evaluation conditions.

**Current implementation:** `src/countlab/` is the existing *single runnable package* and remains unchanged in this non-breaking organizational PR. This is an intentional **phase-A structure**: there is no second implementation under `method/`. After all three contributors approve the v2 interface, make **one separately tested migration PR** for `src/countlab/` → `method/src/countlab/`, then adjust `pyproject.toml`, tests, scripts, CLI paths, and the native backend's repo-root resolution. Until then, imports continue from `src/countlab`.

**Source of truth for histories:** `legacy/thu`, `legacy/loc`, `legacy/thuy` (read-only). Do not transplant historical metrics across protocol versions or silently rewrite old results.

**Status:** design proposed, not a new empirically verified A* method. Current G0 data/protocol approval is HOLD; P06 is diagnostic and does not calibrate risk.
