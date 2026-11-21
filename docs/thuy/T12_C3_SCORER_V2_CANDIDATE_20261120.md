# C3 T12 scorer v2 — Candidate for C1/C2 review

**Status:** experimental scoring adapter. **NOT TEAM APPROVED; G0 HOLD.**
This package does not alter `t12_offline_score.py`, inference code, original frozen protocol or any existing artifact.

## Why

Old scorer used `gt_count > 0` to determine Present MAE population, and required integer `gt_count` on every DEV row. C1 has an authoritative roster `eligibility` field: `COUNT_ELIGIBLE`, `PRESENCE_ONLY`, `INELIGIBLE`, `UNVERIFIED`. This adapter adds eligibility-aware scoring and a provisional **offline-only** label schema.

## Population definitions (PROPOSED; awaiting C1/C2 approval)

- `COUNT_ELIGIBLE`: offline `gt_count` is an integer >= 0, plus explicit `presence_label` (`PRESENT`/`ABSENT`), consistent with whether count is positive. Participates in **overall MAE** and, for GT positive, **Present MAE**. Participates in AZA/FZR+ according to presence.
- `PRESENCE_ONLY`: `gt_count: null`; explicit `presence_label`. Participates in AZA/FZR+ **only**, never count MAE.
- `INELIGIBLE`: both labels null; excluded from efficacy metrics but included in exclusions/total and abstention accounting.
- `UNVERIFIED`: fails clean DEV scoring.

AZA denominator: all verified `ABSENT` in the presence-evaluable stratum. FZR+ denominator: all verified `PRESENT` in the presence-evaluable stratum. Missing strata return `null`, not zero. For A5, report abstention separately and count MAE with pre-declared A0 forced fallback. Paired HELP/HARM and source-cluster descriptive bootstrap use only appropriate eligible strata.

## Provisional offline-label JSONL schema

Every DEV row must have a matching **separate** offline label row. Exactly these keys:

`dataset, image_id, query_id, source_image_id, source_group_id, split, presence_label, gt_count`

For example, a *synthetic* `COUNT_ELIGIBLE` label:

```json
{"dataset":"example","image_id":"image:1","query_id":"query:1","source_image_id":"src:1","source_group_id":"group:1","split":"DEV","presence_label":"PRESENT","gt_count":2}
```

For a *synthetic* `PRESENCE_ONLY` case:

```json
{"dataset":"example","image_id":"image:2","query_id":"query:2","source_image_id":"src:2","source_group_id":"group:2","split":"DEV","presence_label":"ABSENT","gt_count":null}
```

Labels NEVER enter C3 candidate generation, solver or action scheduler. Do not put these fields into the shared roster.

## Actual runtime gates

Command:

```text
python tools/thuy/t12_offline_score_v2.py --predictions /path/predictions.jsonl --labels /path/offline_labels.jsonl --roster /path/c1_roster.csv --contract /path/c1_contract.json --attestation /path/independent_approval.json --pool-manifest /path/frozen_t04_pool_manifest.json --policy /path/frozen_policy.json --out /new/output.json
```

- Must import and use **actual** `countlab.thu_t12_contract.audit_roster` and `validate_scoring_contract` from the integrated C1 branch. If C1 is absent in the worktree, **CLI fails closed**. This is expected in C3-only feature worktree. Unit tests work independently.
- C1 metadata PASS and C1 declared scoring-contract PASS are both required; they **do not certify G0 by themselves**.
- Independent approval **also** required: `status=approved_for_dev_scoring`, `g0_decision=GO_FOR_DEV`, nonempty `g0_evidence_reference`, `approved_by`, `approval_reference`, plus `source_disjoint_verified=true`, `candidate_ids_attested=true`, `source_image_hashes_attested=true`, linked `split_protocol_id`, `frozen_pool_run_id`.
- Approval must match SHA256 of the **exact** predictions, labels, roster, scoring-contract file, pool manifest, policy file, and historical T12 protocol hash (`t12_protocol_sha256`). Field names: `prediction_file_sha256`, `label_file_sha256`, `roster_file_sha256`, `contract_file_sha256`, `pool_manifest_sha256`, `policy_file_sha256`.
- Scoring contract's `paired_roster_sha256`, `frozen_pool_manifest_sha256`, `policy_sha256` must match the corresponding actual files.
- Predictions MUST have the unchanged A0/A1/A3/A4/A5 arm set, unchanged A0 forced fallback, A1/A3 parity, A5 abstention guard, `gt_used=false`, `scene_zero_certified=false`, `pool_scope=full_query_pool`, `cohort_role=dev`, and `model_calls_in_this_replay=0`.
- Predictions' `(image_id, query_id)` must exactly and uniquely join DEV roster and separate offline labels, with `source_group_id` matching. The roster also must be unique on `(dataset, query_id)`. **Predictions do not have per-row image SHA256**; independent source-to-prediction linkage and candidate ID attestation remain required. Never infer lineage from a matching query string alone.
- Existing A2 NMS/complete-link sidecar is not equivalent to a C1-approved B5, and A6/B6+A7/B7 are not scored by this adapter. `model_calls_in_this_replay=0` is not a claim of GPU or token cost savings.
- External approval file is a **declaration**, not a cryptographically signed source-identity proof. Reviewers must inspect underlying provenance; this code does not manufacture or self-grant G0.
- Keep the original frozen protocol and six frozen modules unchanged. The new scorer has its own independent revision/manifest.

## Validation status

18 isolated synthetic Python tests passed at package construction. Tests include a stub C1 PASS during an isolated integration-flow test solely to validate file and hash gates. **No real C1 approved contract, no real G0 approval, no clean DEV run, no empirical accuracy or scientific superiority has been demonstrated**. Run the full Windows repository tests before committing; actual C1 integration needs review in the shared branch.
