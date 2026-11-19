# C1 / Thư — T12 paired evaluation handoff (preflight v1)

## Scope, status

This patch **does not** create an approved clean DEV/CAL/FINAL dataset; T02 technical PASS does not imply G0 release. Existing 1,301 FSC extended, 111 Loc/Thu mirror, and fresh160/fresh126 are historical *exposed* diagnostic cohorts. fresh126 is part of fresh160; 111 mirror is not independent. Do not use 8 exposed detector-pool queries as clean DEV.

Source-level independence requires a **new externally verified lineage roster**: raw-image SHA256 is byte-exact only (not proof against re-encoding/cropping); any `UNEXPOSED` claim must be supported by audited source exposure history. `FINAL_SEALED` needs an access-controlled independent evaluation protocol. `audit_roster` returns G0 HOLD even when metadata syntactically passes.

## Required roster columns

`dataset,image_id,query_id,query_text,source_image_id,source_group_id,image_sha256,split,exposure_status,lineage_status,eligibility,provenance_evidence_sha256,evaluation_access`

- `query_id` should not encode GT: avoid `POS__...`, `HARD__...` and other label-derived identifiers in inference.
- `split`: `TRAIN`, `DEV`, `CAL`, `FINAL_SEALED`, `EXPOSED_REFERENCE`, `UNASSIGNED`.
- `eligibility`: `COUNT_ELIGIBLE` (count GT genuinely defined), `PRESENCE_ONLY` (only presence GT), `INELIGIBLE`, `UNVERIFIED`.
- Labels including present/absent, `count_target`, annotation provenance, and exact-count eligibility evidence should live in a separate **offline evaluator** key table, accessed after all arm outputs are frozen. Do not put GT in the inference manifest; shared clean roster must not embed label values.
- `image_sha256` and `provenance_evidence_sha256` must be content hashes from actual files, not filled with placeholders.
- All query rows for the same source_group must remain in the same split. Matching only image IDs or byte hashes cannot rule out transformed duplicates.

## Arm names & budgets — blocking agreement

Thúy's T12 requested **A0/A1/A3/A4/A5**. Existing `countlab.method.baselines.run_baseline_arm` defines **B0..B9** with *different historical meanings*. Likewise research brief B5/B6/B7 are protocol names, not verified aliases to Thúy's A arms. **No automatic aliasing or matching by position.** For each T12 arm freeze `implementation_id`, `method_commit`, `unary_budget`, `pair_budget`, `recovery_budget`, and unit `observed_calls` before comparison. Missing identity evidence stays UNKNOWN (not SAME). A0/A1/A3/A4/A5 must use the same frozen input pool, exact candidate IDs and image/query namespace; no R10 -> q31 synthesis.

Metric definitions to freeze: **AZA** fraction of genuine absent queries with final predicted count zero; **FZR+** fraction of genuine present queries incorrectly predicted zero; **Present MAE** only on genuine present, count-eligible queries; **Overall MAE** uses predeclared `evaluation_fallback_count` for abstentions; report **abstention rate** separately. Presence-only labels cannot contribute to MAE. Costs must be **observed** model invocations, tokens and actual wall/VRAM/cost accounting as applicable, with model/prompt/revision/source provenance; a CPU replay or masking evidence slots is not evidence of GPU savings.

Calibration is CAL-only or pre-registered; no selection on historical exposed TEST. Candidate pool and paired roster hashes frozen. Any policy threshold chosen by looking at DEV performance invalidates clean DEV claim.

## How to run

Copy patch folders into repository root; run:

```powershell
python -m pytest -q tests/test_thu_t12_contract.py
python -m countlab.thu_t12_contract --roster path/to/roster.csv --contract path/to/scoring_contract.json --out reports/thu_t12/preflight_001.json
```

This CLI never accesses labels or predicts counts. If inputs/contract are incomplete, it raises a blocking status rather than inventing eligible data. Never auto-merge technical PASS with G0 certification.

## Outstanding deliveries

1. **C1:** Source-disjoint new roster & provenance-evidence SHA with exact image/query mapping; GT file keyed by neutral IDs held only in offline evaluator; confirm metric denominators, abstain fallback, budgets.
2. **C2 / Lộc:** Full frozen T04 candidate pool, original candidate IDs, image SHA256, checkpoint/run manifests, observed Qwen raw/evidence refs, measured costs, and requested q31 PAIR_IDENTITY only under its exact namespace.
3. **C3 / Thúy:** Exact definition of T12 A0/A1/A3/A4/A5, whether B5/B6/B7 appear as policy comparisons, scorer schema, paired roster/evidence keys, and final selected budgets. Do not relabel already exposed samples as clean DEV.

**Release decision:** T12 design/preflight can proceed now on exposed technical pilot; clean DEV/FINAL effectiveness stays HOLD until data and provenance independently pass.
