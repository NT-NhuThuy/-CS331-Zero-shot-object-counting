# UNICOUNT v2 — decision log, proposed 2026-11-17

## Architecture decisions recommended to RATIFY at the next team meeting

| ID | Decision | Rationale / consequence |
|---|---|---|
| D01 | **One research problem:** class-conditioned zero-shot/open-world counting for both present and absent target queries | Optimize zero correctness while safeguarding positive counting, not a negative-only gate |
| D02 | **One inference method**, not three independent scalar predictions | Shared candidate graph + active evidence + identity inference |
| D03 | `datasets/` is the only entrypoint for dataset manifests, provenance, annotations and split rules | Raw images/checkpoints remain off-Git; labels never enter inference payload |
| D04 | Historical FSC147 extended 1,301, Lộc's same 111 negatives and fresh160 are EXPOSED_REFERENCE | Diagnoses only, never new untouched FINAL_SEALED |
| D05 | Unit of label/absence is **image–target query**, not image alone | Prevent false negative labels for other categories in same image |
| D06 | Preserve `legacy/` read-only; method runtime stays `src/countlab` in phase A | A separate migration PR will move code once all owners approve design |
| D07 | Prioritize matched **text-only** present/absent protocol for new primary comparisons | Historical FSC positive=3 human exemplars vs absent=0 is mismatched; official pseudo3 only after parity |
| D08 | Initial predictions allow **0, positive K, or abstain** with a registered forced-count fallback for MAE | `No box` is not certified zero; protect positives from global veto |
| D09 | Thư C1 risk/action; Lộc C2 evidence/recovery; Thúy C3 joint identity/final decision | One research contribution with 3 mutually necessary submechanisms |
| D10 | G0 stays HOLD: never train/calibrate on exposed FSC TEST, V14 or synthetic software fixtures | Independent source-disjoint train/dev/cal/final needed |

## Unresolved sign-offs (must record before experiments)

- Which exact official CountGD++ pseudo-exemplar checkpoint/protocol/threshold is PRIMARY? Freeze hash and verify both present and absent prompts; no-pseudo must remain a separate comparator.
- How are absent annotations double-reviewed, and which source-family audit defines eligibility for new DEV/CAL/FINAL_SEALED?
- What is the final solver objective, hard identity constraints, and fallback/abstention rule? This is research design, not yet implemented.
- What shared GPU compute budget (milliseconds/FLOPs/calls/tokens) and baseline costs define fair comparison?
- Before examining new DEV, predeclare false-zero/false-presence guardrails, confidence intervals and go/no-go criteria.

All D01–D10 are **recommendations in this handoff**; do not treat them as approvals by Thúy/Lộc without their review.
