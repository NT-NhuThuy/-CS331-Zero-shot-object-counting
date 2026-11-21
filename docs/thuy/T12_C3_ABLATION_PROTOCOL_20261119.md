# UNICOUNT v2 — T12 C3 Paired Ablation Protocol

**Date:** 2026-11-19
**Owner:** Thúy / C3
**Status:** C3 DESIGN SNAPSHOT — NOT TEAM-APPROVED OR DATA-FROZEN
**Git HEAD:** `UNAVAILABLE`
**Working tree changed entries:** UNKNOWN
**Current reported regression:** 302/302 PASS

## 1. Scientific objective

Determine whether physical-instance reasoning contributes improvements beyond simple dual-NO filtering, baseline NMS, and fixed-schedule detect–verify methods.

Do not infer superiority from unit tests, archived V14 outcomes, or synthetic evidence.

## 2. Frozen-comparison arms

| Arm | Mechanism | Primary interpretation |
|---|---|---|
| A0 | Frozen CountGD++ B0 | Original detector count |
| A1 | V14-style dual-NO rejection-only | Simple verification correction |
| A2 | v2 B5 NMS/complete-link | Simple identity baseline |
| A3 | M7 dual-channel rejection-only | Same operational point-count rule as A1; compare trace and decision semantics |
| A4 | M8+M10 guarded identity | Dual-NO, baseline coalescing, rescue admission and conflict handling |
| A5 | T09 strict | Provisional positive or abstention with explicit fallback |
| A6 | v2 B6 fixed action schedule | Same C1/C2/C3 interfaces without active feedback |
| A7 | v2 B7 active feedback | C1 chooses actions using count-impact/risk hints |

A3 must not be claimed as an independent count-accuracy improvement over A1 when both run the same correction rule.

## 3. Paired evaluation rules

For A0–A5, use identical source-image/query pairs, frozen baseline proposals and the same actually observed evidence set wherever the arm supports it.

Never fabricate SAME/DIFFERENT, substitute IoU for observed identity, or synthesize missing verifier answers.

For A6 versus A7, different action sequences are permitted only under the same starting candidate pool, eligible actions, model/prompt revision and measured cost budget.

Keep original CountGD++ fallback, operational count, strict status and forced evaluation count in separate output fields.

When operational count is zero, evaluate false-zero risk; never label it certified absence.

Record resource-limit and approximate-search outcomes separately from exact search outcomes.

## 4. Metrics and hypotheses

Co-primary: Absent Zero Accuracy (AZA), Present MAE.

Safety: false-zero rate on positive queries (FZR+); non-inferiority threshold to be approved and frozen before DEV analysis.

Secondary: overall MAE, AFM, abstention/coverage, selective error, rescue additions, baseline merge reductions, false merges/splits where independent instance annotations permit, action counts, tokens, GPU cost and latency.

Hypothesis H1: A4 adds useful counting or identity fidelity beyond A1/A2 under identical evidence, without unacceptable positive false-zero risk.

Hypothesis H2: A7 improves on A6 at matched measured verification cost.

Hypothesis H3: executed evidence-dependency traces are faithful to actual count-changing operations, not explanations attached afterward.

Report source-image-clustered paired uncertainty intervals and both regressions and wins.

## 5. Data and evaluation restrictions

Historical 54 V14 verification-task queries are exposed method diagnostics only, not clean DEV/CAL/FINAL.

R10 FP32 P02 and historical q31 belong to different candidate namespaces and must not be silently joined.

No GT may enter the candidate manifest, C3 solver, router or action-selection logic.

Source-disjoint DEV/CAL/FINAL, eligibility rules, instance annotation availability and evaluation matching protocol need approval from C1/C2 before paper claims.

Do not tune policy using FINAL or the historical exposed V14 cohort.

## 6. Required run outputs

For each query and arm: original candidate IDs, input run/checkpoint hashes, observed evidence references, prediction status, operational count, strict count, original fallback, selected action history, cost provenance and errors/limitations.

Keep model/prompt metadata in evidence execution logs referenced by evidence_ref.

Use a new evaluation run ID when pool, policy, model, prompt or split changes.

## 7. Current status

C3 physical-instance reasoning and synthetic tests: PASS.

Real C2–C3 PAIR_IDENTITY q31 evidence: pending Lộc result.

T04 frozen shared pool and T05 exact candidate join: pending handoff.

B5/B6 matched-budget execution: NOT DONE.

Independent DEV/CAL evaluation: NOT DONE.

Team-approved preregistration: NOT DONE.

## 8. C3 implementation hashes

| File | SHA256 |
|---|---|
| `src/countlab/method/joint_solver_v2.py` | `e1cad9113eb81392dcfe039cc2ba2ea9fa2a0ecd50144bf764dbc30e7b902135` |
| `src/countlab/method/v14_rejection_shadow.py` | `ae7853ad5ae43d17f6ec8a9237a41b47db2774a67674564f25c3ac021d3a0fa0` |
| `src/countlab/method/c3_dual_channel.py` | `2a45fe71bce90594ac59dc5db7f7a77dec09b92cfbbab1c853447bad5e6eca8e` |
| `src/countlab/method/c3_rescue_arbitration.py` | `81d78354d6349c287bb31c2a93d19c2b99faf39910683364db11c3692d4b3a37` |
| `src/countlab/method/c3_evidence_refinement.py` | `02edde7ad1bcba25a3ae5ec50244df25b7dd4d7f4bd8630d2a9393463bed65fd` |
| `src/countlab/method/c3_count_impact_hints.py` | `04b0e39ff26f9a23411f08f9f541dc259f934419bd3b89a97f676f4f2f7f6b44` |
| `tests/test_thuy_m8_m9_count_impact.py` | `ddcb7c5eb0ead853aa19159b044f7dec152b31e17225d401ac50078b00fb5d9c` |
| `tests/test_thuy_m10_refinement.py` | `4ac76876d9232fcb1eb4167a2bf7f2f1a6752723a9a9b281ed20f8f746f36306` |

## 9. Go/no-go

GO for diagnostic replay when shared candidate IDs and real evidence are verified.

GO for matched DEV evaluation only after dataset split, budgets, policies and metrics are approved.

NO-GO for accuracy superiority or A* claims based on synthetic PASS alone.
