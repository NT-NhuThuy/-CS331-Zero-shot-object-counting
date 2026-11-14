# FQIE fair results tables (2026-11-13)

> **Superseding result (2026-11-16):** the tables below remain historical rescue-only evidence. The current bidirectional method is reported in `FQIE_BIDIRECTIONAL_HOLDOUT_RESULTS_20261116.md`: V14 MAE 1.7611 to 1.1000, bootstrap 95% CI [-1.3833, -0.1833], paired sign p=0.0001565.

All values use preserve-baseline fallback for queries without rescue evidence.

| Cohort | Full baseline MAE | Full FQIE MAE | Delta | Rescue queries | Rescue baseline MAE | Rescue FQIE MAE | Absent FP baseline→FQIE |
|---|---:|---:|---:|---:|---:|---:|---:|
| V11 | 3.5167 | 3.4944 | -0.0222 | 6 | 2.0000 | 1.3333 | 1 → 1 |
| V12 | 1.4333 | 1.4278 | -0.0056 | 4 | 1.2500 | 1.0000 | 2 → 3 |

## Interpretation

- V11 shows a small fair improvement concentrated in four rescue changes; its exact sign test is not significant.
- V12 is a near-null replication and increases absent false promotion under fair fallback.
- The defensible contribution is therefore the evidence-grounded protocol and auditability, not a universal MAE superiority claim.

## Statistical guardrail

| Cohort | Paired bootstrap 95% CI for fair MAE delta | Exact sign-test p |
|---|---:|---:|
| V11 | [-0.0444, -0.0056] | 0.1250 |
| V12 | [-0.0278, 0.0111] | 1.0000 |
