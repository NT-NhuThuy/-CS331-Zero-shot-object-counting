# Paired presence/count protocol v2 (proposal)

## Primary research regime

**Target text query is available for all image-query pairs.** The preferred new primary experiment is a *matched text-only* open-world query regime (same available prompt information to present and absent queries); do **not** mix FSC147 legacy present=3 human exemplars with absent=0 human exemplars in the primary new comparison. Official pseudo3 auto-prompting can be a frozen internal CountGD++ operating point if compatible and parity is verified on BOTH arms, but it is not equivalent to supplying 3 GT exemplars.

Maintain explicitly labeled alternative regime(s): native single-pass/no-pseudo, and exemplar-assisted where legal/available. Results from unlike regimes are never merged into a single baseline MAE without stratification.

## Data unit and labels

One unit is `(source_image_group, target_query, protocol_id)`. `presence_label=0` implies `count_target=0`; `presence_label=1` implies strictly positive count if an exact-count annotation exists. Category mismatch in FSC147 is NOT proof of absence. Annotator/reviewer evidence and ambiguity status must be tracked.

For an image with many queries, keep all queries in one source group/split. Hard-negative mining from CountGD errors must be documented and cannot be used to estimate unbiased population prevalence. In addition to target-absent negatives, use present scenes with hard distractors and similar semantic categories.

## Stage policy

- **EXPOSED_REFERENCE:** historical FSC147 1301, loc fresh160/fresh126, previously exposed experiments; permitted for failure diagnosis and integration pilot only.
- **TRAIN/DEV:** newly approved source-independent allocation; TRAIN fits model, DEV selects design.
- **CAL:** source-disjoint from TRAIN/DEV, reserved for predeclared calibration after design selection.
- **FINAL_SEALED:** independent source groups, frozen before study, not accessed for feature development, threshold choice, action design or CI troubleshooting.

## Primary reports

1. Positive-only MAE/RMSE and false-zero rate: `P(Nhat=0 | GT>0)`.
2. Absent exact-zero accuracy `P(Nhat=0 | GT=0)` and false-positive mass/average nonzero count.
3. Overall MAE under a predeclared positive/absent mixture; additionally show each stratum, because mining changes prevalence.
4. Present-positive instance region recall/precision, and identity false-merge/split when annotation supports it.
5. Abstention coverage, prediction fallback, measured compute budget (Qwen calls/tokens and GPU latency/VRAM).
6. Paired source-cluster bootstrap confidence intervals. Do not treat images/queries with shared sources as independent.

Current historical cohort and baseline summary are for exploratory comparisons only. No unverified source becomes `FINAL_SEALED` through a file move.
