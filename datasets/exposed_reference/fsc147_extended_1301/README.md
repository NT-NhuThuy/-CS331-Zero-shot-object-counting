# FSC147 extended 1301 (EXPOSED_REFERENCE)

**Source of truth:** `legacy/thu/generated/fsc147_presence_v4/test_pairs_1301.csv` (do not duplicate). The 111 absent pairs appear also in `legacy/loc/data/verified_hard_negatives_111.csv` with 111/111 matching pair IDs and normalized image-query keys. This is ONE historical cohort.

- 1,190 FSC147 TEST present image-query pairs, each with 3 exemplars historically.
- 111 manually audited hard negatives (`GT=0`) from 96 FSC147 source images, with 0 exemplars historically. Hard negatives were mined from CountGD++ failure cases.
- Positive/absent protocol differs: do not interpret aggregate historical MAE as a matched new text-only zero-shot benchmark.
- No original FSC147 image bytes are distributed in this Git repository; consult official dataset source/version and verify hashes.
- `annotation_absent111_empty.json` should not be reinterpreted as empty image-wide annotations: negatives are **target-query-specific**.
- Historical models and thresholds have already observed these pairs. Allowed: diagnosis, error taxonomy, software smoke and historical replay; forbidden: newly claimed untouched final evaluation or hidden threshold tuning.

**Frozen historic comparison** (recomputed in `legacy/thu/outputs/joint_semantic_counting_v1/p17_01_three_view_paired_v1/three_view_metrics.csv`): CountGD++ present MAE 11.6504, absent MAE 62.5225; Joint V3.2+A3 present MAE 10.1488, absent MAE 46.2394 and absent exact-zero 39.64%. These are **exploratory, not UNICOUNT v2 results**.
