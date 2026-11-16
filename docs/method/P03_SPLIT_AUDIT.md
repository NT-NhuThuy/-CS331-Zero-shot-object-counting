# P03 — Source-Disjoint Split & Leakage Audit

## 1. Objective

Audit historical datasets, source-image identities, SHA256 hashes,
and experimental provenance before developing UNICOUNT risk routing.

This audit does not create new dataset splits or use GT for inference.

## 2. Historical dataset results

| Measurement | Result |
|---|---:|
| Historical image-query pairs | 25,301 |
| Declared source-image IDs | 13,190 |
| Images successfully hashed | 13,190 |
| Missing images | 0 |
| Byte-duplicate groups | 16 |
| Cross-cohort byte duplicates | 0 |
| Shared Loc/Thu hard-negative pairs | 111/111 |
| Thuy manifests with image rosters | 0/11 |

FSC147 contains 1,190 present pairs and 111 hard-negative pairs.
The 111 negatives reuse 96 images from FSC147.

All 16 detected byte-duplicate groups occur within FSC147.
No cross-cohort byte-identical duplicates were detected.

Historical LVIS TRAIN, DEV, and TEST have zero recorded image-ID
and filename overlap.

## 3. Fresh-negative comparison

Fresh126 contains 126 rows.

Manual-review160 contains 160 rows.

Shared normalized image-query keys: 126.

Shared image filenames: 126.

Filename comparison alone does not establish image independence
or the validity of absence annotations.

## 4. Cross-member provenance

### Thu

Joint V3.2, Structural V4, and A3 were evaluated using
previously exposed FSC147 TEST and hard-negative data.

These results are historical diagnostic evidence.

### Loc

All 111 verified hard-negative pairs match Thu's 111 pairs
by pair ID and normalized image-query identity.

They are one shared historical cohort, not independent tests.

### Thuy

11 experiment manifests were inspected.

0 provide source-image rosters.

Full source-level exposure tracking remains incomplete.

## 5. Limitations

- SHA256 detects byte-identical files, not re-encoded or cropped images.
- Thuy historical image rosters are incomplete.
- Loc PrACo image lineage still requires verification.
- Historical FSC147 TEST and LVIS TEST are already exposed.
- No independent UNICOUNT FINAL_SEALED cohort is established.
- Presence annotation and exact-count eligibility must be separated.

## 6. Reproducibility

Run the project tests using:

    python -m pytest -q tests/

Relevant implementations:

- src/countlab/split_audit.py
- src/countlab/source_identity_audit.py
- scripts/p03_team_lineage.py
- scripts/p03_finalize.py

Evidence and source hashes are stored in reports/p03/.

## 7. G0 Decision

Historical file integrity: PASS.

**G0 DATASET INTEGRITY: HOLD.**

Before proceeding to independent model evaluation, the team must
verify historical source rosters, establish source-disjoint
TRAIN/DEV/CAL splits, and freeze a genuinely independent
FINAL_SEALED evaluation cohort.

This report does not establish UNICOUNT end-to-end performance
or a state-of-the-art result.
