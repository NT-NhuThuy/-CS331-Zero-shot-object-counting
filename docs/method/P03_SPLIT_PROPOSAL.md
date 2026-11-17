# P03 — UNICOUNT Source-Disjoint Split Proposal

## 1. Objective

Define a provenance-controlled TRAIN/DEV/CAL/FINAL_SEALED
protocol for UNICOUNT.

This document is a proposal, not an approved dataset freeze.

## 2. Historical Exposure Registry

The current audit includes:
- 25,301 historical image-query pairs.
- 13,190 declared source-image IDs.
- 16 byte-identical duplicate groups within FSC147.
- 111 shared hard-negative pairs between Thu and Loc.
- Fresh126 fully contained in Manual160 by image-query key.
- Thuy E00-E10 image rosters pending verification.

All inspected historical pairs remain EXPOSED_REFERENCE.

## 3. Source Grouping

Split at source-image-group level, never query level.

The initial source_group_id is a provisional byte-SHA256
cluster derived from the verified image ledger.

Before split approval, reconcile:
- Same original image under different IDs.
- Re-encoded or cropped duplicates.
- Shared images across research members and experiments.
- Dataset versions, acquisition sources and historical exposure.

Byte-identical grouping alone does not certify independence.

## 4. Proposed Split Roles

TRAIN:
Used only for permitted model training.

DEV:
Used for algorithm development and architecture selection.

CAL:
Source-disjoint from TRAIN and DEV.
Reserved for risk/threshold calibration after design freeze.

FINAL_SEALED:
Independent, locked and inaccessible for tuning.
Opened only under the agreed final evaluation protocol.

EXPOSED_REFERENCE:
Previously used or evaluated historical evidence.

UNASSIGNED:
New candidate source whose eligibility is not yet established.

No automatic allocation to the four evaluation splits
is permitted before provenance review.

## 5. Required Manifest Fields

- dataset and dataset_version
- source_image_id and source_group_id
- source_image_sha256
- query_id and protocol_id
- artifact_sha256
- exposure_status and lineage_status
- proposed_split
- GT availability and exact-count eligibility metadata
- source path or immutable source reference
- experiment/run IDs when applicable

Actual GT count values must not enter inference manifests.

## 6. Required Approval Checks

1. Source identities and lineage verified.
2. No source-group overlap across assigned splits.
3. Historical exposure registry reviewed by all owners.
4. CAL independent from model and algorithm development.
5. FINAL_SEALED roster and access policy frozen separately.
6. Source/protocol hashes recorded before evaluation.
7. No tuning on FSC147 TEST, prior V14 holdouts or
   historical hard-negative diagnostic results.

Unknown provenance must remain UNKNOWN or UNASSIGNED,
never be silently treated as UNEXPOSED.

## 7. Current Decision

Historical metadata and byte-level integrity: PASS.

Independent source-disjoint UNICOUNT split: NOT ESTABLISHED.

G0: HOLD.

P06 may proceed with interfaces, mock fixtures and CPU
unit tests. Risk-head training/calibration and final
performance claims require approved data protocols.
