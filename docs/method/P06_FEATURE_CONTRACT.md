# P06 — GT-Free Risk Feature Contract

## 1. Scope

P06 extracts candidate-level diagnostic features from
CountLab candidate pools using frozen V4/A3 node features.

Current status: SOFTWARE VALIDATION ONLY.

G0 remains HOLD. No risk-model training, calibration,
threshold selection or final evaluation is performed.

## 2. Input Contract

Candidate pool schema:

    countlab_candidate_pool_v1

Required manifest fields:

- image_id
- query_id
- query_text
- baseline_count
- baseline
- rescue
- identity_edges

Each candidate has:

- candidate_id
- view_id
- box_xyxy in original-image pixel coordinates
- optional confidence
- evidence and evidence_refs

Candidate role is determined by membership in
baseline or rescue.

P02 is responsible for producing stable candidate IDs
and validating tile-to-original coordinate transforms.

## 3. V4/A3 Exact-Key Mapping

The existing Thu feature adapter uses:

    candidate_id -> (node_graph_id, raw_query_index)

The frozen NPZ must contain exactly:

- node_graph_id
- raw_query_index
- v4_contribution
- a3_score

Rules:

1. Candidate mapping covers every candidate ID exactly.
2. No extra or missing candidate IDs.
3. No duplicate frozen node keys.
4. No join by array order, box IoU or nearest candidate.
5. Missing mappings and invalid values fail explicitly.
6. GT and label-count fields are prohibited.

Candidate ID stability is conditional on the frozen
P02 protocol and all inputs used to derive the ID.

Changes in runtime, checkpoint, preprocessing or
protocol require renewed mapping/parity verification.

## 4. Diagnostic Features

P06 currently extracts:

- Original-coordinate box area
- Box aspect ratio
- Detector confidence
- 1 - detector confidence (diagnostic baseline)
- V4 contribution
- A3 semantic score
- Exact node graph ID
- Exact raw query index

The features remain diagnostic observations.

A3 scores and V4 contributions are not automatically
calibrated probabilities of counting errors.

## 5. Provenance

The output includes:

- Candidate pool SHA256
- Frozen feature NPZ SHA256
- Candidate mapping SHA256
- Source exposure status
- G0 decision

The exact-key join does not establish that a feature
source matches a real candidate pool experimentally.

External provenance, image identity, checkpoint identity
and run compatibility must be verified separately.

## 6. Risk Hint Policy

The current P06 diagnostic adapter does not generate
or calibrate risk_hint.

The existing CountLab router falls back to
1 - confidence, or 0.5 if confidence is unavailable.

That fallback is a screening heuristic, not a
calibrated error probability.

No new routing thresholds or weights are selected
while G0 is HOLD.

## 7. Historical Exposure

The existing frozen FSC147 1301-pair V4/A3 features
are exposed exploratory data.

Historical FSC147 TEST, previously exposed V14
holdouts and unverified-source experiment records
must not be treated as fresh independent evaluation.

P01-D recovered V14 records remain UNKNOWN where
source-image identity has not been verified.

## 8. Current Verification

Synthetic unit tests validate:

- Exact-key feature mapping
- Baseline and rescue coverage
- Candidate role preservation
- Geometry-derived features
- Deterministic output for fixed inputs
- No input mutation
- Missing mapping rejection
- Duplicate mapping rejection
- NPZ schema rejection
- GT leakage rejection
- Invalid boxes rejection
- Unverified exposure handling

Synthetic CPU smoke validates the data flow and
output provenance fields.

These tests establish software behavior only.

## 9. Next Integration Gates

Before real experiments:

1. Lộc freezes P02 baseline+recovery candidate pool.
2. Candidate IDs and original coordinates are verified.
3. V4/A3 features are reconstructed for the exact pool.
4. Candidate-to-node mapping achieves exact coverage.
5. Input artifacts, checkpoint, protocol and runtime
   provenance are recorded.
6. Approved TRAIN/DEV/CAL split is available.
7. Risk model and thresholds are calibrated only
   under the approved data protocol.
8. Independent FINAL_SEALED remains unopened until
   final evaluation authorization.

## 10. Decision

P06 GT-free feature interface: IMPLEMENTED.

Synthetic contract and smoke tests: PASS.

Real frozen candidate-pool parity: PENDING.

Risk calibration: NOT PERFORMED.

G0: HOLD.

No empirical improvement or novelty claim is made.
