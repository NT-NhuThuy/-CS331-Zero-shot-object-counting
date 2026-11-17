# Split authorization — no final split established

Current approval: **G0 HOLD**. Never assign a public historical pair to FINAL_SEALED just by renaming its CSV or folder.

Suggested roles:
- `EXPOSED_REFERENCE`: historic FSC147 TEST/absent111, prior V14 and other studied cohorts.
- `TRAIN`: model fitting only after source-group disjointness approval.
- `DEV`: engineering/model selection and ablation.
- `CAL`: held out from training and model-choice development for threshold/risk calibration.
- `FINAL_SEALED`: independent, frozen, protected; inference/evaluation only once under agreed protocol.
- `UNASSIGNED`: origin/review unknown; **not** automatically train/dev/final eligible.

Keep grouping by original-image family and visual near-duplicates; byte SHA alone is insufficient. Multiple target queries for the same image must stay in one group/split. Record annotation review status separately from GT-count availability. Store GT in evaluation-only manifests, never inference candidate JSON. Refer to `docs/method/P03_SPLIT_PROPOSAL.md`.
