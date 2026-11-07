# ZOC — Zero-Shot Open-Vocabulary Counting Research Archive

This repository documents the experiments, evidence, negative results, and future directions of the ZOC project. Licensing is intentionally unresolved pending the provenance review in `third_party/PROVENANCE.md`.

## Research goal

Given an image and an arbitrary text query, count the physical instances that match the query. The evaluation must preserve query dependence, localization, instance identity, and verified-absent safety.

## Current status

No final method has been established. This archive is structurally complete and contains selected source code plus source reports for the main claims. It is not a fresh GPU rerun or a fully portable benchmark release.

Core retained evidence:

- **E00:** frozen CountGD++ baseline and failure analysis.
- **E01:** pseudo-exemplar experiments and cross-domain generalization limits.
- **E02:** candidate recall audit.
- **E03:** A4 crop + NMS fallback on a source-disjoint VAL300 cohort.
- **E04:** B0 SAM3 add-only recovery on the primary100 cohort.

Negative and diagnostic evidence is archived in E05–E10. Future ideas are documented separately and are not presented as completed methods.

## Repository map

- `src/` — reusable project code prepared for public release.
- `configs/` — dataset, baseline, and experiment configuration snapshots.
- `scripts/` — public entry points for inference and evaluation.
- `experiments/` — one reproducibility record per experiment.
- `reports/` — research-level summaries and tables.
- `docs/` — protocol, problem definition, reproducibility rules, and decisions.
- `third_party/` — dependency and attribution notes.

## Evidence policy

Results from the working directory are initially marked `reported_pending_audit` unless their source files, manifests, and logs have been checked. Results from different cohorts or protocols must not be combined into a single comparison.

## Reproducibility

Every experiment should include its setup, exact command, manifest, source commit, checkpoint hash, metric definition, and limitations. Large datasets, checkpoints, runtime outputs, and raw logs are intentionally excluded from Git.
