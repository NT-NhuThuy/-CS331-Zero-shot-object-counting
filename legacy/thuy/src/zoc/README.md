# Public source package

This directory contains the selected implementation used to make the research archive executable. It is intentionally smaller than the working directory.

## Included

- `backends/countgd_grounding/` — CountGD grounding backend and its focused tests.
- `methods/fqie_v2/` — FQIE v2 implementation and tests.
- `methods/bounded_recovery/` — bounded recovery selector/verifier.
- `methods/reference_correspondence/` — reference/candidate correspondence utilities.
- `evaluation/` — metric entry point and target-aware audit scripts.
- `tests/` — target-aware tests.

The backend's `train.py` and `train_verifier` are archived research utilities only. They are not imported by, or part of, the strict frozen zero-shot inference pipeline. Training a verifier changes the research contract and requires explicit approval and a separate experiment.

## Provenance

These files were copied from the local ZOC workspace on 2026-11-07. The exact source commit and per-file hashes must be recorded before the first public release. Large model files, datasets, virtual environments, caches, and runtime outputs are excluded.

## Release rule

Do not describe a module as a validated final method merely because its source is present here. Validation status belongs to the corresponding experiment record in `experiments/`.
