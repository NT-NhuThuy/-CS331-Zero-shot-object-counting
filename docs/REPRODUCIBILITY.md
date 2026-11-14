# Reproducibility / strict research protocol

## Immutable inputs

Git commit SHA + model/checkpoint SHA + image manifest checksum + query
manifest checksum + source-image IDs + prompt/vocabulary + preprocessing +
exemplar count + config + budgets + decision logic + inference hardware +
random seeds. Require SHA for external model artifacts before submitting.

## Unit of independence

An `image–query pair` is NOT necessarily an independent observation if the
same source image is queried multiple times. Bootstrap by source-image ID,
or by an explicit source cluster to prevent falsely tight confidence intervals.

## Exact split discipline

- Development used FSC-147 TEST and 111 hard-mined negatives for Thu and Lộc.
  These are **exposed and exploratory**, regardless of SHA locks.
- PrACo Test1 34,510 class-pairs is a separate native CountGD++ baseline.
- V14 LVIS-derived 180-pair result is supported by compact metrics/hashes in
  this mirror; its sealed raw predictions and independent split/provenance
  proof are not present in Git, so verify original artifacts on server.
- The mirror carries files named `20261116`, **later than local date
  2026-11-14**. Treat embedded dates as author-provided labels, not evidence
  that a future validation has independently occurred.
- Changed core methodology after looking at holdout => create independent
  validation and NEW test split; never rename a revised V14 as frozen.

## Distinguish 3 settings

1. LVIS query-only category counting (instance-level labels/boxes available).
2. FSC-147 positive and mined absent (exemplars differ; often counting
   annotations do NOT provide full instance masks).
3. PrACo Test1 class mismatch (mismatch may still exist in the image).

## Mandatory testing

- baseline parity on matching checkpoint, prompts, preprocessing, and inputs;
- actual GT-blind inference; seal predictions before GT is loaded;
- complete full-cohort fallback and results for zero selected rescues;
- source disjointness and dedup across train/dev/val/test;
- paired cluster-aware CIs, sign tests with dependence caveats;
- present + absent + prompt-mismatch + large-count strata;
- cost-matched baselines, calls/latency/VRAM;
- external dataset replication and third-party baselines.
