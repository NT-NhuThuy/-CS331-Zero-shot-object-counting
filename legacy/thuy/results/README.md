# Curated paper evidence

This directory contains only compact, reviewable evidence necessary to support
the current FQIE paper claims. Raw model predictions, image pools, encodings,
intermediate manifests, and runtime logs remain in the working tree and are
excluded from Git.

Retained records:

- `fqie_bidirectional/dev/`: development ablation summaries used for method
  selection.
- `fqie_bidirectional/holdout/`: source-disjoint holdout metrics, paired audit,
  and failure audit.
- `validation/`: claim gates and CPU validation reports.
- `fsc147/`: preflight evidence only. FSC147 test predictions are never stored
  here before the method is frozen.

Every retained result is a small JSON/Markdown summary. Reproduction scripts,
contracts, and tests live in `branch/`.
