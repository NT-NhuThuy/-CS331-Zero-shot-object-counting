# ZOC Research Snapshot

This directory is the lightweight, version-control-friendly research mirror of
`ZOC`, intended to be pushed as a GitHub repository. It contains only source
code, tests, contracts, paper-facing documentation, experiment manifests, and
small, curated evidence summaries. It deliberately excludes downloaded data,
checkpoints, images, raw predictions, model traces, runtime logs, and complete
experiment output directories.

The canonical research documents and implementation are mirrored under
`branch/`. `results/` contains compact, immutable paper-evidence records; it
is not a cache for generated artifacts. The active validation direction is
Safe-IIC / FQIE: semantic correctness, whole-instance evidence, cross-view
identity consistency, abstention, and interval counts.

## Repository layout

- `branch/` — active FQIE implementation, contracts, tests, and paper plans.
- `target_aware_counting/` — reusable research modules and experiment drivers;
  generated `output/` is intentionally absent.
- `src/` — package-oriented research code.
- `experiments/` — compact manifests, commands, and summary metrics.
- `results/` — curated claim-gate and dev/holdout evidence only.
- `docs/`, `reports/`, and root Markdown files — design decisions and history.
- `configs/`, `scripts/`, `tests/` — reproducibility support.

To refresh the mirror from the working tree after the shell backend is
available:

```bash
bash SYNC_FROM_ZOC.sh
```

The source tree remains the working copy. Re-running the sync intentionally
does not restore raw outputs into this repository.
