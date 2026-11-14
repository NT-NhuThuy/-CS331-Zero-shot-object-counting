#!/usr/bin/env bash
set -euo pipefail

SRC="/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC"
DST="/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC_research"

test -d "$SRC" || { echo "Missing source: $SRC" >&2; exit 1; }
mkdir -p "$DST/branch" "$DST/target_aware_counting"

# Preserve research documents and executable source/configuration. Raw output
# trees are intentionally excluded: `results/` is curated manually, so a sync
# cannot accidentally bloat this GitHub snapshot again.
rsync -a --include='*.md' --include='*.py' --include='*.sh' --include='*.json' \
  --include='*.jsonl' --include='*.yaml' --include='*.yml' --include='*.toml' \
  --exclude='__pycache__/' --exclude='*.pth' --exclude='*.pt' --exclude='*.bin' \
  --exclude='*.jpg' --exclude='*.jpeg' --exclude='*.png' --exclude='*.webp' \
  --exclude='*.log' --exclude='images/' --exclude='pool/' --exclude='captures/' --exclude='output/' \
  "$SRC/branch/" "$DST/branch/"

rsync -a --include='*.md' --include='*.py' --include='*.sh' --include='*.json' \
  --include='*.jsonl' --include='*.yaml' --include='*.yml' --include='*.toml' \
  --exclude='__pycache__/' --exclude='*.pth' --exclude='*.pt' --exclude='*.bin' \
  --exclude='*.jpg' --exclude='*.jpeg' --exclude='*.png' --exclude='*.webp' \
  --exclude='*.log' --exclude='images/' --exclude='pool/' --exclude='captures/' --exclude='output/' \
  "$SRC/target_aware_counting/" "$DST/target_aware_counting/"

date -u +%Y-%m-%dT%H:%M:%SZ > "$DST/SYNCED_AT_UTC"
echo "Synced source/docs snapshot to $DST"
