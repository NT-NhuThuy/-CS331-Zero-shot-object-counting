#!/usr/bin/env bash
set -euo pipefail
ROOT=/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC
EXP="${1:?usage: RUN_ABLATION_CPU.sh <experiment-dir> [reference-promotions.jsonl] [reference-gated-promotions.jsonl]}"
REF="${2:-}"
GATED="${3:-}"
if [ -n "$REF" ] && [ -n "$GATED" ]; then exec /usr/bin/python3 "$ROOT/target_aware_counting/ablation_matrix.py" --experiment "$EXP" --reference-promotions "$REF" --reference-gated-promotions "$GATED"; fi
if [ -n "$REF" ]; then exec /usr/bin/python3 "$ROOT/target_aware_counting/ablation_matrix.py" --experiment "$EXP" --reference-promotions "$REF"; fi
exec /usr/bin/python3 "$ROOT/target_aware_counting/ablation_matrix.py" --experiment "$EXP"
