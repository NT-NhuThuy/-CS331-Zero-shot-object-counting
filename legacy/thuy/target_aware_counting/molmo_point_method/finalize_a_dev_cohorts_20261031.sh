#!/usr/bin/env bash
set -euo pipefail

cd /mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC

PYTHON_BIN=countgd_venv_pc152/bin/python
RUN_DIR=target_aware_counting/output/molmo_point_a_dev_cohorts_20261031

for cohort in lvis_dev300 fsc_primary_dev100 fsc_verified_absent111; do
  source="$RUN_DIR/${cohort}_cached_beam2.jsonl"
  predictions_dir="$RUN_DIR/predictions/$cohort"
  freeze_dir="$RUN_DIR/freeze"
  evaluation_dir="$RUN_DIR/evaluation"
  test -f "$source"
  mkdir -p "$freeze_dir" "$evaluation_dir"
  "$PYTHON_BIN" -m target_aware_counting.molmo_point_method.export_a_predictions \
    --cohort "$cohort" --source "$source" --output-dir "$predictions_dir"
  for method in native proper_map proper_mbr guided_map guided_mbr; do
    prediction="$predictions_dir/${cohort}_${method}.jsonl"
    frozen="$freeze_dir/${cohort}_${method}.json"
    evaluation="$evaluation_dir/${cohort}_${method}.json"
    label="$method"
    if [[ "$method" == native ]]; then
      label=a_native
    fi
    "$PYTHON_BIN" -m target_aware_counting.molmo_point_method.freeze \
      --cohort "$cohort" --predictions "$prediction" --output "$frozen"
    "$PYTHON_BIN" -m target_aware_counting.molmo_point_method.evaluate \
      --cohort "$cohort" --predictions "$prediction" --freeze "$frozen" \
      --method-label "$label" --output "$evaluation"
  done
done

"$PYTHON_BIN" -m target_aware_counting.molmo_point_method.summarize_a_dev \
  --evaluation-dir "$RUN_DIR/evaluation" \
  --output "$RUN_DIR/summary.json" \
  --markdown "$RUN_DIR/summary.md"
