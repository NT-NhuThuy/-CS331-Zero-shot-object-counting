#!/usr/bin/env bash
set -euo pipefail

cd /mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC

PYTHON_BIN=${PYTHON_BIN:-countgd_venv_pc152/bin/python}
SNAPSHOT=target_aware_counting/output/molmo_point_a_20261030/model_188130f
REVISION=188130f961c8e0888a34e11121a1423c461a01ba
RUN_DIR=target_aware_counting/output/molmo_point_a_dev_cohorts_20261031
REFERENCE=${REFERENCE_PILOT:-target_aware_counting/output/molmo_point_a_20261030/search_gpu152_lvis1_beam2.json}
CACHED=${CACHED_PILOT:-target_aware_counting/output/molmo_point_a_20261030/search_gpu152_lvis1_beam2_cached.json}

test -x "$PYTHON_BIN"
test -f "$SNAPSHOT/config.json"
test -f "$REFERENCE"
test -f "$CACHED"
"$PYTHON_BIN" -c 'import transformers; assert transformers.__version__ == "4.57.1", transformers.__version__'
"$PYTHON_BIN" -m target_aware_counting.molmo_point_method.compare_cached_reference \
  --reference "$REFERENCE" --cached "$CACHED"

mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

for cohort in lvis_dev300 fsc_primary_dev100 fsc_verified_absent111; do
  output="$RUN_DIR/${cohort}_cached_beam2.jsonl"
  resume=()
  if [[ -f "$output" ]]; then
    resume=(--resume)
  fi
  skip=()
  if [[ "$cohort" == lvis_dev300 ]]; then
    skip=(--skip-item-id lvis_dev:233)
  elif [[ "$cohort" == fsc_primary_dev100 ]]; then
    skip=(--skip-item-id fsc_val_primary:914)
  fi
  "$PYTHON_BIN" -m target_aware_counting.molmo_point_method.run_cohort_cached \
    --cohort "$cohort" --snapshot "$SNAPSHOT" --revision "$REVISION" \
    --output "$output" --device-map cuda:0 --dtype bfloat16 \
    --max-new-tokens 4096 --beam-width 2 --max-language 4 \
    --max-spatial 8 --guided-slots 4 --max-candidates 256 \
    --gospa-cutoff-pixels 32 --tail-mass-epsilon 1e-6 \
    --fail-fast "${resume[@]}" "${skip[@]}" \
    2>&1 | tee -a "$RUN_DIR/${cohort}.log"
done

echo "Inference finished for available items. LVIS item 233 and FSC primary item 914 remain missing; do not finalize until resolved."
