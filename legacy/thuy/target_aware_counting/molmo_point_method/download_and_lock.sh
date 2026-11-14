#!/usr/bin/env bash
set -euo pipefail

cd /mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC
revision=188130f961c8e0888a34e11121a1423c461a01ba
snapshot=target_aware_counting/output/molmo_point_a_20261030/model_188130f
export HF_HUB_DOWNLOAD_TIMEOUT=60

cgd/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("allenai/MolmoPoint-8B", revision="188130f961c8e0888a34e11121a1423c461a01ba", local_dir="target_aware_counting/output/molmo_point_a_20261030/model_188130f", max_workers=1))'
target_aware_counting/output/molmo_point_a_20261030/runtime_py310/bin/python \
  -m target_aware_counting.molmo_point_method.snapshot_lock \
  --snapshot "$snapshot" --revision "$revision"
