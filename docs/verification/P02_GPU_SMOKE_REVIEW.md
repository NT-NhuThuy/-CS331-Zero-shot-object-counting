# P02 — code review and real WSL GPU smoke (2026-11-15)

| Evidence | Result |
|---|---|
| Software contract / complete CPU suite | **PASS — 114 tests** |
| CUDA tensor + compiled deformable-attention forward | **PASS** |
| Real model GPU smoke | **PASS — one fixed image/query, executed twice** |
| V14 count parity | **NOT MEASURED** |
| V14 bbox parity | **NOT MEASURED** |

**GO for this engineering smoke; HOLD for V14 parity. NO-GO for claiming V14
equivalence, accuracy improvement, generalization or benchmark-level timing.**
Prepared for an owner-approved **Draft PR**, without merge authorization. The
integration base remains `2d8703125dbc19d9b435892f7d08f0298adf54a4`.

## Evidence labels in the immutable artifacts

The candidate manifest retains `backend_provenance.evidence_level =
GPU_PATH_UNVALIDATED`. This is the adapter constructor's historical label for
the real-engine path: it is assigned before inference and is not automatically
updated after a successful run. It is **not a per-run completion status**.
We preserve the experimental manifest byte-for-byte instead of relabeling it.
Its SHA256 remains
`e57594ecf3e4d5c84469a3f1543a9b3d24f3999ccbb308d8b947c89b5d115020`.

The separate smoke summary records `gpu_smoke = PASS` and
`evidence_level = REAL_GPU_SMOKE_NOT_V14_PARITY` after the smoke runner completes
two actual detector calls and exports their outputs. This describes the
**observed fixed image/query execution**, supported by recorded raw hashes,
boxes, scores and resource measurements; it does not certify all GPU inputs
or establish count/bbox parity with V14. One unique pair was executed twice.

Thus the manifest's conservative constructor label and the report's run-specific
smoke result describe different scopes. Neither is upgraded to V14 validation.
This clarification changes documentation only; neither frozen GPU fixture,
its summary, local raw predictions nor protocol/asset hashes are changed.

## Review findings and changes with technical reasons

1. **Source identity:** runtime verifies the pinned official commit
   `2ecbc4331d6730d017f52a4a97bbcbae7d5d714a`, clean tracked source and six exact
   source Git blobs. A separate sparse checkout contains code only; the old
   PrACo checkout is not changed. Only the upstream pseudo-selector function
   is executed through AST, not its annotation-reading evaluation module.
2. **Checkpoint loading:** SHA256 is checked before the preserved loader loads
   the existing local checkpoint. The loader's explicit state-dict mismatch
   checks remain unchanged. It uses `torch.load(weights_only=False)`, so this
   remains a trusted-serialized-asset boundary: a digest locks bytes but does
   not prove an author supplied them. This previously used mirror checkpoint
   is labeled **author equivalence unverified / V14 correspondence not established**.
   No checkpoint was downloaded, edited or silently substituted.
3. **Pseudo selection:** real source helper, strict `>0.23`, full-token max
   sigmoid, descending `torch.sort`, at most 3, resized-pixel xyxy, second forward
   even for an empty selection. Independent offline replay of the actual first
   output exactly reproduced all three selected pseudo boxes.
4. **Geometry and scores:** the final output contains 16 original-image pixel
   boxes, real max-sigmoid scores and native query indices. Raw-index filtering,
   scores and coordinate conversion were independently recomputed from saved
   raw output and matched exactly. A selected nonphysical box fails rather than
   manufacturing an instance. No threshold/preprocessing/prompt/seed/precision
   was changed after seeing predictions.
5. **GT separation:** the fixed pair was chosen from a public upstream example
   before model execution: image `336.jpg`, query `strawberry`. No annotation,
   count or presence-label file was read for inference or adjustment. The
   source helper receives a constant query-class index, not a dataset label.
   Raw output and final candidate manifest pass the existing GT-blind schema.
6. **Compatibility:** no tracked change in `legacy/`, native NO-PSEUDO, shared
   schema/method/router, regime locks, or `FINAL_SEALED`. Existing native paths
   still pass their original regression tests. No global package install,
   source reset, model rewrite or mutation of the old WSL environment was used.

Two production changes were justified by this review, without changing model math:

- `online.py`: reject a supplied official provider if its **protocol hash**
  differs from the requested config. Previously only the protocol ID was
  checked, allowing a reused provider loaded with different asset locks.
  Added a regression test that fails before any forward.
- `countgd_official.py`: record original and resized dimensions in the trace,
  making pseudo-coordinate provenance independently auditable. Extended tests.

`scripts/p02_gpu_smoke.py` is a new create-only, offline smoke runner. It captures
both actual detector outputs unchanged, exports candidates/prediction/config
and hashes, and writes FAIL diagnostics if execution fails. A small genuine
GPU fixture and sanitized summary were added; full raw outputs and all local
paths remain ignored. No synthetic CPU test was relabeled GPU parity.

## Verified runtime / assets

- WSL Ubuntu 24.04, existing `~/praco-venv`; RTX 5060 Laptop, 8,151 MiB,
  driver 591.74, CUDA capability `(12,0)`.
- Torch `2.9.1+cu128`, torchvision `0.24.1+cu128`, transformers `4.39.1`,
  timm `0.9.16`, NumPy `2.5.4`, Pillow `13.0.0`; CUDA runtime 12.8.
- Runtime flags: matmul TF32 false; cuDNN TF32 true; cuDNN benchmark false;
  deterministic algorithms false. Seed 42, float32, AMP off, batch 1,
  initial exemplars 0, pseudo 3, no recovery/Qwen/identity/crop/TT-Norm/NMS.
- Checkpoint SHA256:
  `b89395d79711c9b9f13f3ef80cd2e528386fb1d70453f99443be4eccab33592f`.
  The small `.pth` symlink resolves to the existing 1,250,122,586-byte HF blob.
- BERT config, vocab, tokenizer files and `model.safetensors` are local, fully
  hashed in the lock. Actual token IDs for `strawberry .`: `[101,16876,1012,102]`.
- Compiled-op binary SHA256:
  `cfed81c8b0b0ade63907ba8c4796beb2de046ba2378a44fb7518f979955a5650`.
  All 13 tracked build-source files, including the `.cuh` kernel header, were
  compared with pinned upstream: only two `AT_DISPATCH_FLOATING_TYPES(value.type(), ...)` sites differ, using
  `value.scalar_type()` for Torch compatibility. Patched CUDA file SHA256:
  `c8918c5222fb7e74ac040f7f6cca549d617e3ea15ad498396852b3bb79bb6b9b`.
  This existing binary was reused and hash-locked, not freshly rebuilt or
  certified bitwise equivalent to an author/V14 build. The known 2x2 CUDA
  operator check returned exactly `[6,7,8,9]`.
  The initial smoke lock records 12 source files; the supplementary complete
  tracked-tree manifest adds the kernel header without altering that lock or
  prediction artifacts. Complete manifest SHA256:
  `2e4bfed286c13f4603f33f99ab865eed2038fb20b2e05d87eab27fe4232f3799`.
- Complete local config, with no placeholder:
  `.venv/p02_gpu/local_config.json`; protocol SHA256
  `d0ff6f37830bf3a8ac535414dda47b9e9c0ed6a49c77583fdc192f1b690a6213`.
  Its preflight returns `READY_FOR_GPU_SMOKE_NOT_MODEL_PARITY`, no blockers.
  The repository example stays a blocked template for unknown V14 assets.

## Measured result and artifacts

- Image: `336.jpg`, original **458x384**, resized **954x800**.
- Image SHA256: `aea3767562c09cc516f972743428152d6c796394624f68e4a9f5507394bae2c9`.
- **16 predicted proposals**, confidence range **0.8570859432–0.9615222812**.
  This is an observed prediction, not a GT accuracy statement.
- Both stages return 900 detector queries; exactly **2 detector calls**, 3 pseudo boxes.
- Attempt 1: model load/checks 18.070595 s, instrumented inference 2.285461 s.
- Final attempt after guard/provenance changes: model load/checks **14.056251 s**,
  instrumented inference **1.929466 s**, total smoke **16.568024 s**.
- Peak allocated VRAM: **1,498,123,776 bytes = 1.39524 GiB**, measured with
  `torch.cuda.max_memory_allocated`. This excludes CUDA context/NVML reserved
  memory. Timing includes preprocessing, synchronization, CPU transfers and raw
  capture; it is not a warm steady-state throughput benchmark.
- The same fixed pair was repeated only to validate the reviewed code change;
  both raw frame hashes are identical across attempts:
  - First: `485867fca6e617308e66113f648158b0a11d71c24ff0eb28e497fc4095ec283c`.
  - Second: `7a10bd45d93e759ba271f410a50dca4cb3adeeca079f353ba1888644b1c43af0`.

Reviewable small artifacts:

- `fixtures/countgd_official_gpu_smoke.json`: exact final GT-blind candidate
  manifest, 16 boxes/scores/provenance, no raw photo or personal path.
- `fixtures/countgd_official_gpu_smoke_summary.json`: sanitized runtime,
  measurements, hashes and explicit **V14 parity NOT MEASURED**.
- Local full artifacts: `.venv/p02_gpu/smoke_attempt_02/` contains
  `frozen_candidates.json`, `prediction.json`, `raw_detector_predictions.json`,
  `smoke_summary.json`, `input_request.json`, `config.lock.json`, `preflight.json`
  and `SHA256SUMS.json`; logs/source/op manifests remain in `.venv/p02_gpu/`.

CPU rerun after review: **114 passed in 1.69 s**; the later Draft PR preparation
rerun also passed **114 tests in 2.59 s**, including the original legacy artifact
audits. The new fixture test validates recorded data/schema/hash,
not another GPU run or model accuracy.

## Reproduce without changing the working environment

Run from this worktree's root **inside WSL**, using the already generated lock
and its existing source/assets. Each output directory must be new.

```bash
set -euo pipefail
REPO="$PWD"
PY="$HOME/praco-venv/bin/python"
CFG="$REPO/.venv/p02_gpu/local_config.json"
IMG="$HOME/PrACo_Local/input/FSC147_384_V2/FSC147_384_V2/images_384_VarV2/336.jpg"
OUT="$REPO/.venv/p02_gpu/reproduce_$(date -u +%Y%m%dT%H%M%SZ)"
export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
"$PY" -m countlab.integrations.preflight --config "$CFG"
"$PY" scripts/p02_gpu_smoke.py --config "$CFG" --image "$IMG" \
  --query strawberry --image-id p02-smoke-336 --query-id strawberry-verbatim \
  --out-dir "$OUT"
```

Do not replace a mismatching hash with an observed one to bypass preflight.
For V14, obtain its own lock/raw artifacts first. Exact request for Thúy/P01:
`docs/verification/P02_V14_ARTIFACT_REQUEST.md`. PrACo FULL counts, this one-pair
smoke and V14 are three distinct evidence scopes.
