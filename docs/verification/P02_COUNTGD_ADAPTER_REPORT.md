# P02 — Official CountGD++ adapter and parity readiness

**2026-11-15 — SOFTWARE CONTRACT GO / REAL GPU SMOKE GO / HOLD for V14 parity.**
The fixed one-pair GPU smoke passed in WSL2; count/bbox parity against V14 and
benchmark improvement remain unmeasured. See `P02_GPU_SMOKE_REVIEW.md` for the
measurements, code review and the distinction between `GPU_PATH_UNVALIDATED`
and `REAL_GPU_SMOKE_NOT_V14_PARITY`. The owner has approved Draft PR preparation;
merge and any V14 equivalence claim remain outside this approval.

## Input audit and scope

- Base: `research/unified-integration-verified` at `2d8703125dbc19d9b435892f7d08f0298adf54a4`.
- Feature: `feat/loc-countgd-adapter`, in an isolated worktree. No matching remote
  feature branch or P02 PR was found before implementation. The original
  worktree's old untracked research ZIP was preserved.
- Read the root README, team onboarding, interface, prioritized tasks, unified
  audit, online/preflight tests, and all five integration modules requested.
- Only P02 is implemented. `legacy/`, `FINAL_SEALED`, shared method/schema/router,
  regime locks, NO-PSEUDO adapter and its example configuration are unchanged.

## Operating points are different

| Property | Existing NO-PSEUDO | Historical PrACo A | New P02 query-only core |
|---|---|---|---|
| API | `predict(image, query)` | `infer(image, pos_text, neg_text, pos_image, neg_image, pos_boxes, neg_boxes)` | provider `infer(...)`, same runner contract |
| Detector calls | 1 | 2: synthetic, then self-generated exemplars | 2: empty initial exemplars, then up to 3 positive pseudo-exemplars; reruns even if none selected |
| Pseudo selection | none | positive/negative max-score dominance, top 3 of each | pinned upstream `get_pseudo_exemplars`: max sigmoid, strict threshold, descending `torch.sort`, top 3 |
| Final filter/score | max sigmoid + all content tokens above threshold | with negative text: positive score > negative score; otherwise max score; only count/density returned | uncropped official evaluator core: max sigmoid > 0.23 AND all content tokens > 0; actual max sigmoid confidence; no NMS |
| Prompt/tokenizer | verbatim `query + ' .'`, BERT period separator | `inflect` singularization, one filename-specific correction, optional positive/negative caption | verbatim `{query} .`, no singularization; BERT files and caption SHA locked |
| Preprocessing | RGB, `transforms_app`, 800/1333, ImageNet normalization | RGB, PrACo `transforms`, 800/1333, same normalization constants | explicit RGB / `transforms_app` / 800/1333 / same normalization; original-image pixel output |
| Weights/CUDA | local loader, no automatic parity claim | third-party mirror, author equivalence unverified; recorded CUDA compatibility patch | exact checkpoint/BERT/compiled-op SHA and library/runtime locks required; no silent source patch |
| Coordinates | normalized cxcywh converted to original pixels, optional tile offsets | internal normalized boxes, count matrix/density persisted | original pixel xyxy, clipping, raw query index, raw normalized box and offset retained |

PrACo source was read at commit `438a968aef9d31ded7519733bd84e89baae68896`.
Its `models/countgdplusplus_model.py` SHA256 is
`b07365ea348c1bbd09dd0783ab04e635b73e345c9cfef4357e5f8e28252a3dd9`.
The preserved A notebook, A/B ZIPs, roster, protocols and recorded checkpoint
SHA exist in `legacy/loc/`. The two FULL ZIPs contain **34,510 PrACo Test-1
pairs (1,190 FSC TEST images x 29 classes)**, supporting count/policy regression.
They do not contain per-pair raw boxes/scores and cannot certify V14 bbox
parity. Never infer boxes from density centers or rename them V14 baselines.

## Implementation and contract

- `src/countlab/integrations/countgd_official.py`: separate
  `CGD_OFFICIAL_QUERY_ONLY_PSEUDO3_V1`; strict protocol, source and asset checks,
  two-stage orchestration, actual optional torch bridge, original-coordinate
  proposals and measured-call instrumentation. Reuses the preserved model
  loader without changing its source. Loads only the pinned pseudo-selector
  function through AST, **not** the GT-reading upstream evaluator. The helper's
  label argument is a constant query-class index, never a dataset label/count.
- `online.py` / `preflight.py`: explicit protocol selection; existing native
  selection and routing/reconciliation rules retained. No model download or
  GPU execution in preflight. The official example disables recovery, Qwen and
  identity to isolate P02. P04/P07 are not enabled by this change.
- `configs/integrations/official_pseudo3_example.json`: deliberate placeholders
  for unknown V14 assets/versions. Source is pinned to the audited public
  CountGD++ commit `2ecbc4331d6730d017f52a4a97bbcbae7d5d714a`, **not claimed to
  be V14's source commit**. Engine/config/transform/model/CUDA source Git blob
  IDs are pinned; runtime also records SHA256, loader source and device flags.
  Template protocol SHA256:
  `a5c58bce922474d73def2afdf2f644ece184b22359f56e7a20b9cd4aafce921c`.
  Filling assets requires a new canonical protocol digest; hashes cannot be
  borrowed from PrACo to imply V14 provenance.
- `fixtures/countgd_official_software.json` and `tests/test_countgd_official.py`:
  one artificial 100x80 image/query contract, manufactured detector outputs,
  asset/hash-negative tests and NO-PSEUDO regression. No real dataset split or
  inference accuracy is represented by these fixtures.
- `official_pseudo3_parity_plan.json`: proposed tolerances recorded before any
  raw V14 opening: exact per-query counts, raw-query-index box matching with
  absolute `1e-4` original pixels and zero relative tolerance; confidence
  absolute `1e-6`. Team must approve this plan before comparison; never adjust
  it after seeing mismatches.

Output stays `countlab_candidate_pool_v1`: unique deterministic IDs include
protocol/image/query/view/role/offset/raw-index identity; real confidence,
UNKNOWN evidence and evidence references are retained. Additional provenance
is in metadata and optional candidate fields; **no shared schema change**.
Count/density-only or nonphysical selected boxes fail rather than inventing
instances. GT fields are rejected before official inference and in raw output.
Per-view trace contains caption/raw hashes, first/second-stage query totals,
pseudo boxes in resized pixels, detector calls, elapsed seconds and peak VRAM.
Fake engines are marked `SOFTWARE_CONTRACT_TEST`, never actual GPU inference.

## CPU validation

```powershell
python -m pip install -e '.[test]'
python -m pytest -q tests/ legacy/thuy/branch/ legacy/loc/tools/praco_test1/test_verify_praco_artifacts.py
# On this Windows host, use a fresh workspace-owned temp path and disable cache:
python -m pytest -q tests/ legacy/thuy/branch/ legacy/loc/tools/praco_test1/test_verify_praco_artifacts.py --basetemp .venv/pytest-p02-review-UNIQUE -p no:cacheprovider
python -m countlab.integrations.preflight --config configs/integrations/official_pseudo3_example.json
```

Baseline **75 passed**; initial P02 **112 passed**; after GPU review **114 passed**
(39 additional cases). Latest Draft PR full-suite run: **114 passed in 2.59 s**.
Python 3.12, pytest 9.2.0,
NumPy 2.3.5, pandas 3.0.1, Pillow 13.0.0; isolated editable-install environment.
The first baseline attempt encountered host permission errors in pytest's
shared TEMP directory; the fresh workspace-owned `--basetemp` resolved them.
No product code or global dependency lock was changed to work around TEMP.

Checks cover clipping/offsets, stable IDs, accepted schema, zero first pass,
both-stage behavior, failed second pass, invalid pseudo boxes, count-only
rejection, NaN/Inf scores/boxes, threshold/content filtering, GT leakage,
protocol/config mismatches, missing assets, source/checkpoint/BERT/CUDA hash
mismatches and unchanged native one-pass behavior.
Both example preflights return **BLOCKED (exit 2)** as expected. The official
example reports missing checkpoint hash and local repo/checkpoint/BERT paths.
This Windows CPU environment lacks torch; GPU execution used the separately
verified existing WSL `praco-venv`, without changing either environment's lock.

## GPU / V14 blockers and handoff

WSL smoke verified **RTX 5060 Laptop GPU, 8,151 MiB, driver 591.74**, Torch
`2.9.1+cu128`, CUDA 12.8 and the existing compiled deformable-attention op.
The two runs of one fixed image/query both produced 16 proposals; final
instrumented inference was 1.929466 s with 1,498,123,776 peak allocated bytes.

1. **Thúy / P01:** provide the exact V14 run lock: source commit/diff, checkpoint
   SHA + provenance, BERT/tokenizer/weights inventory, prompts, preprocessing,
   thresholds, precision/AMP/TF32 flags, seed, CUDA extension and library pins.
   Supply GT-free baseline raw boxes/scores/native indices, image/query hashes,
   coverage and raw `full_cohort_output.json` provenance. The mirror contains
   only summary/hash references, not that raw file. Evaluation GT stays separate.
2. **Source compatibility:** this adapter is query-only, batch=1, pseudo=3,
   full-image, no adaptive cropping/TT-Norm. If V14 uses a different source or
   any different setting, stop and review a separately versioned implementation;
   do not alter a frozen protocol or silently patch CUDA for this host.
3. **Runtime:** a source/asset-locked local smoke config now exists under
   `.venv/p02_gpu/local_config.json`. It intentionally identifies the reused
   checkpoint as a third-party mirror with no established V14 correspondence.
   The old compiled op's two-site CUDA type-API compatibility patch is recorded;
   no upstream or PrACo source was patched during this smoke. V14 still requires
   its own runtime/build lock and a fresh process to avoid namespace collisions.
4. **Cohort + preregistration:** Thư/Thúy approve a GT-free, non-`FINAL_SEALED`
   20–50-pair smoke/parity roster and the tolerance plan **before raw outputs
   are opened**. Agree on the new deterministic candidate IDs for Thư's exact
   node mapping and Thúy's raw-index matching; old frozen FSC mappings are not
   silently reused. No schema/policy changes are requested in this patch.
5. Run preflight, then baseline-only create-only smoke with the filled local
   config using `countlab-online`. Freeze candidate/raw prediction hashes,
   report missing/extra boxes and count parity separately. Archive actual raw
   detector outputs from the reference and new engine for a GPU review fixture.

**Real GPU smoke: PASS, one unique image/query executed twice, two detector calls
per execution. V14 count parity: NOT MEASURED. V14 bbox parity: NOT MEASURED.**
Raw prediction hashes are identical across the two executions. The small real
candidate fixture and sanitized summary are in `fixtures/countgd_official_gpu_smoke*.json`;
full raw predictions/config/logs remain local under `.venv/p02_gpu/`.
Existing PrACo assets and this smoke do not establish V14 provenance. The exact
handoff list is `P02_V14_ARTIFACT_REQUEST.md`.

Primary implementation references:
[official pseudo selector/evaluator](https://github.com/niki-amini-naieni/CountGDPlusPlus/blob/2ecbc4331d6730d017f52a4a97bbcbae7d5d714a/engine_inference.py),
[PrACo native adapter](https://github.com/ciampluca/PrACo/blob/438a968aef9d31ded7519733bd84e89baae68896/models/countgdplusplus_model.py).
