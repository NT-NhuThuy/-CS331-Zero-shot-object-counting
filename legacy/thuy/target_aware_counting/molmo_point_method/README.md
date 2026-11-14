# MolmoPoint direction A: implementation checkpoint

This directory implements a **native greedy control**, a slow branch-isolated
reference spatial search, GT-free CountGD proposal-to-patch mapping, and the
finite-bank GOSPA/MBR decision rule from
`RESEARCH_AND_TWO_METHOD_DIRECTIONS_20261030.md`. The spatial search is an
implementation prototype. Full 29-token width-one replay parity and an
eight-step spatial-head score audit passed on `lvis_dev:1`; search and
cohort-level accuracy remain unverified.
Do not call it a validated direction A method or report its MBR function as a gain.

Official model source/config was fetched at revision
`188130f961c8e0888a34e11121a1423c461a01ba` into `vendor_source_188130f`.
The model's generation wrapper writes the patch-token logit to only the
argmax patch address. `head_scoring.py` retains the language-versus-PATCH_TOKEN
probability, then uses the raw patch head's conditional distribution over
valid addresses. It uses the native logit processor for masks. Subpatch and
location choices use the native head values after the same grammar masks.
`reference_replay.py` replays each branch with its own KV/image/last-patch
state; it is intentionally expensive and only suitable for parity probes.
`run_reference.py` compares that replay's width-one **native policy** token by
token against `generate`, then runs proper and proposal-guided beam searches.
Their branch probabilities are never given a CountGD score bonus.
The native grammar permits some EOS-terminated text that the official point
parser cannot extract (for example a point-token triple without its numeric
instance label). `run_reference.py` retains these completed branches with a
decode error and null count, reports valid/invalid completion totals, and
excludes invalid branches from MAP/MBR. A valid empty set remains a candidate
with count zero. If no valid complete sequence remains, the decision is null
with status `no_valid_complete_sequences`.

Full 29-token native replay parity passed for `lvis_dev:1` on pc152, including
the complete four-point decode and original-image coordinates. This is a
correctness probe on one item, not a development-cohort accuracy result.
`--audit-native-path` now reads the locked native prediction JSONL and checks
representative first/last patch, subpatch, location, no-more-points, and EOS
steps against an independent normalization of the frozen model heads. This
mode skips native `generate` and the full width-one parity replay; it still
replays each audited prefix with isolated KV state. It fails on any identity,
completion, missing token, head mismatch, or log-probability discrepancy.
The pc152 audit artifact `output/molmo_point_a_20261030/score_audit_gpu152_lvis1.json`
passed all eight selected steps with maximum absolute log-probability delta
`3.725290298461914e-07` at frozen revision
`188130f961c8e0888a34e11121a1423c461a01ba` in bfloat16. The audited
steps were the first and last patch/subpatch/location choices, the native
no-more-points token, and EOS. This validates those choices on one image; it
does not establish proposal-guided search quality or MAE.

The first pc152 beam-2 search artifact is
`output/molmo_point_a_20261030/search_gpu152_lvis1_beam2.json` on `lvis_dev:1`
(`max_new_tokens=40`, `max_language=4`, `max_spatial=8`, `guided_slots=4`).
Proper and F1-guided search each produced 63 EOS-complete traces: 55 decoded
validly, eight had point-token triples that the official text parser did not
extract, and two active paths reached the token limit. Both valid banks have
nine unique point sets. The entire proper/guided trace lists and probabilities
were identical; guided search explored one extra option but it did not survive
beam pruning. Native, MAP, and MBR all selected the same four points on this
item. The GOSPA cutoff was 32 pixels and the two searches took 324 and 314
seconds respectively, excluding model load and native parity. This one-item
diagnostic supplies no paired accuracy gain or MAE conclusion.

## Cached cohort runner and final development metrics

`reference_search.spatial_search_cached` reuses a private KV cache for every
surviving beam path and only advances candidates retained by beam pruning.
`reference_search.spatial_search` remains the slow oracle. A CPU toy model that
mutates its cache in place verifies identical paths and scores, but the cached
runner must also match the existing real-checkpoint reference artifact on
pc152 before cohort inference. `compare_cached_reference.py` checks every
complete token sequence, decoded set, log probability, and MAP/MBR decision.

After that gate passes, run from the ZOC root on pc152:

```bash
bash target_aware_counting/molmo_point_method/run_a_dev_cohorts_20261031.sh
bash target_aware_counting/molmo_point_method/finalize_a_dev_cohorts_20261031.sh
```

The first script loads the frozen model once per cohort, writes and fsyncs each
item in a resumable JSONL, and stops on the first error. The second refuses
missing items, exports native/proper MAP/proper MBR/guided MAP/guided MBR
predictions, freezes each full denominator before labels are opened, and
writes paired M0 evaluations plus `summary.md`. On items without complete F1
raw proposals, guided outputs use the proper branch by a declared fallback.
The output-token cap of 4096 and a `1e-6` search-frontier mass ratio bound are
fixed in the run contract. That bound only concerns expanded paths retained by
this beam procedure; it is not a bound on the model's full posterior or on
pruned paths. No cohort metrics are available before these GPU runs complete.

On pc152, after the native one-item JSONL exists:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
timeout 30m countgd_venv_pc152/bin/python \
  -m target_aware_counting.molmo_point_method.run_reference \
  --cohort lvis_dev300 --item-id 'lvis_dev:1' \
  --snapshot target_aware_counting/output/molmo_point_a_20261030/model_188130f \
  --revision 188130f961c8e0888a34e11121a1423c461a01ba \
  --output target_aware_counting/output/molmo_point_a_20261030/score_audit_gpu152_lvis1.json \
  --device-map cuda:0 --dtype bfloat16 --max-new-tokens 29 \
  --audit-native-path target_aware_counting/output/molmo_point_a_20261030/native_gpu152_lvis1_512.jsonl
```

`native_greedy.py` reads only the three frozen development manifests in
`sink_target_selection_20261029_v1`. It checks their locked SHA256 hashes and
denominators. It accepts only a local Hugging Face cache snapshot for
`allenai/MolmoPoint-8B` with an explicit 40-hex revision. Transformers runs
with `local_files_only=True`; no labels or external services are used.

Run from the ZOC root after the official snapshot and compatible Transformers
dependencies are installed. The checkpoint declares Transformers 4.57.1;
`output/molmo_point_a_20261030/runtime_py310` is the isolated runtime being
prepared for that version. Use a new output path for each cohort:

```bash
SNAPSHOT=target_aware_counting/output/molmo_point_a_20261030/model_188130f
REVISION=188130f961c8e0888a34e11121a1423c461a01ba
target_aware_counting/output/molmo_point_a_20261030/runtime_py310/bin/python \
  -m target_aware_counting.molmo_point_method.snapshot_lock \
  --snapshot "$SNAPSHOT" --revision "$REVISION"
target_aware_counting/output/molmo_point_a_20261030/runtime_py310/bin/python \
  -m target_aware_counting.molmo_point_method.native_greedy \
  --cohort lvis_dev300 --snapshot "$SNAPSHOT" --revision "$REVISION" \
  --output target_aware_counting/output/molmo_point_a_20261030/native_lvis_dev300.jsonl \
  --max-new-tokens 4096 --dtype bfloat16 --fail-fast
```

Create the snapshot lock only after all eight weight shards are downloaded.
It hashes the config, model source and all shards. Use `--resume` only with the
same manifest, snapshot and token limit. The runner
appends and fsyncs each row. It records complete, truncated and error separately;
count is null for incomplete generations. A valid empty prediction is a complete
generation with zero extracted points. Output includes the generated token IDs,
decoded text, points in original image pixels, token count and elapsed time.
The published safetensors total about 32.3 GiB and the model config declares
float32; `--dtype bfloat16` is an explicit inference load choice for a 24 GiB
GPU. The choice is recorded in the prediction contract and parity compares
both decoding paths under the same loaded dtype.

Before any evaluation, verify every row is complete and freeze the JSONL hash.
Use `python -m target_aware_counting.molmo_point_method.freeze --cohort
lvis_dev300 --predictions PATH.jsonl --output PATH.freeze.json`. Freeze refuses
any error, truncation, duplicate, count mismatch or item identity mismatch.
After freeze, `python -m target_aware_counting.molmo_point_method.evaluate
--cohort lvis_dev300 --predictions PATH.jsonl --freeze PATH.freeze.json
--output PATH.evaluation.json` compares the same items with M0. FSC point
matching uses a declared pixel radius (default 16), while LVIS point matching
uses one-to-one point-in-GT-box. M0 box centers are used for the common point
metric; neither point metric substitutes for LVIS box IoU50. The evaluator
reports count error, exact, TP/FP/FN, false zeros, absence errors and a
source-image bootstrap CI. Its first label read occurs after freeze and
denominator validation.
The locked snapshot now loads and native generation runs in bfloat16 under
CUDA autocast. Short-prefix runtime equivalence passed; full native completion
and spatial-step probability parity are still unverified. The
official example uses `build_logit_processor_from_inputs`,
`return_pointing_metadata=True`, and `extract_image_points`; this runner uses
the same interfaces: [official MolmoPoint inference example](https://github.com/allenai/molmo2/blob/main/MOLMO_POINT_README.md#molmopoint-transformers-inference).

`set_risk.py` accepts *complete-sequence* log probabilities supplied by a
future validated spatial search. It never assigns a synthetic score to the
empty set. Identical point sets are deduplicated before normalizing bank
weights. GOSPA uses `alpha=2` and a declared pixel cutoff; sensitivity to that
cutoff still needs an experiment. The normalized weights apply only to the
truncated hypothesis bank, not to the model's full posterior. Assignment cost
can be substantial for dense images and large banks.

Next correctness gate: finish and lock the official snapshot; reproduce native
greedy output, including zero and dense cases; compare width-one replay token
by token; verify step log probabilities on real outputs; then measure proper
and guided search under the same beam and compute budget. The reference
search's repeated replay is not suitable for full-cohort runs. Optimize branch
caches only after parity is established, then run paired full development
cohorts and evaluate with the locked protocol.

## Direction B: capacity allocation prototype

`capacity_warp.py` reads the actual processor `subpatch_mapping` and
`token_pooling` for one image. It computes maximum one-to-one matching from F1
proposal centers to every available decoder patch/subpatch address across
crops. Only proposals in a deficient alternating component create warp demand.
The uniform floor is 0.5 and slope bounds are 0.5 to 2.0 by default. With no
deficiency, the map is identity. The inverse returns decoded points to source
pixels. This capacity is a proposal proxy; duplicate or part proposals can
create false demand.

The forward-map density is integrated on uniform source bins and inverted to
the source edges used by the renderer. This direction matters: high-demand
regions expand in warped coordinates.

`run_capacity_b.py` is a **single development item diagnostic**. It checks
the locked manifest and F1 sidecar, processes source and warped images, and
refuses an unequal input tensor budget. `--diagnostic-only` writes the CPU
capacity comparison without loading model weights. By default it also skips
model loading if matched proposal count did not increase; `--allow-no-gain`
overrides this for an explicit control. With positive capacity gain it runs
one native greedy decode on the warped image. It records map edges, matching
before/after, tokens, and prediction status. It never adds points from source
and warp.

`audit_capacity_b.py` runs the locked LVIS300, FSC primary VAL100, and FSC
verified absent111 manifests on CPU. It writes one resumable row per item and
a summary with positive/zero/negative capacity gain, missing F1 pools, and
identity warps. It validates processor files against the snapshot lock and
never loads model weights or labels. Optional `--native-predictions
COHORT=PATH.jsonl` adds a separate native point support diagnostic for complete
rows only; those points do not change the F1-only warp. Missing native rows
remain explicitly unavailable. The 511-item audit command is:

```bash
countgd_venv_pc152/bin/python \
  -m target_aware_counting.molmo_point_method.audit_capacity_b \
  --cohort all \
  --snapshot target_aware_counting/output/molmo_point_a_20261030/model_188130f \
  --revision 188130f961c8e0888a34e11121a1423c461a01ba \
  --output target_aware_counting/output/molmo_point_b_20261031/capacity_cpu_all_v1.jsonl \
  --summary target_aware_counting/output/molmo_point_b_20261031/capacity_cpu_all_v1.summary.json
```

The completed CPU audit has 511/511 rows. It found positive assignment
capacity gain on 262/300 LVIS items and 69/100 FSC primary items. Every one
of the 111 verified negative items lacks a complete F1 raw proposal pool, so
the F1-only warp is identity there. No full native point file was supplied;
native support coverage is 0/511. These are geometry results, not count or
localization gains.

`run_capacity_b_cohort.py` consumes the complete audit and decodes exactly one
image per item. It selects the warp only for positive assignment capacity gain
and uses the source image otherwise. The rule and audit hash are stored in every
prediction row. It saves complete, truncated and error separately and supports
resume. Its full-cohort output can be checked with `freeze.py` before labels
are opened. A one-item smoke selected by the first positive FSC item in locked
manifest order is:

```bash
countgd_venv_pc152/bin/python \
  -m target_aware_counting.molmo_point_method.run_capacity_b_cohort \
  --cohort fsc_primary_dev100 --item-id 'fsc_val_primary:153' \
  --audit target_aware_counting/output/molmo_point_b_20261031/capacity_cpu_all_v1.jsonl \
  --snapshot target_aware_counting/output/molmo_point_a_20261030/model_188130f \
  --revision 188130f961c8e0888a34e11121a1423c461a01ba \
  --max-new-tokens 4096 --dtype bfloat16 --fail-fast \
  --output target_aware_counting/output/molmo_point_b_20261031/fsc153_capacity_smoke.jsonl
```

This smoke tests the new runner only. It needs GPU and must not be described as
method performance. Full FSC primary, LVIS and verified negative evaluation
requires paired native/uniform controls on the same items and budget.
For a complete B cohort JSONL, run `freeze.py` with the matching cohort and
then `evaluate.py --method-label capacity_b` with that freeze. The evaluator
checks the B method contract before opening labels; its result compares B with
M0 on the same locked items. Compare that result separately with the native
MolmoPoint control on identical items before making a B claim.

### FSC100 first-pass stopping failure and locked retry

The first FSC100 B run (`fsc_primary_dev100_b_v1.jsonl`) contains 92 complete
rows and eight truncated rows at 4096 output tokens. None of the truncated
rows emitted `no-more-points`; their partial points are not final predictions.
`FSC_PRIMARY_B_RUN_STATUS_20261031.md` records the failed attempt. Increasing
the cap alone is not a validated fix because the output tails contain long
repetitive text or digits.

`recover_capacity_b.py` validates the full v1 file, the audit and checkpoint
lock, copies the 92 complete predictions under a new provenance contract, and
retries only the eight truncated items with the shorter prompt from the
official MolmoPoint pointing example: `Point to the {query}.` It keeps the
same image selection policy, model, dtype and 4096-token cap. `--prepare-only`
has already created `fsc_primary_dev100_b_v2_retry_short.jsonl` with the 92
unchanged prediction payloads. Resume on a GPU with `--resume` and the same
arguments, omitting `--prepare-only`. The eight retries remain in the
denominator even if they truncate again. Only a 100/100 complete file may be
frozen; evaluate it with `--method-label capacity_b_retry`.
Example after checkpoint locking:

```bash
countgd_venv_pc152/bin/python \
  -m target_aware_counting.molmo_point_method.run_capacity_b \
  --cohort fsc_primary_dev100 --item-id 'fsc_val_primary:83' \
  --snapshot target_aware_counting/output/molmo_point_a_20261030/model_188130f \
  --revision 188130f961c8e0888a34e11121a1423c461a01ba \
  --max-new-tokens 4096 --dtype bfloat16 --diagnostic-only \
  --output target_aware_counting/output/molmo_point_b_20261031/fsc83_assignment_diagnostic.json
```

This is not a validated result. Before full paired evaluation, verify the
native processor address mapping against real checkpoint outputs, then compare
native, uniform resolution at the same input budget, and capacity allocation
on all locked development cohorts. Keep any incomplete generation in the
denominator as an error, never as an empty prediction.
