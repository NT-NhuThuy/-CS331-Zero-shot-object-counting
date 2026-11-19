# T01 integration verification — 2026-11-19

**G0: HOLD. T01: HOLD_WITH_EVIDENCE_NOT_DONE. Strict present/absent parity: NOT_MEASURED.** This submission publishes frozen diagnostic evidence and comparison tooling for review; it does not complete G0.

## Integration scope

The new, independent clone `UNICOUNT_T01_CLEAN_WORKSPACE` was fetched from the team repository. Branch `feat/loc-t01-g0-review-20261119` starts at integration commit `3fb58ccedb690212270806313fe005c4c0c16213`, after PR #11. The earlier `.clones/unicount-v2-t01-t04` checkout was not changed, reset, cleaned, stashed or staged.

Input archive: `UNICOUNT_T01_G0_PR_OVERLAY_20261119.zip`, SHA256 `843b19630d0b8424dacd618da3c989da24de27dda41fa4b44278f8f33cf59d94`. It was inspected and extracted into an OS temporary directory outside the repositories. All 35 archive entries were audited; every payload covered by the supplied checksum inventory matched before integration. No existing repository target had differing content, and no existing file was overwritten.

The bundle's `CHECKSUMS_SHA256.json`, metadata, instructions, status and prior test log are retained as **historical bundle/snapshot records**. The checksum inventory also names scaffold files that were not in the overlay. In particular, PR #11 changed Thuy's `src/countlab/method/pipeline.py` after that snapshot; the current base implementation was reviewed and preserved. Those context-only snapshot hashes are not an attestation of the current integration branch. Current test evidence is recorded separately below.

One documentation-only integration correction removed two trailing Markdown spaces from the owner/deadline line of `parity_report.md` so whitespace checks pass. Original report SHA256: `b5a5200074f8ef3a4f8660cf9b53fc1703cdf91a1814e50a88ca7e9e72d28633`; integrated report SHA256: `134b035f5996d9d82c0b8e4a45623dc5e255c354a071e05c4ced32c2d9c1d8e7`. No statement, number, threshold, tolerance or scientific status changed.

The archived `r10_v14_strata_diagnostic.csv` has original CRLF bytes. The local `.gitattributes` entry disables newline conversion for that one file and recognizes CRLF during whitespace checking while retaining checks for actual trailing spaces. Its SHA256 remains `4d22f91ae18d9ceeb768fad5fc77c634097990f71a633496fc52f448b66bc772`; no frozen data was rewritten for Git.

All 668 files originally tracked in the new base retain their working bytes. Shared online/preflight/native/schema/method modules, legacy, checkpoints, datasets, sealed artifacts and T04/T07/T10/T13 were not changed. No model inference, package installation or GPU execution was performed.

## Fresh local CPU verification

From the new clone root, with `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1` and an empty `CUDA_VISIBLE_DEVICES`:

```text
python -m pytest -q tests/test_countgd_official.py tests/test_t01_parity.py tests/test_t01_r10_replay.py tests/test_t01_v14_diagnostic.py --basetemp .venv/t01_submission_20261119/pytest_cpu -p no:cacheprovider --tb=short
```

**53 passed in 4.22s**, exit 0, freshly executed on this integration base. The test process did not import Torch or Transformers. Runtime was Windows Python 3.12.14 using an existing CPU test interpreter; these versions are not a new GPU runtime lock. Simulated reference/sign-off fixtures in the unit tests are not independent GPU evidence or real team approvals. The five original P02 shared-dispatch tests remain outside this additive adapter scope.

The actual raw replay CLI was also run:

```text
python tools/t01/audit_r10_raw_cpu_replay.py --r10-zip docs/method/parity_logs/UNICOUNT_T01_R10_GPU_EVIDENCE.zip --image-dimensions docs/method/parity_logs/r10_image_dimensions_verified.json --out .venv/t01_submission_20261119/r10_raw_replay_local.json
```

Result: **32/32 pairs, 38/38 candidates and 64/64 stage canonical hashes PASS**. Candidate IDs/native indices, scores and original-pixel boxes replayed from raw tensors. The complete result equals the supplied frozen `r10_internal_parser_replay.json`. This is internal parser reproducibility, not independent model parity.

Invoking `t01_parity.py` with only the R10 archive and a new output path returned the expected **exit 2**, `HOLD_REFERENCE_LABELS_AND_SIGNOFF_REQUIRED`, `numeric_parity=NOT_MEASURED`, and `g0=HOLD`, after checking all 32 pairs. This expected HOLD exit is not a test failure.

Current logs and per-pair results: [`t01_cpu_tests_local_20261119.txt`](parity_logs/t01_cpu_tests_local_20261119.txt), [`t01_local_submission_verification_20261119.json`](parity_logs/t01_local_submission_verification_20261119.json). The original supplied `TEST_RESULTS_LOCAL_20261119.txt` is not presented as a fresh test run.

## Evidence inventory and limits

- Official adapter content SHA256: `b0890cd15a5e46d055ad6680a0a29bdaf2aae9c73e125bb9cbf9892c69a87773`, exactly the prepared P02 adapter.
- R10 protocol candidate SHA256: `121786020bead2e87d58e9dd6896d022ba2068d58feecfbbc4026113b57f497e`, independently recalculated from the complete embedded protocol.
- R10 checkpoint SHA256 recorded in provenance: `b89395d79711c9b9f13f3ef80cd2e528386fb1d70453f99443be4eccab33592f`. No checkpoint or BERT weights are included or reloaded here.
- R10 evidence ZIP: **1,790,399 bytes**, SHA256 `ac20048bf11fd858951caa71a7c3bcec576e371b40013b1b84c73eea91c12554`. All 40 members were inventoried: 32 numeric raw NPZ files plus JSON/JSONL/Markdown/log provenance. NPZ arrays were inspected with `allow_pickle=False`; no object arrays, checkpoints, BERT, image files, virtualenv or nested private dataset was present. Text files were scanned for token/private-key patterns with no matches.
- R10 observed GPU diagnostic: **32/32 pairs, 64 calls, 38 candidates**. These are preserved Kaggle measurements, not GPU inference rerun during this integration.
- Archived V14 count-only diagnostic: **31/32**, absent **17/17**, present **14/15**; `113 / cone`: R10 FP32 **18**, V14 AMP **19**. The assertions against the archived diagnostic record were rerun, but the external V14 tar comparison was not repeated or uploaded. Strata remain separate evaluation/diagnostic data and are not supplied to inference.

Threshold 0.23 is frozen only for the recorded R10 candidate operating point; it has not become a team-approved official threshold. CountGD++ and Qwen runtime requirements remain separate. No protocol/hash/tolerance was changed to turn HOLD into PASS.

## Remaining review blockers

1. Team sign-off for the precise primary source/checkpoint/FP32 pseudo3 protocol and cohort.
2. An independently produced, matched reference with full native indices, original float boxes and scores on all image-SHA/query pairs.
3. Independently reviewed present/verified-absent strata and pre-approved comparison tolerances, retained outside inference.
4. Strict present and absent comparison acceptance. V14 AMP versus R10 FP32 count-only agreement cannot satisfy that gate, and historical V14 source/runtime binding remains incomplete.

The supplied `T01_STATUS.json` and original parity report record the bundle's pre-submission Git state. Current Git publication and review status belong to the actual branch/PR; publishing this submission resolves the delivery step without changing G0 or declaring T01 DONE.
