# PrACo Test-1 — CountGD++ official adapter + frozen Qwen V7.6

**Scope:** FSC-147 TEST negative-label protocol: 1,190 images × 29 target classes = 34,510 image/query pairs. This experiment is **not** PrACo Test-2 (Mosaic) and is not a re-run of the team's 1,190-positive + 111-verified-negative V7/V7.5 evaluations.

## Code supplied here

- `CS331_PrACo_RTX5060_WSL2_FULL_Local.ipynb` — original **A** notebook copied from the experimenter's WSL2 laptop with `collect_original_a_notebook.sh` on 2026-11-14. All 9 cells retain their exact source; only saved outputs, execution counts and private metadata were cleared. Its embedded 14,778-pair resume cache is inference-result data, not model weights.
- `CS331_PrACo_OfficialAdapter_B_V76_Colab.ipynb` — exact Google Colab **B** inference/verification notebook source from the completed run; outputs and execution counts removed for safe git commits. It loads the A FULL ZIP, runs/reuses Qwen3-VL verdicts, checkpoints to Google Drive, seals predictions before reading GT, and computes paired Test-1 metrics.
- `../../tools/praco_test1/verify_praco_artifacts.py` — offline audit of both finished ZIP artifacts (no GPU or model download needed).
- `../../tools/praco_test1/collect_original_a_notebook.sh` — helper used to collect the **original A WSL2 notebook from the author's laptop**. The included A source was copied from the actual local file, not reconstructed from predictions.

## Inputs / run B on Colab

1. Create `MyDrive/CS331_PrACo_B/` and upload `CS331_PrACo_OFFICIAL_ADAPTER_A_FULL_FULL.zip` from `../../results/05_praco_official_adapter/archives/`.
2. If new Qwen inference is needed, also provide FSC-147 image files separately as `hihihi5.zip` or a directory, not committed here. An optional **validated** prior frozen Qwen cache can reduce inference, but was **not** used in the archived B FULL run.
3. Open the B notebook in Google Colab, enable GPU. Start `RUN_MODE='PLAN'`; then `RUN_MODE='RUN', MAX_NEW_PAIRS=3`; finally `RUN_MODE='RUN', MAX_NEW_PAIRS=0` to run FULL and resume from Google Drive.
4. Compare results with the original B FULL archive rather than expecting bitwise-identical GPU output across different environments.

The model revision, prompts, two-step `NO/NO` rule, precomputed SHA checks and images must not be silently changed. Some Google Colab runtimes may terminate; checkpoints allow resuming but do not guarantee uninterrupted remote execution.

## Exact A source collected from the author's laptop

The official-adapter A was executed on Windows + Ubuntu WSL2 with RTX 5060 8 GB. The completed A archive documents PrACo commit `438a968aef9d31ded7519733bd84e89baae68896`, model SHA, synthetic-exemplar SHA, roster SHA, and Torch `2.9.1+cu128`. The checkpoint is labeled as a **third-party Hugging Face mirror, equivalence to the author's checkpoint unverified**.

The exact `CS331_PrACo_RTX5060_WSL2_FULL_Local.ipynb` was found in `~/PrACo_Local/` and copied using the helper above. A cell-by-cell comparison confirmed unchanged source and cell types. The cleaned notebook SHA256 is `50bcd6ccf81bbbf9bcfe6b35b6288296253233fcf74c47248a71f8049174622e`. Its embedded resume CSV passes its original SHA256 check and contains 14,778 completed pairs. Static validation does not establish bitwise-identical GPU reproduction, and the checkpoint-mirror limitation remains.

For measurements, caveats and provenance, see `../../docs/PRACO_TEST1_A_B_RESULTS.md`.
