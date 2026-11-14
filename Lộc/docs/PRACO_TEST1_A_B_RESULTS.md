# PrACo Test-1 — completed CountGD++ official-adapter A/B with frozen V7.6

**Status (2026-11-14):** A and B FULL complete, checked on 34,510 image–query pairs from 1,190 FSC-147 TEST images × 29 target classes. Results were provided by the experimenter as original ZIP artifacts; internal consistency was audited offline. **The official TEST had already been used/observed in related development. These results are descriptive of this setup, not an untouched held-out validation claim.**

## Evaluation design

1. **A baseline:** CountGD++ PrACo native adapter (positive/negative prompts, synthetic exemplars, pseudo-exemplar rerun), not the group's earlier text-only interface. PrACo commit: `438a968aef9d31ded7519733bd84e89baae68896`.
2. **B method:** freeze `Qwen/Qwen3-VL-4B-Instruct` at revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, and use two sequential image-based checks on pairs with baseline count > 0. Only `NO` and `NO` sets count to 0; all other verdicts retain baseline count. Qwen's input is image + target class. B results used **4,303 newly executed** Qwen cases and **0** reused cache cases; 30,207 baseline zeros required no Qwen.
3. Complete prediction matrix and evaluation: same 34,510 pair IDs, identical baseline counts in both archives, no duplicate pairs, both stage statuses FULL. GT is only consumed after B predictions are sealed, as designed by the Colab notebook.

## PrACo Test-1 metrics (copied from sealed B FULL)

| Metric | CountGD++ native adapter (A) | A + frozen V7.6 (B) | Direction |
|---|---:|---:|---|
| AvgNP | 1.085750 | 1.068461 | ↓ 0.017289 (toward 1) |
| AvgNMN | 0.081101 | 0.002042 | ↓ ~97.48% (better) |
| PCCN (%) | **97.983193** | 97.058824 | ↓ 0.924370 percentage points (worse) |
| Positive MAE | **12.180672** | 13.528571 | ↑ 1.347899 (worse) |
| Positive RMSE | **100.007231** | 115.145480 | ↑ 15.138249 (worse) |
| Class-mismatch nonzero rate | 9.342737% | **0.216086%** | ↓ ~9.126650 percentage points |
| Positive-class zero counts | **0** | 19 | Worse |

**Measured impact:** B suppressed 3,060 positive baseline counts: 3,041 belonged to class-mismatch queries, 19 belonged to true positive-class queries. In 72 class-mismatch pairs, B still predicted a positive count. This is a **precision/recall-type trade-off in semantic suppression**, not evidence that positive counting improved.

## Audit / provenance

- ZIP A SHA256 `523db846cd2f46ce8df7c7ae6f3b472ddffa9ceaa8cf5a17e1b05f3e90355e27`.
- ZIP B SHA256 `e0301ef2d65bfa77310f6386105ea4b9716e07beb2d5769022e49b04da6c7b62`.
- A protocol SHA256 `ab3a19afe6447047e71e4688fab39111b3c0ae90e02868a486bbecca341ba91c`.
- B protocol SHA256 `dbca095a01558ce10361255ac2c94a71e070560f7f24be8b712a1cb192d6cbc7`.
- Sealed B predictions SHA256 `0346f78f2727ee50d8a67e477a8d70bb56ccedf56e94580d46bb97fc662a6b6a`.
- A CountGD++ checkpoint SHA256 `b89395d79711c9b9f13f3ef80cd2e528386fb1d70453f99443be4eccab33592f`; **third-party HF mirror**. Official author checkpoint bitwise equivalence is **unverified**.
- Synthetic exemplar JSON SHA256 `36f2142946464917f95b4503a7baa76a5d58a3a501a9db28554d9a64f0ec7fc2`.
- Official Test-1 roster SHA256 `b73d78d0e21d089cdd875913fe861929904bc7842e600e84ff5a562d439eae39` (seed=123).
- A contains 299 negative `elapsed_sec` values. These do not affect counts but prevent treating that column as reliable per-inference runtime evidence without further investigation.

## Scope and limits (read before citing)

- **Not PrACo Test-2 Mosaic.** Test-2 requires different mosaic/density-map evaluation.
- **Not a verified-absent hard-negative FPR**: Test-1 class mismatch does not guarantee that a requested category is absent in the photograph.
- **Do not combine the four protocol configurations as if all used the same CountGD++ baseline.** Earlier `text-only + V7.6` is a separate protocol and is not included in this two-row A/B comparison table.
- **No re-tuning on TEST.** If analyzing the 19 suppressed positive-class cases or 72 remaining mismatch cases, treat as failure analysis, not a fresh unbiased model selection run.
- Original **A notebook source was collected from the experimenter's WSL2 laptop** on 2026-11-14 using `collect_original_a_notebook.sh`; all 9 cells retain their exact source, with only outputs, execution counts and private metadata cleared. The A and B notebooks are included under `../notebooks/06_praco_official_adapter/`. This source collection does not remove the checkpoint-mirror limitation or establish bitwise-identical GPU reproduction.

## Verification

From repo root (needs Python + `pandas`, `numpy`):

```bash
python -m pip install -r 'Lộc/tools/praco_test1/requirements.txt'
python 'Lộc/tools/praco_test1/verify_praco_artifacts.py' \
  --a 'Lộc/results/05_praco_official_adapter/archives/CS331_PrACo_OFFICIAL_ADAPTER_A_FULL_FULL.zip' \
  --b 'Lộc/results/05_praco_official_adapter/archives/CS331_PrACo_OFFICIAL_ADAPTER_B_FULL.zip'
python -m pytest -q 'Lộc/tools/praco_test1/test_verify_praco_artifacts.py'
```

Optional: add `--gt <path-to-gt_cnts_FSC147.json>` to recompute 5 primary metrics independently rather than checking only the sealed CSV's internal consistency and recorded metrics.

### Local verification (2026-11-14)

- CPU artifact audit: **PASS**, matching the supplied `summary.json`: 34,510 pairs, 3,060 suppressed, 19 positive-class suppressed, 3,041 class-mismatch suppressed, 72 class-mismatch pairs remaining nonzero.
- Independent ground-truth recomputation: **PASS** for AvgNP, AvgNMN, PCCN, MAE and RMSE for both A and B (absolute tolerance `1e-9`). The local GT was checked byte-for-byte against `data/gt_cnts_FSC147.json` at PrACo commit `438a968aef9d31ded7519733bd84e89baae68896`; GT SHA256 `420edf45568706fb703aa0aaa975545d614947fb56e599c50fa765b866f1df85`. This GT is not added to this repository.
- Pytest: **2 passed**, including a Windows-line-ending regression check. The audit now explicitly serializes the roster with LF for its original SHA256, avoiding a Windows CRLF hash mismatch without modifying any CSV, ZIP or protocol.
- Both A/B notebooks pass notebook-schema and static Python-syntax validation and have no saved outputs or attachments. A's embedded 14,778-pair resume cache matches the corresponding final A rows; B's exact prompts and model revision match the sealed protocol.
- Original A/B archive SHA256 checks pass; no credentials, model weights, raw images or logs were found in the upload. No GPU inference or model download was performed.
- Validation environment: Python 3.12, NumPy 2.3.5, pandas 2.3.3, pytest 9.2.0 and nbformat 5.11.1 in a temporary environment.

Original sources: [PrACo](https://github.com/ciampluca/PrACo); [CountGD++](https://github.com/niki-amini-naieni/CountGDPlusPlus).
