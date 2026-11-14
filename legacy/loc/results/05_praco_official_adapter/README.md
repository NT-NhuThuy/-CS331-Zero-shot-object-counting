# Completed PrACo Test-1 official adapter A/B artifacts

Both ZIP archives are **original completed run files** (not recompressed, not regenerated). SHA256 values are in `SHA256SUMS.txt`.

`BUNDLE_MANIFEST.csv` records the paths, byte sizes and SHA256 of the final upload, including the collected A notebook, updated documentation, validation tools and the short link added to `Lộc/README.md`; it excludes itself. The earlier V4–V7.5 `Lộc/FILE_MANIFEST.csv` remains the historical manifest of the 2026-11-06 package.

| Artifact | Description |
|---|---|
| `archives/CS331_PrACo_OFFICIAL_ADAPTER_A_FULL_FULL.zip` | CountGD++ native PrACo adapter FULL, 34,510 rows; roster, baseline CSV and 1,190 × 29 matrix, A source/protocol metadata |
| `archives/CS331_PrACo_OFFICIAL_ADAPTER_B_FULL.zip` | Frozen Qwen V7.6 + adapter FULL, 34,510 sealed rows, B protocol, paired metrics, replay checkpoint |
| `PrACo_OfficialAdapter_Paired_Test1_metrics.csv` | Exact official-adapter-vs-V7.6 metrics copied from B ZIP |
| `summary.json` | Independently recomputed audit summary + hashes from A/B archives (audit itself has no GT without `--gt`) |
| `positive_class_suppressed_19.csv` | 19 positive-class image/query pairs suppressed to 0 by `NO/NO` |
| `class_mismatch_remaining_72.csv` | 72 class-mismatch pairs that still have positive semantic counts |
| `A_official_baseline_protocol.json`, `A_official_sources.json`, `A_run_state.json` | Source provenance and status copied from A ZIP |
| `paired_protocol.json`, `prediction_sha256.txt` | Frozen B provenance, sealed predictions SHA256 |

Use `../../tools/praco_test1/verify_praco_artifacts.py` to validate these archives offline. Ground-truth-based recomputation of AvgNP, AvgNMN, PCCN, MAE and RMSE requires the separately supplied official PrACo `gt_cnts_FSC147.json` (fixed PrACo commit).

**Research caution:** Test-1 negative labels are different target classes and **not guaranteed verified-absent**. Do not call class-mismatch nonzero rate verified-absent FPR. V7.6 greatly reduces AvgNMN but makes MAE, RMSE and PCCN worse; do not claim unqualified superiority. PrACo Mosaic Test-2 not run. Original dataset images, model weights, Hugging Face tokens and `.env` files are intentionally excluded.
