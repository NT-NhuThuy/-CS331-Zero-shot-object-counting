# FQIE external evaluation plan for a top-tier submission

## Frozen contribution

FQIE is a GT-blind bidirectional evidence controller around a frozen open-world counter. It removes a baseline proposal only after semantic=`NO` and whole-instance=`NO`, adds only verified unmatched rescues, and retains every deferred/ambiguous proposal as `UNKNOWN`. V14 rules and prompts are frozen.

The scalable extension is Selective Evidence Routing (SER): rescues are always verified; sparse queries verify all baseline proposals; dense queries route a deterministic budget using maximum proposal overlap, low proposal score, then task ID. Deferred proposals remain `UNKNOWN` and cannot lower the baseline count.

## Dataset ladder

1. **FSC-147 VAL then TEST (mandatory).** Standard text-guided zero-shot counting benchmark. Use VAL for SER budget/runtime selection only; freeze once, then open TEST once.
2. **CARPK (mandatory domain shift if acquired).** Single-class vehicle counting; report MAE/RMSE and runtime. It measures transfer and over-count correction but not semantic/absent safety.
3. **ShanghaiTech-A (recommended dense stress test).** Report MAE/RMSE, task coverage, and runtime. Treat it as a density/scalability test, not open-vocabulary evidence.
4. **Official FSCD-LVIS/COCO query evaluation (mandatory safety table).** Present, verified-absent, semantic-confuser, and part/whole queries. Report absent FP, hard-negative rejection, localization, UNKNOWN/risk coverage.
5. **Blood-cell dataset (optional qualitative domain shift).** Include only if labels and query protocol are auditable; it cannot replace the standard benchmarks above.

## Comparisons

### Directly reproduced

- frozen CountGD++ (primary paired baseline);
- CountGD text-only;
- PseCo zero-shot if its official environment/checkpoint is runnable;
- CountSE and T2ICount if official checkpoints/code reproduce their published protocol.

### Published-number context (clearly separated)

- ZSC, CLIP-Count, CounTX, VLCounter;
- DAVE prompt-based, GeCo zero-shot/low-shot;
- current text-guided methods such as QICA where protocol is compatible.

Never mix text-only with visual-exemplar results in one ranking without a prompt-format column.

## Required ablations

- baseline / rescue-only / rejection-only / full bidirectional;
- SER budgets 8/16/32 on FSC-147 VAL, with TEST budget frozen once;
- verify-all versus SER on a feasible stratified subset;
- semantic-only rejection versus frozen semantic+whole rejection (VAL only);
- `UNKNOWN→retain` versus an unsafe `UNKNOWN→reject` diagnostic (never primary);
- prompt-paraphrase robustness on a fixed stratified subset;
- sparse/dense, low/high count, low/high score, and absent/present strata.

## Metrics and figures

- MAE, RMSE, bias, exact-count accuracy;
- paired bootstrap CI and exact sign/Wilcoxon analysis;
- absent false-positive, present over-pruning, hard-negative rejection;
- proposal precision/recall or point localization where annotations permit;
- UNKNOWN rate and risk–coverage curve;
- wall time, verifier calls, task coverage, peak VRAM, and GPU-hours;
- paired per-image plot, budget–accuracy Pareto curve, accepted/rejected/unknown gallery, and three failure galleries.

## FSC-147 preflight and execution gate

Existing VAL preflight: 1,286 images, 68,587 proposals, median 23, p90 134, max 900. Verify-all is estimated at roughly 48 GPU-hours on one 3090-class GPU. Budget 16 routes at most 16,217 baseline tasks (about 11.4 GPU-hours encoding+unary before sharding); budget 8 routes 8,796 tasks.

Before using an old prediction file, verify annotation path, split, checkpoint hash, config, and image count. The current preflight prediction is workload evidence only until its exact standard CountGD++ protocol is certified.

## Claim gates

- **External transfer claim:** improvement CI excludes zero on FSC-147 TEST and at least one domain-shift dataset, with no material safety regression.
- **Scalability claim:** SER lies on a documented accuracy–GPU-hour Pareto frontier and deferred proposals never change counts.
- **General controller claim:** requires improvement on at least two base counters; otherwise state explicitly that FQIE is evaluated around CountGD++.
- **SOTA claim:** allowed only under identical prompt/split/training protocol and direct compatible baselines. Otherwise report improvement over frozen CountGD++ and comparisons without “SOTA”.

## Primary sources

- CountGD (NeurIPS 2024): https://papers.nips.cc/paper_files/paper/2024/hash/57c56985d9afe89bf78a8264c91071aa-Abstract-Conference.html
- FSC-147 (CVPR 2021): https://openaccess.thecvf.com/content/CVPR2021/html/Ranjan_Learning_To_Count_Everything_CVPR_2021_paper.html
- ZSC (CVPR 2023): https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Zero-Shot_Object_Counting_CVPR_2023_paper.html
- PseCo (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/html/Huang_Point_Segment_and_Count_A_Generalized_Framework_for_Object_Counting_CVPR_2024_paper.html
- DAVE (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/html/Pelhan_DAVE_-_A_Detect-and-Verify_Paradigm_for_Low-Shot_Counting_CVPR_2024_paper.html
- T2ICount (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/papers/Qian_T2ICount_Enhancing_Cross-modal_Understanding_for_Zero-Shot_Counting_CVPR_2025_paper.pdf
- CountSE (ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025/html/Liu_CountSE_Soft_Exemplar_Open-set_Object_Counting_ICCV_2025_paper.html
