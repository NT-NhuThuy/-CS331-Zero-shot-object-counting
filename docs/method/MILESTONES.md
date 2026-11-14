# One-method execution plan to reach an A* submission (NOT a guarantee of acceptance)

## M0 — Software skeleton completed in this branch (CPU only)

- [x] Shared candidate/evidence schema; rejection/recovery/identity replay and GT-input guard.
- [x] One `countlab-method` CLI; deterministic toy fixture; create-only output; source SHA.
- [x] Exact historical FQIE task-index adapter; old trees remain unchanged.
- [x] Owner map, CI, methods document and source-level tests.
- [ ] Running CountGD++ model, Qwen verifier and learned router connected end-to-end.

## M1 — Frozen backbone parity (NO benchmark claim without this)

**Shared + Lộc**: produce native CountGD++ boxes in the exact source operating point, with model/exemplar/prompt hashes. **Thúy**: import V14 raw tasks and unary; verify old frozen results exactly, document any deviation. **Thư**: audit FSC/PrACo/LVIS regimes: inputs, supervision and source splits. Exit: native-model parity, GT-blind input inspection, all old metrics tagged by version; no test overlap used for tuning.

## M2 — Real shared method running on GPU

**Lộc:** implement per-instance `UnaryVerifier` and raw multi-view proposal adapter (not global Qwen zero-gate). **Thư:** implement `RiskAdapter` with benchmarked GT-blind features + budget policy; learned head must be trained only on allowed development data. **Thúy:** implement pairwise identity evidence, original-image coordinate parity, and non-bridge association. Exit: a complete **single-script** online run with stable logs, source hashes, cost stats, no GT fields and pre-registered fallback policy.

## M3 — New algorithmic core, one shared novelty hypothesis

Build a joint risk- and evidence-consistent arbitration mechanism, not a stack of unrelated filters. Explicit contradictory evidence policy, known/unknown identity constraints, verification budget, and either a nontrivial theoretical property or demonstrated new mechanism. Exit: beat independent dual-NO rules, optimized threshold, Qwen image-level veto, NMS, simple graph clustering and matched-cost routing on NEW DEV/CAL. If not, simplify rather than claim novelty.

## M4 — Real evaluation and SOTA attempt

- New independently sourced LVIS-derived test / category-held-out and hard identity annotations.
- Standard counting datasets with **matched prompt, supervision, checkpoint and protocol**; class-mismatch PrACo++/MUCCA in native form where available.
- Baseline CountGD++; adapted DAVE/ValidCounter/CountSE if protocols align; threshold and model-cost baselines; FSC-147 TEST already exposed, not independent holdout.
- Per-image/cluster bootstrap CIs, MAE/RMSE, present/absent, semantic region precision, false merges/splits, risk–coverage, latency, calls, peak VRAM.
- Release reproducible checkpoints/configs and a **pre-registered** final test; no tuning after seeing FINAL_SEALED.

## M5 — A* paper gate

Method must answer: (1) new technical idea compared with closest prior art, (2) controlled ablation showing WHY it helps, (3) strong/fair empirical results, (4) reproducible source/threshold provenance, (5) clear limitations. Conference rank/acceptance and SOTA are targets, never guaranteed.
