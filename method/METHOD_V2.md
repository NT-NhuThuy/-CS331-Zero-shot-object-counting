# UNICOUNT v2 — Presence-Aware Active Evidence and Identity Counting

## Research question

For an image `I` and target-language query `q`, predict `N(I,q) ∈ {0,1,2,...}` even if the target is absent, with strong positive-count retention, identity consistency and a finite verification budget. A single numerical answer and an uncertainty/abstention trace are needed; the two errors of greatest interest are false presence (`N=0, Nhat>0`) and false zero (`N>0, Nhat=0`).

## Scientific contributions (proposed)

**C1 (Thư), count-impact-aware action selection:** Maintain TWO levels: (i) image-query presence/search-coverage risk that still works when the detector returns zero candidates, and (ii) candidate/identity count-impact risk when boxes exist. Use detector confidence, V4 structural contributions, A3 semantic scores, multi-view support and graph conflicts to prioritize the verification/recovery/identity query most likely to improve the count outcome per cost. Compare against score, uncertainty, random and fixed schedules. Historical FSC V4/A3 NPZ is not transferable without exact-key reconstructing the same P02 candidate nodes.

**C2 (Lộc), adaptive evidence acquisition:** A single frozen CountGD++ operating point produces baseline and reprojected multi-view candidates. On demand, Qwen provides target semantic, whole-instance and pairwise physical-identity evidence. Record model revision/prompt/checkpoint/protocol/source hashes, latency and failures/UNKNOWN, including on query-absent images. No evidence is promoted from generic Qwen image-level verdict to a region verdict.

**C3 (Thúy), joint presence/count/identity arbitration:** Maintain graph nodes for baseline and rescue candidates and edges `SAME/DIFFERENT/UNKNOWN`. Reason under hard/soft compatibility constraints; suppress double counts, preserve real positive instances despite misleading global absence evidence, and return `zero`, `positive K` or `abstain` with trace. Current greedy complete-link and dual-NO rules are comparison baselines, not declared A* novelty.

## Proposed state and action loop

Let `G_t` contain candidate geometry, embeddings/scores, source view, semantic/whole verdicts, and pair identity evidence. Maintain hypotheses `N=0`, `N>0`, and unresolved. At each step choose an action from `{REGION_SEMANTIC(i), REGION_WHOLE(i), RECOVER_VIEW(v), PAIR_IDENTITY(i,j), STOP}` until cost limit, decision sufficiency or fail-safe stop. A conceptual acquisition rule is:

`a* = argmax_a ExpectedDeltaCountRisk(G_t,a) / EstimatedCost(a)`

This is a **design objective**, not a validated calibrated probability or a learning algorithm. Initially implement a deterministic count-impact proxy and log utility/cost; on approved TRAIN/CAL only, fit/calibrate a risk model. Do not turn the historical scalar Joint V3.2 output into instance labels by fiat.

## Proposed count inference

Each node has target-validity state; same-object edges form equivalence hypotheses subject to source-view, geometry and evidence consistency. The count is the number of supported, distinct target-object components, not the number of boxes. Unknown and contradictory evidence must be represented rather than automatically coerced to 0 or 1.

Do not claim absent just because zero boxes survived. To issue count=0 as an evidence-backed decision, an explicitly approved absent policy must be met (adequate source/search coverage, valid negative evidence and no reliable positive region). This is still a *statistical decision*, not a proof of physical absence. If evidence is inadequate, allow abstain and pre-register a separate forced-output fallback for comparable MAE.

## Evidence levels

- P03 111 hard negatives, FSC147 TEST and historical V14/PrACo: `EXPOSED_REFERENCE` for diagnostics, NOT fresh evaluation.
- Existing `countlab` P06/online/replay software: CPU/protocol-level proof, not full GPU UNICOUNT v2 results.
- Only a newly audited source-disjoint TRAIN/DEV/CAL/FINAL_SEALED protocol can support v2 calibration and confirmatory claims.

## Strong comparisons needed for novelty

CountGD++ fixed protocol; tuned CountGD++ threshold; global Qwen gate; semantic-only, whole-only and region dual-NO; full verify; recovery-only; score-based and random router; greedy NMS/Union-Find/complete-link; fixed one-pass combination of all three; **active joint UNICOUNT v2** at the same compute/prompt/backbone/sources. Report absent zero accuracy AND present false-zero rate, present MAE/RMSE, instance false-merge/split, coverage/abstention and real latency/tokens/GPU costs.

## Stop rule

If one-pass Qwen gate+simple NMS or a calibrated threshold matches v2 on source-disjoint DEV at matched cost, do not claim joint reasoning novelty. If abstention or false zero is unacceptably high, revise presence-evidence policy before expanding the system.
