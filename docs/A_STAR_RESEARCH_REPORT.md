# CS331 / CountGD++ — Tổng kiểm toán repository và chiến lược nghiên cứu hướng A*

**Audit snapshot:** 2026-11-14 (Asia/Ho_Chi_Minh)  
**Source:** user-supplied Git bundle `cs331-all.bundle` with four Git refs; offline inspection of code, tables and selected tests; additional contemporary literature sources cited below.  
**Nature of this report:** evidence-driven research decision and local non-destructive codebase reorganization, **not** a claim of having trained or re-run all models.  
**Target:** CVPR / ICCV / NeurIPS-level methodological contribution and competitive, *matched-protocol* counting/semantic robustness. No promise of publication or SOTA.

## 0. Executive research decision

**One scientific story:** *When a frozen open-world counter is uncertain, can one correct BOTH falsely included and missed physical instances by adjudicating local evidence, preserving identity consistency, and controlling the risk/cost of rejecting true detections?*

**Recommendation:** Keep Thúy's FQIE bidirectional controller + Safe-IIC as the **primary candidate**, not an accepted final method. Treat Thư's learned HOW-MANY correction (Joint V3.2) and A3 semantic gate as meaningful **alternative correction baselines** and possible risk-estimation ablations. Treat Lộc's Qwen3-VL verification and PrACo official adapter as indispensable **semantic verifier baselines and robustness evaluation infrastructure**, and V7.5 multi-scale as an optional **candidate recovery baseline**. Never stack all three automatically.

**A* novelty is NOT established:** detect-and-verify and semantic filtering already exist (DAVE 2024, ValidCounter 2025, CountSE 2025); CountGD++ 2026 supports positive/negative multimodal prompting. Bidirectional + UNKNOWN rules alone can be incremental. The strongest route is a **new, explicit evidence arbitration / constrained instance selection / risk-controlled abstention mechanism**, with direct matched controls, multi-dataset testing and instance-level evaluation. This mechanism is a **research proposal**, not existing code or proven novelty.

**SOTA is NOT demonstrated:** Current results belong to substantially different datasets, operating points and levels of prior exposure. They cannot be stitched into one leaderboard.

## 1. Git audit and exact snapshots

| Ref | Git SHA | Audit disposition |
|---|---|---|
| `main` | `f8cfe6857ded52acca2a72b8b08192c6cb308fa8` | merged code snapshot; no unified, reproducible root pipeline |
| `feature/thu-joint-semantic-counting` | `c945ea53b904fa37c4d79acc5e9901464c235100` | Thư's self-contained learned correction scripts + limited frozen predictions |
| `loc/research-v4-v75-2026-11-06` | `86aefa41599d290f408f1274c77fcdb0cb486272` | Lộc's V4–V7.5 notebook program |
| `loc/praco-official-adapter-v76-full` | `6c86d30304bb55a21d31bcd6f5aa0f4289cccfb6` | Lộc's significant V7.6 official PrACo adapter and sealed A/B archives |

Original `main` tracked 530 files (69 Lộc, 25 Thu, 435 Thuy, plus root README). The V7.6 branch adds 23 files and updates `Lộc/README.md`; these additions were imported into `legacy/loc` in the new local branch. The original Git history is retained in the reorganization **bundle**; the user must explicitly push/merge to their remote.

Local audit tests: **8 passed** in new `countlab` contracts; **39 passed** in `Thuy/branch` on original main; **2 passed** for Lộc's PrACo verifier in the official adapter branch; independent local PrACo artifact checker reproduced recorded *internal* A/B summary numbers. A CPU read-only recomputation from Thư's archived 1,301 paired CSV rows reproduced baseline MAE **15.9907763** and Joint V3.2+A3 MAE **13.2280458**, identifying **1,190 distinct source-image clusters**. Its illustrative 1,000-replicate source-image bootstrap CI of method-minus-baseline MAE was **[-3.4068, -2.2155]**, but remains **EXPLORATORY / TEST-exposed**, not independently confirmatory. No GPU end-to-end CountGD++ or Qwen inference was performed here. Some required caches, checkpoints, image assets, manifests and full V14 inference records do not exist in the Git mirror.

## 2. Provenance caveat: timestamps and the locked V14 holdout

The repository includes `Thuy/branch/FQIE_BIDIRECTIONAL_HOLDOUT_RESULTS_20261116.md` and similarly labelled result files. The current local date is **2026-11-14**, so these `20261116` names refer to a **future** date. Do not treat filenames or authored dates as independent chronological proof. The summary claims V13 training/development -> freeze -> source-disjoint V14, but the mirror lacks complete inference outputs and source split manifests to verify chronology, leakage, or replicate the 180 predictions.

Evidence **in mirror**: aggregate metrics, per-arm ablation, bootstrap summary, failure diagnostic for the three regressions, and stored SHA values. Evidence **missing from mirror**: the actual sealed full-cohort inference outputs claimed by SHA, source membership manifests, original frozen unary decision files, checkpoint/inputs, and independent pre-unblinding freeze evidence. **Do not convert reported results into independently replicated findings.** The 37.54% result is promising, not yet our independently confirmed result.

## 3. Three owners — methods, evidence, limitations, keep/remove

### 3A. Thư: Joint Semantic Counting (`legacy/thu`)

**Mechanism actually represented:** `V4` reweights structural proposal/node contribution; `Joint V3.2` uses structural and semantic reliability features and trained risk/candidate models to decide count correction; `A3` semantic verifier changes image-query count to zero when a presence check rejects. Source: `legacy/thu/scripts/p14_08b_09_joint_unified.py`, `train_s1_semantic_verifier_a3_clip_v1.py`, `RESEARCH.md`, `p17_01_three_view_paired_v1/three_view_metrics.csv`.

**Key numbers**, 1,190 FSC-147 TEST positive + 111 hard-mined absent, **already observed**:

| Arm | Positive MAE | Negative MAE | Unified MAE | Unified exact % |
|---|---:|---:|---:|---:|
| CountGD baseline (this specific protocol) | 11.6504 | 62.5225 | 15.9908 | 30.21 |
| V4 seed42 | 10.9379 | 61.2487 | 15.2303 | 32.82 |
| Joint V3.2 | 10.1488 | 61.2487 | 14.5086 | 33.82 |
| Joint V3.2 + A3 | **10.1488** | **46.2394** | **13.2280** | **37.20** |

**Strength:** genuine structural/count correction mechanism beyond a binary global VLM veto; can provide a technically meaningful counterfactual to FQIE's reject/accept strategy. **Risk:** trained correction heads mean “frozen CountGD++ backbone” is not “training-free end-to-end”; fractional counts and graph features do not map naturally to physical-instance IDs or auditable set membership. Training and frozen model artifacts (`locked_p11_models.joblib`, lock manifests, decoder/graph caches) are not all in Git; `p14_08b_09_joint_unified.py` asserts missing paths, so a clean GPU run is not established. Dataset TEST/hard negatives were used for iterative development.

**Decision:** **KEEP** as a named learned count-correction baseline and exploratory risk-score source. **DO NOT** combine its scalar/fractional prediction with an instance count without a new model of how/why the bridge works, a new protocol version, and an independent dev split. A3 remains a global semantic-gate baseline, not default production veto.

### 3B. Lộc: Dual Evidence / Multi-scale / PrACo (`legacy/loc`)

**Mechanism:** V4 Qwen3-VL proposal verification -> V5 native negative text/visual prompts -> V6 guarded router -> V7 dual-evidence reject/router -> V7.2 repaired parser/zero-only tiling -> V7.3 context verification -> V7.4 high-resolution 8×8 -> V7.5 multi-scale 4×4/8×8/12×12; V7.6 native official CountGD++ PrACo Test1 adapter with Qwen's two sequential checks. Most implementations are notebooks and therefore need extraction into independently testable modules before any verified fusion.

**Earlier FSC-147 exploration:** On 1,190 TEST positive and 111 old negatives, Lộc's CountGD++ baseline positive MAE **17.78** and negative MAE **62.55** (different protocol from Thư: never compare directly); V7 positive MAE **18.24**, negative **4.33**; V7.4 positive **15.79**, negative **4.33**, but improvement concentrated in marker-heavy `1123.jpg`. V7.5 validation 480 positives across two runs: baseline **12.64**, V7.5 **9.81**, **only 6 images changed**; absent safety not demonstrated on this subset. V7.1 increased negative MAE to 12.27 and is a failed branch.

**PrACo Test1 official-adapter V7.6 audit** (34,510 pairs = 1,190 images × 29 queries; full sealed A/B archives, 4,303 Qwen checks, 30,207 baseline zeros untouched; same 34,510 pair IDs):

| Metric | Native CountGD++ A | +Qwen V7.6 B | Evaluation |
|---|---:|---:|---|
| Class-mismatch nonzero | 9.3427% | **0.2161%** | strong semantic suppression |
| AvgNMN | 0.08110 | **0.00204** | greatly reduced |
| PCCN | **97.9832%** | 97.0588% | worsens |
| Positive MAE | **12.1807** | 13.5286 | worsens |
| Positive RMSE | **100.0072** | 115.1455 | worsens |

Out of 3,060 suppressed positive baseline counts, 3,041 were class-mismatch queries and **19 were actually positive-class queries**. Class mismatch != independently verified visual absence. Published experiment already viewed official TEST; not an untouched validation. Original official adapter docs flag an **unverified mirror checkpoint bitwise equivalence** against official author weights.

**Decision:** **KEEP** V7.6 official A/B source, verifier tests, provenance hashes and semantic suppression as MUST-RUN baseline; keep V7.5 multi-scale as recovery ablation; V7 dual-evidence as verifier comparison. **ARCHIVE** failed V7.1 and V7.4’s isolated dense case, plus purely exploratory V4–V6/unsafe global zero rules, without erasing history. **DO NOT** promote whole-image all-zero gating to FQIE's physical-instance gate without calibration and demonstrated false-removal safety.

### 3C. Thúy: FQIE PR/IIC + bidirectional correction (`legacy/thuy`)

**Method represented in code:** frozen CountGD++ baseline with GT-blind multi-view candidate generation, cross-view consensus, unary semantic and whole-instance verification, Safe-IIC identity compatibility; later **bidirectional** controller additionally adjudicates each baseline detection and permits unmatched accepted rescues. Important files: `branch/bidirectional_evidence_controller.py`, `prepare_bidirectional_verification.py`, `route_bidirectional_tasks.py`, `freeze_bidirectional_decisions.py`, `safe_iic.py`, `build_evidence_output.py`, `assemble_bidirectional_full_cohort.py`.

**Historical:** V7 `MAE 2.044→1.967`; V8 `3.078→2.994` source-disjoint exploratory audits (earlier method); V11 fair `3.5167→3.4944` (sign-test p=0.125), V12 fair `1.4333→1.4278` (p=1.0; absent false promotion **2→3**). These cannot validate the new Safe-IIC globally. V5 raw union and V9 ownership caused overcount; earlier identity merged 49 candidates into 1 / 55 into 3 on problematic dense queries.

**Bidirectional V14 claim as recorded in repo:** 180 (90 present/90 verified-absent), full MAE **1.761111→1.100000**, present **3.355556→2.200000**, absent **0.166667→0**, exact **56.67→60.56%**, wins/losses/ties **22/3/155**, source-cluster bootstrap 95% CI for delta **[-1.3833,-0.1833]**, sign test two-sided **0.00015652**. Reported post-freeze ablation: baseline **1.7611**, rescue **1.7444**, reject **1.1167**, full **1.1000**. At the cohort average level, rejection accounts for ~97.5% of MAE reduction; rescue contributes ~2.5%, but causal explanation requires stronger controls. Three over-pruning cases: source-image IDs `4130`, `555050`, `142454` (report-provided diagnostic, not manually re-verified image).

**Code/contract issues to address:**

1. `bidirectional_evidence_controller.reconcile()` reports an `exact` candidate-set interval only when no unknowns; that does **not certify complete scene count**, and the interval can be uninformative if candidate recall is incomplete. `query_abstained` always `False` in this function. Need explicit semantics for **partial count vs true scene count**.
2. In `freeze_bidirectional_decisions.py`, `ACCEPTED` requires YES/YES, `REJECTED` requires NO/NO, all others UNKNOWN; this is conservative but can fail category-only mismatches and still over-prunes positive objects (three reported regressions). “UNKNOWN monotone” is not a measured false-reject guarantee.
3. Rescue matching at IoU 0.5 is geometric matching, not full physical-identity proof, and cross-view duplicates or adjacent instances need independent true identity annotations. The original Safe-IIC logic should be evaluated as an identity graph, not treated as automatically validated by unit tests.
4. `assemble_bidirectional_full_cohort.py` preserves fallback baseline point count, but produces `exact` with zero baseline count even though zero detections do not certify absence.
5. A source-image-cluster bootstrap is a good choice, but sign test over 25 changed pairs should not be described as an independence-aware cluster test when multiple queries share images; disclose grouping.
6. Raw V14 sealed inference/GT-free prediction files and selection chronology are not in this Git mirror; hash labels alone cannot substantiate freeze order.

**Decision:** **KEEP as primary research candidate** provided fresh validation passes. **SAFE-IIC + bidirectional arbitration** is the best coherent scientific direction, **NOT** proof of a new method yet. Do not use V14 to retune after observing these results.

## 4. How the three methods interact: no fictional synergy

| Slot | Primary proposal | Alternative / control | Why not stack unconditionally? |
|---|---|---|---|
| Baseline backbone | frozen CountGD++ matched mode | additional frozen detector for transfer | checkpoints/prompts/exemplars differ now |
| Count calibration | FQIE identity-based integer count | Thư Joint V3.2 | fractional learned scalar vs audited physical-instance set |
| Global semantic evidence | lightweight category check if proven reliable | Thư A3 / Lộc Qwen V7.6 | hard zero-veto can remove true instances |
| Proposal recovery | Thúy GT-blind PR + consensus | Lộc V7.5 multi-scale | tiled union can create part/duplicate hallucinations |
| Unary verification | semantic and whole YES/NO/UNKNOWN | Lộc dual-evidence, Thư A3 | different input (crop vs full image) and training/cost |
| Identity | Safe-IIC | greedy NMS / IoU / pairwise Union-Find | geometry alone cannot prove SAME; transitive collapse |
| Reporting | FQIE candidate sets + trace + fallback | scalar-only baselines | partial observed-candidate interval is NOT certified truth interval |
| Robustness benchmark | source-disjoint LVIS-derived, PrACo++ MUCCA where compatible | official PrACo Test1 and FSC-147 | category mismatch vs verified absence vs FSC count differ |

**Strict compatibility rule:** any merger changes a method version. Before fusion: one common pair-ID/image-ID/prompt/checkpoint schema; expose unique object-level boxes, same coordinate system, provenance, feature budgets, GT-blind decisions and null/UNKNOWN semantics; GPU parity on a prospective validation set. On violation, leave components as baselines.

## 5. Novelty audit: what threatens A* acceptance

| Intended claim | Nearest known prior | Verdict |
|---|---|---|
| Detect then verify false positives | DAVE (CVPR 2024) | NOT new standalone |
| Training-free spatial + semantic filter | ValidCounter (Information Sciences 2025) | NOT new standalone |
| Multi-scale semantic exemplars / filtering | CountSE (ICCV 2025) | NOT new standalone |
| Negative prompting and pseudo-exemplars | CountGD++ (CVPR 2026) | NOT new standalone |
| Prompt-robustness evaluation | PrACo (WACV 2025), PrACo++ / MUCCA (2026) | evaluate against, do not rename as new benchmark |
| Dense scale handling | GeCo2 (AAAI 2026), SAM3Count (CVPRW 2026) | must compare cost and mode fairly |
| Bidirectional correction on frozen counter | Partial overlap with detect-verify and calibration | POTENTIAL combination; novelty NOT yet proven |
| Explicit identity/uncertainty-aware evidence arbitration | Need closer comparison including selective prediction, structured grounding/graph consistency | BEST novel-mechanism candidate, requires actual algorithm and new validation |

Relevant primary literature:

- CountGD++ CVPR 2026: https://robots.ox.ac.uk/~vgg/publications/2026/AminiNaienib26/
- DAVE CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Pelhan_DAVE_-_A_Detect-and-Verify_Paradigm_for_Low-Shot_Counting_CVPR_2024_paper.html
- ValidCounter 2025: https://www.sciencedirect.com/science/article/pii/S0020025525002932
- CountSE ICCV 2025: https://openaccess.thecvf.com/content/ICCV2025/html/Liu_CountSE_Soft_Exemplar_Open-set_Object_Counting_ICCV_2025_paper.html
- PrACo WACV 2025: https://openaccess.thecvf.com/content/WACV2025/html/Ciampi_Mind_the_Prompt_A_Novel_Benchmark_for_Prompt-Based_Class-Agnostic_Counting_WACV_2025_paper.html
- PrACo++/MUCCA 2026: https://arxiv.org/abs/2605.02752
- GeCo2 AAAI 2026: https://ojs.aaai.org/index.php/AAAI/article/view/37780
- SAM3Count CVPRW 2026: https://github.com/Joan947/SAM3Count
- GroundCount 2026: https://arxiv.org/abs/2603.10978

This is a prioritized audit, not proof that *no* closer paper exists. Re-run a systematic literature search before submission.

## 6. Proposed A* research thesis — not yet an implemented contribution

**Working thesis:** *Risk-Aware Evidence-Constrained Counting* (working name only). Recast counting as **constrained selection of physical instances** from incomplete, correlated detector proposals with bidirectional evidence and explicit abstention.

Let candidate graph `G=(C,E)` contain baseline/recovery proposals, local semantic/whole evidence with three-valued states, pairwise identity evidence (SAME/DIFFERENT/UNKNOWN), geometric compatibility and source-view provenance. A future method may jointly select accepted instances and identity components under a query-specific risk/compute budget, instead of blindly applying isolated binary gates and union-find.

**Potential technical contribution A (highest priority):** a principled **evidence arbitration operator** that handles contradictory positives/negatives, identity uncertainty, duplicate proposals and bounded-cost verification; deterministic invariants (e.g., no UNKNOWN promoted, no candidate added twice, consistency across all component pairs). The method must beat Qwen-only, confidence-only, DAVE-style filters, and Safe-IIC's simpler rules. New algorithm and proof/audit are required.

**Potential contribution B:** **calibrated selective correction**: use held-out calibration to bound or estimate risk of falsely removing a real instance, report empirical risk–coverage and simultaneous absent safety. A statistically valid guarantee needs exchangeability/assumptions and a prospective calibration split. Don't call this conformal/certified without formal theory and demonstrated validity.

**Potential contribution C:** **query- and evidence-aware verification allocation** at matched Qwen/GPU budgets: decide WHICH baseline detections or rescue proposals deserve extra calls. Use Thư's uncertainty/structural signals, Lộc's verification findings, and Thúy's candidate graph, but compare to confidence, entropy, random routing and straightforward margin baselines. If simple policies win, do not claim a learned intelligent router.

**Optional alternative paper:** robust semantic counting with a learned structural+semantic count corrector on a standard public benchmark. That is a DIFFERENT claim and a different training setting. Do not combine it into the evidence-certification paper unless the mechanism is unified.

## 7. Precise research questions and falsifiable tests

**RQ1, mechanism:** On truly annotated hard cases, does joint local evidence+identity arbitration reduce false rejection, false merges/splits and duplicate counts relative to (a) independent binary verifier, (b) NMS/IoU matching, (c) ordinary threshold filtering at identical budget?

**RQ2, asymmetry:** Rejection dominates V14. Is the gain better than (a) shifting detector confidence threshold, (b) calibrating a scalar prediction to reduce overcount, (c) full-image Qwen veto (Lộc V7.6), (d) learned Joint V3.2 (Thư)? Compare matched cohorts and include undercount and absent strata.

**RQ3, coverage:** Does PR + whole/semantic gates recover *new* physical target instances (not parts/duplicates) without materially increasing absent false promotions? Evaluate target boxes, candidate precision, recall, AP or matched-instance F1 when GT supports it.

**RQ4, uncertainty:** Are `UNKNOWN` and abstentions correctly calibrated and actually reduce selective error at lower coverage? Compare to random abstention and maximum-score threshold controls. Mark candidate-set bounds as observational unless assumptions justify scene-wide coverage.

**RQ5, transfer:** Does the method generalize across source-disjoint images, new categories, densities, image sources and a SECOND frozen backbone without per-test tuning? If not, narrow the claim.

**RQ6, efficiency:** Is improvement preserved at equal GPU/VLM calls, wall time and peak VRAM? Compare risk–coverage–cost Pareto frontiers.

## 8. Mandatory evaluation matrix for SOTA-quality evidence

| Benchmark/setting | Metric to publish | Scientific guardrail |
|---|---|---|
| New LVIS source-disjoint image/query test | full count MAE/RMSE/bias/exact; present/absent; box AP/recall/instance matching | source-disjoint; GT only after freeze |
| Multi-category external dataset, e.g. MUCCA / PrACo++ if available | category grounding, class mismatch, distractor error, absent verified when annotation exhaustive | not equivalent to old PrACo Test1 |
| FSC-147 official VAL/TEST as applicable | MAE/RMSE; density/center metrics compatible with annotations | historical official TEST used for development; NEW method selection cannot re-use it as untouched holdout |
| PrACo Test1 official matched | AvgNP, AvgNMN, PCCN, positive MAE, class-mismatch nonzero, false suppression | use *native* CountGD++ baseline; disclose TEST exposure |
| Dense/scale transfer (e.g., OmniCount/Fruits or FSCD-147) | MAE/RMSE, high-count slices, runtime | ensure mode/input and annotation compatibility |
| Instance-level manual hard cases | semantic / whole correctness, false merge/split, unknown, localization | at least two independent annotations for critical uncertain cases |
| Second frozen backbone | gains and tradeoffs vs its matched baseline | report failure as well as success |

**Paired reporting:** report full-cohort outcomes with baseline fallback; source-image clustered bootstrap and paired effect sizes, calibration curves, per-query scatter, exact-count accuracy, absent false-promotions, identity mistakes and 95% CI; do not hide negative branches. If 90 present/90 absent sampling is artificial, present strata separately and avoid claiming deployment-prevalence performance.

**Strong reference comparisons:** original CountGD++ matched mode; tuned simple score threshold/correction; Qwen V7.6 baseline; Joint V3.2; DAVE/ValidCounter/CountSE when input/supervision/budget allow reproducible matched evaluation; prompt-aware PrACo(++), GeCo2 and SAM3Count in comparable regimes. A table should explicitly label when different inputs/supervision make direct numerical comparison invalid.

## 9. Ablations to run, and the required control per module

| Arm | Why | Required control |
|---|---|---|
| Frozen CountGD++ | true baseline | same checkpoint, exemplars, prompts |
| Baseline + calibrated threshold | checks rejection triviality | threshold set on training/dev, not test |
| Baseline + Lộc Qwen V7.6 | global semantic rejection | equal calls/latency or Pareto |
| Baseline + Thư Joint V3.2 | learned count correction | label training split and fairness |
| FQIE rescue only | tests recovery | raw proposal-only / no whole check |
| FQIE rejection only | tests filtering | confidence-only and semantic-only/whole-only |
| FQIE reject + rescue | reported V14 variant | separate from new graph algorithm |
| + Safe-IIC vs NMS vs naive Union-Find | tests physical identity | GT-annotated false merges / splits |
| + calibrated abstention | tests uncertainty | matched coverage random/score-based |
| + adaptive verifier budget | tests cost allocation | random / margin / score baseline |
| Full proposed joint arbitration | new claimed contribution | module ablations + grouped bootstrap |

Use **new independent validation** for any new thresholds, gates or feature choices. V14 is now exposed by design; do not optimize based on its 3 failure cases and then re-test it as unseen.

## 10. Actual repo refactor delivered here

This local branch **moves the complete historical trees** intact to `legacy/loc`, `legacy/thu`, `legacy/thuy`, imports Lộc's V7.6 branch additions, and provides:

- `src/countlab/`: explicit owner/method registry; evidence/candidate schema; conservative decision utility; CREATE-ONLY SHA manifests; paired cluster bootstrap evaluator; file inventory.
- `configs/regimes/`: separate non-interchangeable LVIS query-only, FSC-147 exploratory and PrACo official modes.
- `tests/`: CPU contract and audit unit tests.
- `.github/workflows/ci.yml`: CPU validations of new code and archived existing suites.
- `paper/claim_ledger.csv`: immutable statement of claims and their evidential restrictions.
- `docs/`, `experiments/`, `paper/`: research governance, protocols and the A* roadmap.

**Not implemented by this refactor:** a fake GPU adapter claiming all methods fused, a new learned graph optimizer, a sealed validation set, repaired historical inference, or a remotely pushed PR. Preserve branch history in the delivered Git bundle.

## 11. Staged A* roadmap — GO/NO-GO gates

### P0. Lock foundations (required before any new test)

- Reconstruct on real server each of the three CountGD++ operating points (Thu FSC, Lộc PrACo, Thúy LVIS). Record checkpoint SHA, pseudo exemplars, input mode, model revision, preprocessing and parity checks.
- Retrieve V14 source-image manifest, raw sealed predictions, stage decisions, and timestamped freeze proofs. Compare SHA with mirror. Until present, call V14 a *reported* holdout.
- Split true TRAIN / DEVELOPMENT / CALIBRATION / FINAL_SEALED with source-image and class leakage checks. Tag all previously seen FSC TEST, negatives, V13 and V14 as exposed.
- **Gate:** 100% baseline pair matching + no GT on inference path + versioned protocols; otherwise halt metric comparisons.

### P1. Prove the anomaly and independent value of modules

- On fresh dev: threshold baseline vs FQIE rejection; Qwen full-image veto vs region verification; Joint V3.2 and A3 vs conservative arbiter; Safe-IIC identity vs NMS/Union-Find; V7.5 vs PR.
- Audit 3 reported V14 false rejections descriptively only (no tuning). Collect independent instance-level annotations for false merge/split and semantic/whole verdicts.
- **Gate:** evidence arbiter must beat simple cost-matched controls on corrected count or verification quality without unacceptable false removals. If not, primary method is no-go; refocus to benchmark/analysis contribution.

### P2. Research genuinely new mechanism

- Formulate evidence-constrained consistent selection with uncertainty and cross-view identity; implement a **single explicit algorithm** with invariants and conflict resolution, not spaghetti gates.
- Establish computational complexity, objective/assumptions, failure behavior and clean CPU synthetic tests for bridge edges, ambiguous identity, adjacent targets, multiple-object crops, absent/no detections.
- Explore calibrated false-rejection risk (if statistical assumptions defensible), with selective abstention and cost-aware verifier call allocation.
- **Gate:** meaningful measured advance over P1 controls in 2+ independent dev regimes, ablation isolates novelty; otherwise do not claim technical novelty.

### P3. Paper-grade independent evaluation

- Full run: 2+ external domain/category regimes, at least one dedicated instance-annotated dataset, 2 backbones if feasible; report full-cohort MAE, absent, risk, identity and cost.
- Strong baselines at matched prompt/exemplar/label/compute budgets; explicit non-comparable literature rows separately.
- Statistics: clustered CI, paired tests, multiple-comparison awareness; inspect failure distributions, not merely percentage gains.
- **Gate:** improvement robust or scope explicitly limited; no SOTA claim without verified leaderboard-compatible protocol and enough baselines.

### P4. Paper packaging and red-team review

- Intro should **not** lead with “we add a verifier to CountGD++.” Instead motivate contradictory evidence and physical-instance uncertainty; show counterexamples and quantitative safety.
- Main figure: baseline mistakes -> candidate graph -> joint evidence arbitration -> accepted/rejected/unknown with provenance. Main tables: reproducible SOTA-compatible results, PrACo robustness tradeoffs, risk–coverage and runtime, instance-level error, ablation and negative results.
- Red-team check: novelty relative to DAVE, ValidCounter, CountSE, CountGD++; no GT in inference/thresholds; missing artifacts; original code licenses and source attribution; dates and frozen split proof.
- **Gate:** independent researcher reproduces major results from published manifest/code/checkpoints; claim ladder survives review.

## 12. Responsibility matrix (proposal, not authorship attribution decision)

| Owner | Primary engineering/research task | Cross-review by |
|---|---|---|
| Thúy | Bidirectional FQIE, Safe-IIC, evidence graph, uncertainty/abstention, LVIS source provenance | Thư checks metrics / Lộc checks false rejections |
| Thư | Joint V3.2/A3, calibration and scalar threshold baselines, feature utility, grouped statistics, parity auditor | Thúy verifies GT-blind split and matching |
| Lộc | Qwen per-instance/global verifier comparison, V7.5 multi-scale, PrACo official + PrACo++/MUCCA audit, cost & GPU reproducibility | Thư checks train-vs-test fairness / Thúy checks evidence trace |
| All | common protocol, external test freeze, hard-case annotation, paper/figures and independent sanity reviews | rotate reviewer per module |

**One-owner-one-module is not one-owner-one-paper contribution:** author credit and ordering should follow actual intellectual and practical contributions, agreed by the team.

## 13. Primary claim ladder

**NOW defensible as internal descriptive findings:** three independent directions contain meaningful exploratory signals; V14 *reported* full-cohort count MAE decrease 37.54%; Lộc PrACo semantic suppression has a measured positive-cost tradeoff; Thư Joint+ A3 has unified FSC exploratory MAE improvement.

**AFTER provenance parity:** “On one pre-frozen, source-disjoint LVIS-derived holdout, our frozen-backbone bidirectional controller improves CountGD++ count MAE while presenting instance-level evidence states.” Specify it is *one* cohort and method is not independently proved universal.

**AFTER algorithmic novelty + instance annotation:** “A constrained identity/uncertainty-aware evidence arbitration method improves both corrected physical-instance validity and scalar count under a fixed risk and compute budget, beyond simple detect-and-verify controls.”

**ONLY AFTER benchmark-compatible external comparisons:** claim SOTA on **named benchmark, named input/supervision setting and exact metric**. Do not claim SOTA in general object counting.

**Prohibited now:** first-ever bidirectional verification, certified count interval/absence, universal zero-shot SOTA, no-overcount, independent generalization from seen FSC TEST, or universal safe false rejection.

## 14. Immediate next research decision

**Choose one primary question:** whether *joint local semantic + whole-instance + identity consistency under uncertainty* can beat rejection-only and strong scalar/global-verifier baselines without unacceptable false removal. Implement validation so the answer might be NO; that is the correct falsifiable design.

If NO: do not force all three methods into one paper. Best fallback is a substantive paper about **prompt reliability and evidence validity with a careful multi-benchmark failure taxonomy**, provided the benchmark/analysis itself contains genuine novelty and reproducibility.

If YES: elevate that single mechanism as the paper contribution; use Thư and Lộc components as baselines, probes, and principled optional elements. Preserve all negative results.
