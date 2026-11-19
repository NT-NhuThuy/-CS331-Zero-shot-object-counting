# UNICOUNT v2 — owner allocation and dependencies

All work branches from `research/unified-integration-verified` into reviewed PRs. One active method; no more full alternative pipelines under `legacy/`.

| Priority | Owner | Task / PR theme | Concrete output | Dependency and acceptance |
|---|---|---|---|---|
| **P0 — NOW** | **Thư** | Dataset registry + source protocol, **DSET-01** | `datasets/` registry, query-pair schema, FSC extended 1301 vs Lộc 111 equality, exposed ledger, label-review sampling plan | Existing P03 docs; no raw images/GT in inference. Verify 111 same cohort and note fresh126⊂fresh160. |
| **P0 — NOW** | **Lộc** | Freeze main CountGD++ baseline, **P02-PROTO** | Written single official pseudo3 operating point, full-box candidate ID schema and source/weights/prompt/tiles hashes, GPU raw evidence exports on a small exposed diagnostic pilot | Confirm uniform text-only query regime and native/no-pseudo is comparator only. Include stable original pixel box and per-query baseline count. |
| **P0 — NOW** | **Thúy** | Formal method graph, **GRAPH-SPEC** | Graph node/edge semantics for baseline+rescues, `zero/positive/abstain`, constraint table, pseudocode, explicit baseline identity policy | Requires agreed input schema; document what no candidate/unknown relation means; NO GT from inference. |
| **P1 — FIRST INTEGRATION** | **Lộc** | Actual evidence generator, **P04+P07** | Frozen recovery pool + Qwen region semantic/whole & pair identity evidence with prompt, latency, model revision; one cohort | Must be from same P02 protocol; handle failures as UNKNOWN; avoid global veto as unary evidence. |
| **P1 — FIRST INTEGRATION** | **Thư** | P06 runtime risk reconstruction, **P06-REAL** | Exact-key V4/A3 for *the same Lộc candidate IDs*; feature/protocol hashes; explicit query-level presence/recovery signal when baseline=0 and candidate-level count-impact signal otherwise | Requires P02 candidate map and frozen node semantics; no learned calibration while G0 HOLD. Cross-view features only if valid nodes exist. |
| **P1 — FIRST INTEGRATION** | **Thúy** | Graph reconciliation prototype, **P09-PROTOTYPE** | Algorithm consumes frozen unary+identity+candidate features, yields count and trace; baseline AND rescue dedup, unresolved = UNKNOWN, explicit policy for count=0 | Requires Lộc evidence; compare against v1 dual-NO/complete-link and simple NMS. |
| **P2 — CLEAN DATA** | **Thư (lead), all review** | New source-independent splits, **DSET-02** | Approval-ready train/dev/cal/final metadata; label annotation agreement, image-family audit, positive/absent/distractor balance | Existing exposed 1301 never final; no training/threshold selection before approval; 3-person sign-off. |
| **P2 — ACTIVE LOOP** | **Thư + Thúy**, Lộc integration | Count-impact active action control, **P10+P09-JOINT** | Action utility/graph state update, budget scheduler and stop/abstain; Qwen/recovery callbacks plugged into one algorithm | Requires P1 full candidate/evidence trace and approved DEV/CAL for calibration. Compare against random/confidence/fixed scheduling at matched cost. |
| **P3 — SCIENTIFIC EVAL** | **All** | Comparators + ablations and cost, **P08+P11** | Absent zero accuracy, positive false-zero, present MAE, graph identity and compute; paired CIs; masked final eval | Independent protocol and locked comparator configs; report negatives, not just successful examples. |

## Ownership boundaries

**Thư** owns scientific component C1, presence/count-risk computation, query-level decision utility and dataset/split coordination. She does **not** directly overwrite final `count` using Joint V3.2 scalar correction in the unified graph.

**Lộc** owns scientific component C2, frozen candidate sourcing, tiling and region evidence generation. He does **not** decide the final count via an independent global Qwen image-veto path.

**Thúy** owns scientific component C3 and end-to-end graph arbitration and output. She does **not** infer object identity from IoU alone or reinterpret UNKNOWN as SAME.

All three jointly review schema changes, split approvals, acceptance targets and claim ledger. The primary result is one merged UNICOUNT output, not 3 averaged counts.

## Immediate deliverable order

- **Meeting 1:** approve `METHOD_V2`, `PROTOCOL`, P02 candidate fields, owner boundaries and status 0/PRESENT/ABSTAIN; P0 PRs run in parallel.
- **Meeting 2:** choose an exposed diagnostic cohort (suggest 32 image-query pairs stratified absent/present), freeze source list before inspecting outcomes, then connect P1 data flow on real GPU evidence. Not a paper endpoint.
- **Meeting 3:** first failure taxonomy, assess whether joint active reasoning is needed vs simple Qwen + NMS; only then implement learned action utility under approved splits.

## Concrete acceptance per owner

- **Thư DONE P1:** exact ID coverage on *same* pool, no silent missing/duplicate joins, output feature hashes, presence risk diagnostics including no-baseline-candidate queries; no calibration claim.
- **Lộc DONE P1:** stable official-protocol candidate pool and original-pixel coordinates, GT-free source provenance, Qwen region/pair evidence and failure/latency stats on same cohort.
- **Thúy DONE P1:** one algorithm consumes all outputs and produces reproducible zero/positive/abstain plus trace; graph never double counts a known SAME component; audit contradictory evidence and unresolved associations.
