# UNICOUNT / CountLab — one research pipeline for three contributors

**Status (2026-11-14):** CPU replay and an optional **real-model online runner** are implemented. Online GPU inference was **not executed in this audit** because the necessary upstream CountGD++ repo/checkpoint, BERT, Qwen weights and runtime dependencies are not bundled. `UNICOUNT` is a working internal name, not a scientific novelty or SOTA claim.

> Image + category → frozen CountGD++ proposals → Thư risk features/router → Lộc multi-view recovery and Qwen unary verification → Thúy identity-aware bidirectional reconciliation → count + accepted/rejected/UNKNOWN + evidence trace.

## Start here

**For all three contributors, read [`docs/TEAM_START_HERE.md`](docs/TEAM_START_HERE.md) before creating a feature branch.** This is the latest onboarding and G0–G4 owner checklist.

1. **Verified audit and A* GO/NO-GO:** [`docs/verification/UNIFIED_RESEARCH_AUDIT_20261114.md`](docs/verification/UNIFIED_RESEARCH_AUDIT_20261114.md)
2. **Method spec and conservative semantics:** [`docs/method/METHOD_V1.md`](docs/method/METHOD_V1.md)
3. **Original research trees:** [`legacy/{thu,loc,thuy}/`](legacy/) (original source preserved)
4. **Shared package:** [`src/countlab/`](src/countlab/) — one inference/evidence schema and result API

## What runs now

| Command | Purpose | Evidence level |
|---|---|---|
| `countlab-method` | CPU replay from frozen candidate/evidence manifests | Contract test, not model inference |
| `countlab-import-fqie` | Convert old FQIE task/verdict files to one shared schema | Legacy interface bridge |
| `countlab-batch` | Run one pipeline across frozen query manifests | CPU replay |
| `countlab-online` | Call preserved CountGD++ native model, tile inference, Qwen3-VL region verifier, optional Qwen identity, then unified reconciliation | **Actual GPU code path, unrun on real weights** |
| `countlab-preflight` | Find missing upstream, weights, Python dependencies, unpinned model version | No download, no inference |

### CPU tests and old artifact audits

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests legacy/thuy/branch/test_*.py legacy/loc/tools/praco_test1/test_verify_praco_artifacts.py
```

### Actual online GPU path (no fake parity claim)

The preserved native backend runs **SINGLE-PASS without pseudo-exemplar feedback**. This is a *different experimental operating point* from V14 (`num_pseudo_exemplars=3`) and the Lộc PrACo adapter. Before any paper comparison, add a separate official pseudo-exemplar adapter with source/checkpoint parity.

```bash
# Edit all placeholder paths and pin exact Qwen revision
countlab-preflight --config configs/integrations/native_single_pass_example.json

# Only once preflight and source parity are clear:
countlab-online --image /local/images/demo.jpg --image-id demo --query-id q1 \
  --query 'chair' --config configs/integrations/native_single_pass_example.json \
  --out-dir outputs/unicount_q1
```

This produces create-only `frozen_candidates.json` and `prediction.json`. GT is forbidden in the inference schema. `UNKNOWN` baseline proposals are retained in the point estimate, and output intervals are **observed-candidate intervals**, not certified full-scene count bounds.

### Thư V4/A3 feature adapter

`src/countlab/integrations/thu_features.py` reads the *real* frozen `.npz` node features with explicit `(node_graph_id, raw_query_index)` ↔ candidate mapping. When separately frozen on TRAIN/CAL, a calibration can supply `risk_hint` to the shared router **before** verification. The current 1,301-pair frozen FSC features are exposed research data, **not transferable to arbitrary LVIS queries**; learned Joint V3.2 scalar count correction remains a comparator, not a silent posthoc correction in the instance method.

## Owner map

| Stage | Owner | Primary code |
|---|---|---|
| Candidate risk / budget | Thư | `src/countlab/integrations/thu_features.py`, `method/router.py` |
| CountGD++ native / multi-view / Qwen | Lộc + shared | `integrations/countgd_native.py`, `recovery.py`, `qwen_unary.py` |
| Identity and bidirectional arbitration | Thúy | `method/identity.py`, `method/pipeline.py` |
| Evaluation, GT separation, provenance | Shared | `audit.py`, docs and versioned configs |

**Do not claim that these are three additive gains.** Thư FSC, Lộc PrACo and Thúy LVIS metrics were collected under different input/operating-point protocols.

## Research release gates

- **G0:** real GPU baseline parity, original V14 raw output and freeze proof, stable protocol hashes.
- **G1:** beat threshold-calibrated baseline and global Qwen veto at matched cost; quantify false rejection.
- **G2:** genuinely new uncertainty/identity-constrained arbitration; compare NMS, Union-Find and conservative dual-NO.
- **G3:** independent source-disjoint evaluation plus multi-category and external datasets.
- **G4:** standard benchmark SOTA only if comparisons are truly matched; paper claims limited to observed results.

See the verified audit for detailed metrics, prior art, concrete PR tasks and exit criteria.
