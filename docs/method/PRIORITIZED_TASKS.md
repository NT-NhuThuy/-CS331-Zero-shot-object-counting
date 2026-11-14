# First 12 PRs — one shared method, ordered by dependency

Every PR needs an issue, a strict JSON fixture, source-hash provenance, a CPU test, GPU parity where applicable, and a negative/ablation result. No PR changes `FINAL_SEALED`.

| PR | Owner | Concrete branch and deliverable | Depends on | Gate |
|---|---|---|---|---|
| P01 | Thúy | `feat/thuy-v14-import`: frozen V14 raw index/unary task converter + paired parity report | M0 schema | G0 |
| P02 | Lộc | `feat/loc-countgd-adapter`: CountGD++ native full-image boxes + exact prompt/checkpoint/SHA | M0 schema | G0 |
| P03 | Thư | `feat/thu-split-audit`: source-image-disjoint train/dev/cal roster and leakage tests | M0 schema | G0 |
| P04 | Lộc | `feat/loc-region-verifier`: Qwen region crops semantic/whole ternary evidence + source trace | P02 | G1 |
| P05 | Thúy | `feat/thuy-identity-adapter`: pairwise SAME/DIFFERENT/UNKNOWN evidence in original coords | P01/P04 | G1 |
| P06 | Thư | `feat/thu-risk-features`: exported per-proposal risk_hint from locked trained features; no GT in inference | P03/P02 | G1 |
| P07 | Lộc | `feat/loc-multiview-recovery`: tile crops coordinate unprojection + cross-view proposals | P02/P04 | G1 |
| P08 | Shared | `feat/matched-controls`: threshold calibration, native baseline and Qwen-only and Joint V3.2 comparators | P01-P07 | G2 |
| P09 | Thúy | `feat/thuy-evidence-arbitration`: new algorithm proposal, invariants + failure diagnostics | P05/P08 | G2 |
| P10 | Thư | `feat/thu-budget-calibration`: risk–coverage and matched-GPU-cost router ablation | P06/P08 | G2 |
| P11 | Shared | `research/independent-evaluation`: LVIS / PrACo / external supervised compatible tests + CI | P09/P10 | G3 |
| P12 | Shared | `paper/reproducible-release`: claims audit, paper tables, hard-case gallery, model cards | P11 | G4 |

**Stop rule:** If P08 shows simple thresholding or global Qwen veto matches the method, rethink the research thesis before implementing P09. If P09 fails false-merge/false-rejection safety, restrict or remove it. No SOTA claim without P11 matched evaluation.
