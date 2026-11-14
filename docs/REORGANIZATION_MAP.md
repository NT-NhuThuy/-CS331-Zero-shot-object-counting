# Path and provenance migration map

| Original path | New path | Status |
|---|---|---|
| `Lộc/*` (main V4–V7.5) | `legacy/loc/*` | exact historical folder layout retained |
| Lộc branch `loc/praco-official-adapter-v76-full` additions | `legacy/loc/notebooks/06_praco_official_adapter/`, `legacy/loc/results/05_praco_official_adapter/`, `legacy/loc/tools/praco_test1/`, `legacy/loc/docs/PRACO_TEST1_A_B_RESULTS.md` | additive V7.6 snapshot |
| `Thu/*` | `legacy/thu/*` | exact historical folder layout retained |
| `Thuy/*` | `legacy/thuy/*` | exact historical folder layout retained |
| none | `src/countlab/*` | new protocol tools; not GPU inference |
| none | `configs/regimes/*` | distinct regime specs |
| none | `docs/A_STAR_RESEARCH_REPORT.md` | research decision and gates |
| none | `paper/claim_ledger.csv` | claim/evidence registry |

**Why no direct import into one `src/method.py`:** legacy methods consume
incompatible model features, pseudo-exemplars, full-image/region inputs, scalar
versus instance outputs and partially unavailable checkpoints. Hiding such
mismatches under shared names would make the paper technically invalid.

The new method registry `src/countlab/registry.py` gives each module a stable
role and points to its original implementation for auditing. Runtime extraction
must follow the validation gates in `A_STAR_RESEARCH_REPORT.md`.
