# CS331 CountLab — A*-oriented research unification

**Status (snapshot 2026-11-14):** A non-destructive local research reorganization
of all three contributions. NOT a published, benchmark-verified SOTA system.
The refactor is in Git branch `research/a-star-unification`, not pushed to the
original GitHub repository.

## Read first

1. [`docs/A_STAR_RESEARCH_REPORT.md`](docs/A_STAR_RESEARCH_REPORT.md): full audit,
   owner-by-owner decisions, novelty hazards, architecture, SOTA protocol,
   staged roadmap, release gates.
2. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): data types and proposed method.
3. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): splits, hashes and rules.
4. [`paper/claim_ledger.csv`](paper/claim_ledger.csv): results vs evidence level.

## Repository

```text
src/countlab/             Shared CPU-only contracts, method registry and audits
configs/regimes/          Separate LVIS, FSC experimental, PrACo protocols
tests/                    CPU smoke / contract tests
experiments/              Future frozen experiment registry and protocol templates
paper/                    Claim ledger and paper scaffold
docs/                     Architecture, audit and A* roadmap
legacy/loc/               Lộc V4–V7.6 including official PrACo adapter
legacy/thu/               Thư V4, Joint V3.2 and A3
legacy/thuy/              Thúy FQIE PR/IIC/Safe-IIC and bidirectional evidence
```

**Why legacy?** The historical notebook/script trees are retained intact,
including their exact subdirectory layout, for auditability. They are not
silently merged into a fictitious single working model. New pipelines must
reference a provenance-locked adapter and pass validation first. `legacy/loc`
includes V7.6 additions from `loc/praco-official-adapter-v76-full`.

## Local CPU checks

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/
python -m pytest -q legacy/thuy/branch
python -m pytest -q legacy/loc/tools/praco_test1/test_verify_praco_artifacts.py
countlab-inventory
```

Use `countlab-audit --help` to evaluate NEW matched and frozen CSVs. It never
runs inference or chooses thresholds from test GT.

## Rules

- NEVER train, tune or debug on a sealed test set; V14 must not be reused to
  select a new method variant.
- Never conflate FSC147 positive, mined hard-negative, PrACo mismatch, LVIS
  verified-absent, or free-form query-only evaluation.
- Training status must be explicit: Thu correction heads are trained;
  frozen-backbone is not equivalent to fully training-free.
- No claim of exact physical count or certified absence from schema alone.
- Keep original Git history and author attribution. No destructive cleanup.

For offline environments with pytest already installed, omit pip and run
`PYTHONPATH=src python -m pytest -q tests legacy/thuy/branch` and
`PYTHONPATH=src python -m countlab.audit --help`.
