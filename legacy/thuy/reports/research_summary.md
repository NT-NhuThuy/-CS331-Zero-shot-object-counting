# ZOC research summary

## Scope

This summary covers E00–E10 as selected for the public archive. Source reports for the main claims are included under `reports/source_reports/`; no GPU inference was rerun during archive construction. See `reports/source_audit.md` for the remaining provenance limitations.

## Core evidence

| ID | Experiment | Reported result | Cohort/protocol constraint | Status |
|---|---|---|---|---|
| E00 | Frozen CountGD++ baseline | MAE 19.83 on positive primary100; absent diagnostic 0/111 exact-zero | Positive and absent groups are separate | reported_pending_audit |
| E01 | Pseudo-exemplars | FSC-147 improved in some runs, held-out LVIS worsened | Multiple protocols; do not pool runs | reported_pending_audit |
| E02 | Candidate recall audit | Raw top-900 recall 85.11%; thresholded recall 81.51% | Audit-specific cohort and denominator | reported_pending_audit |
| E03 | A4 crop + NMS fallback | MAE 17.07 → 14.95; TP +620; FP +21 | VAL300 source-disjoint; not directly comparable to E00 | reported_pending_audit |
| E04 | B0 SAM3 add-only recovery | MAE 19.83 → 18.47; TP +151; FP +27 | Same primary100 comparison | reported_pending_audit |

## Main interpretation

Candidate recovery can rescue missed instances, but additional views, proposals, or masks do not guarantee query-correct counting. Semantic identity, duplicate/merge errors, and verified-absent safety remain unresolved.
