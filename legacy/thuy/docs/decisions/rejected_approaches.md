# Rejected and non-promoted approaches

- Raw multi-scale or two-view support: more candidates do not guarantee better counting.
- OWLv2 hard filtering: can remove valuable proposals; do not use as a hard veto.
- Box containment merging: can falsely merge separate physical instances.
- SAM2 mask inclusion rules: do not reliably resolve instance identity.
- Higher resolution: may increase TP while worsening FP, absent-positive rate, or MAE.
- Uncontrolled pseudo-exemplar pooling: protocol and cross-domain effects must remain separate.
