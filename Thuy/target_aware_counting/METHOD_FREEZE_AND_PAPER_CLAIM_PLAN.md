# FQIE paper-claim freeze plan

## Method decomposition

`CountGD++` remains frozen and is never fine-tuned by this work. The proposed inference cascade has three auditable modules:

1. **Proposal recovery (PR):** full-image zero/low-evidence queries receive multi-view tile proposals; cross-view IoU consensus removes single-crop artifacts.
2. **Instance identity control (IIC):** a candidate must pass semantic and whole-instance checks; pairwise same-physical-instance edges are unioned conservatively. `UNKNOWN` never promotes or merges.
3. **Reference consistency (RC):** independent reference crops and multi-reference affinity are used only as evidence-quality gating. RC must be evaluated on the same frozen cohort before it is included in the primary method.

## Freeze rule

V7 and V8 are source-disjoint audits of PR+IIC. Their prompts, tile scales, IoU threshold, and merge policy are frozen. Any change to those components starts a new version and cannot be reported as a continuation of V7/V8.

## Required ablation matrix

On a new validation cohort (not V7/V8):

| variant | PR | IIC | RC | purpose |
|---|---:|---:|---:|---|
| A | no | no | no | frozen CountGD++ baseline |
| B | yes | no | no | proposal-recall contribution |
| C | yes | yes | no | current V7/V8 method |
| D | yes | no | yes | reference-only contribution |
| E | yes | yes | yes | full cascade |

The primary method is E only if E improves MAE/present recall over C on the held-out cohort without increasing absent-query error. Otherwise C remains the primary method and RC is reported as an optional analysis.

## Claim gate

A strong claim requires:

- at least two source-disjoint cohorts (already satisfied for C by V7/V8);
- one held-out cohort for the A–E ablation;
- per-query paired metrics, absent-query false-positive audit, and bootstrap confidence intervals;
- no test-GT use in any selection, prompt, threshold, or merge decision;
- explicit failure analysis for dense scenes and multiple-object crops.

Current defensible claim: **a GT-blind cross-view proposal-recovery and whole-instance identity cascade improves zero/low-evidence CountGD++ predictions on two source-disjoint LVIS audits without worsening absent-query MAE.**

Not yet defensible: universal counting, SOTA, complete recovery of dense scenes, or certified absence.
