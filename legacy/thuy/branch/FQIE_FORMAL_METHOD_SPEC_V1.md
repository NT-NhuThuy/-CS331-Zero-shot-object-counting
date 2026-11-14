# FQIE formal method specification v1

> **Current extension:** FQIE is bidirectional: it conservatively deletes disproved baseline proposals and adds verified unmatched rescues. Ambiguous evidence is monotone-safe and cannot reduce the frozen baseline count. See `FQIE_BIDIRECTIONAL_HOLDOUT_RESULTS_20261116.md`.

## Problem

Given an image `I` and open-world text query `q`, return an auditable estimate of the number of whole physical instances matching `q`, without reading ground truth during inference.

The output is not only a scalar. It is:

```text
O(I,q) = (S_verified, S_rejected, S_unknown, [L,U], status, trace)
```

where `S_verified` is a set of accepted physical-instance components, `S_rejected` contains explicitly rejected regions, `S_unknown` contains unresolved regions, `[L,U]` is an auditable count interval, and `status ∈ {exact, partial, abstained}`.

## Evidence state machine

For every candidate `c`, FQIE stores four independent states:

```text
semantic(c)       ∈ {YES, NO, UNKNOWN}
whole_instance(c) ∈ {YES, NO, UNKNOWN}
identity(c,d)     ∈ {SAME, DIFFERENT, UNKNOWN}
decision(c)       ∈ {ACCEPTED, REJECTED, UNKNOWN}
```

Admission invariant:

```text
decision(c) = ACCEPTED
⇒ semantic(c) = YES ∧ whole_instance(c) = YES
```

Unknown is not a negative label and cannot create an identity edge.

## FQIE cascade

```text
Frozen CountGD++ full inference
→ zero/low-evidence trigger
→ multi-view proposal recovery
→ original-coordinate cross-view consensus
→ semantic target verification
→ whole physical-instance verification
→ Safe-IIC identity association
→ verified-instance set + count interval + evidence trace
```

The backbone, checkpoint, prompt, exemplar configuration, tile protocol, and decision policy are frozen per experiment version.

## Safe-IIC invariants

1. Candidates enter identity association only after semantic YES and whole-instance YES.
2. Same-view or overlapping-view candidates cannot merge.
3. UNKNOWN edges cannot merge.
4. A single bridge edge cannot induce transitive union.
5. A component must satisfy pairwise compatibility for all member pairs.
6. Large components are diagnosed explicitly and may be abstained from.
7. Every accepted component retains representative box, source image/view provenance, evidence references, and artifact hashes.

## Count semantics

```text
exact:    L = U = count; no unresolved component remains
partial:  count = L; unresolved candidates/components remain; U is auditable
abstained: scalar count is withheld because evidence is insufficient
```

Baseline-only queries must preserve the baseline scalar for fair paired counting comparisons. Their unverified regions may still be represented as `UNKNOWN` interval evidence, but `UNKNOWN` must never be silently converted into count zero.

## Contribution claim

The method contribution is the evidence-grounded protocol: query-only open-world counting is reformulated as auditable physical-instance verification with explicit uncertainty, identity control, and provenance. This claim is independent of claiming universal MAE superiority.

## Evaluation hierarchy

1. Protocol validity: schema, hash locks, GT-blind execution, no UNKNOWN promotion.
2. Candidate validity: semantic/whole-instance correctness and localization.
3. Identity validity: false merge and false split.
4. Safety: absent false promotion and risk–coverage.
5. Counting: rescue-only and fair full-cohort MAE/RMSE/bias.
6. Generalization: source-disjoint replication and dataset transfer.

## Non-claims

FQIE does not claim complete dense-scene recovery, certified absence, universal counting, no-overcount guarantee, or SOTA without direct matched-protocol evidence.
