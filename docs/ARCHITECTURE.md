# Intended architecture (PROPOSED, not implemented end-to-end)

```text
image + query + frozen protocol
         |
        CountGD++ baseline => (count, proposals, scores, source hashes)
         |
   [view/candidate routing based on GT-blind risk and budget]
        /                                    \
  baseline proposals                   recovery proposals
        |                                    |
  same unary evidence contract: semantic, whole-instance, confidence/UNKNOWN
        \                                    /
         \---- original-coordinate matching + Safe-IIC ----/
                                |
               conservative evidence arbitration
                                |
               accepted/rejected/unresolved candidate set
                                |
         observed-candidate interval + backed-off scalar + trace
```

The intended *new* scientific contribution must be a principled, measured
mechanism for resolving contradictory evidence and correlated instances,
not merely concatenating a verifier with a tiling pipeline.

## Slot decisions

- `bidirectional` (Thúy): primary candidate, NOT established cross-dataset.
- `Safe-IIC` (Thúy): identity association, MUST validate false merge/split.
- `Joint V3.2` (Thư): learned count-calibration baseline, cannot be added to an
  integer physical-instance count without a new and independently tested bridge.
- `A3` (Thư): semantic presence baseline; hard zero-veto may overprune.
- `Qwen V7/V7.6` (Lộc): strong image-level semantic baseline and cost comparator;
  do not blindly replace per-instance verifier with global yes/no.
- `V7.5` (Lộc): optional multi-scale recovery; include cost/failure analysis.

## Safety contracts

- `ACCEPTED` only if semantic YES and whole-instance YES.
- Contradictory/ambiguous evidence stays UNKNOWN, unless a prospective
  policy with new validation is registered.
- Rescue matches a baseline? Evidence about existing object, never +1.
- Unmatched rescued candidates must be deduplicated with identity evidence;
  IoU alone is only a geometric cue, not physical-instance proof.
- Never claim [L,U] from detected proposals bounds the true scene count unless
  full recall/coverage assumptions have been established.
- 'Abstain' must withhold scalar when insufficiency is material, not merely
  mark an optional flag while returning a confident exact number.

## Implementation state

`src/countlab` currently provides method registry, schema contracts, hashing,
and a read-only paired evaluator. Runtime adapters from historical GPU code
are NOT yet wired together. This separation is deliberate: making it appear
functional before protocol parity would risk GT leakage and invalid science.
