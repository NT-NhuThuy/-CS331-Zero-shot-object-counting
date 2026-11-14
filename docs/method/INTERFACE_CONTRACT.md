# Shared interface contract v1

**Ground-truth NEVER enters inference manifests.** The parser rejects known GT/label-count keys. Any evaluation reading GT occurs in a separate command using outputs already frozen.

## Inputs

One JSON file per image–query with `schema=countlab_candidate_pool_v1` and:

- `image_id`, `query_id`, `query_text` (required), `baseline_count` equal to the number of `baseline` proposals;
- `baseline` and `rescue`: candidate objects with a globally unique `candidate_id`, `box_xyxy` in **original-image pixels**, optional `confidence∈[0,1]`, optional `risk_hint∈[0,1]`, `view_id`, `evidence_refs` and `evidence={semantic,whole_instance}`;
- `identity_edges`: `{a,b,relation}` where relation is `SAME`,`DIFFERENT`,`UNKNOWN`, and `a,b` are candidate IDs;
- `metadata`: immutable source hashes, checkpoint/prompt/exemplar identifiers, provenance.

Counts with only an aggregated density integral and no one-to-one proposals are OUTSIDE v1; use a protocol-specific baseline fallback rather than manufacturing fake instances. Different visual views MUST map boxes back to original pixels with verified transforms. Metadata is provenance, not a data source for GT/count labels.

## Inference and verification

The v1 router uses `risk_hint` when present, else `1-confidence`, else `0.5`; it is not a trained/calibrated risk model. A budget masks unselected precomputed verdicts as UNKNOWN. It **does not estimate actual GPU compute saved**, only available evidence slots used. `--no-identity` is an unsafe ablation.

A future real `UnaryVerifier` must return separate semantic and whole-instance ternary decisions with prompt/revision/seed/source SHA + per-candidate evidence reference. If Qwen makes only image-level category decisions, store those separately as an auxiliary *screening hint*, not a fabricated region YES/NO. Lộc and Thư adapters may propose features but may NOT directly rewrite `count`.

## Outputs

`schema=countlab_unified_method_output_v1` including `count`, `evaluation_fallback_count`, observed-candidate `[L,U]`, `accepted_instances`, `rejected_regions`, `unknown_regions`, `matched_rescue_to_baseline`, `evidence_trace`, `input_sha256`, `query_abstained`, `count_status`.

`count_status=partial` is the safe default even if observed candidates are all decided, since completeness of the physical scene is unproven. If `--abstain-on-conflict`, `count=null`, `query_abstained=true` and `evaluation_fallback_count` records conservative full-cohort comparison count. For scientific exact/partial/abstained semantics at scene level, new annotation and guarantees are needed.

## Run (CPU replay only)

```bash
python -m pip install -e '.[test]'
countlab-method --input fixtures/unified_toy.json --config configs/method/unicount_v1.json --out /tmp/unified_result.json
countlab-method --input fixtures/unified_toy.json --out /tmp/budgeted.json --budget 2
countlab-import-fqie --index /path/bidirectional_task_index.json --unary /path/identity_unary_decisions.jsonl --out-dir /tmp/fqie_imported
countlab-batch --inputs /tmp/fqie_imported --outputs /tmp/fqie_unified --config configs/method/unicount_v1.json
```

The adapter consumes the actual existing FQIE task index schema but intentionally **does not invent missing pairwise identity edges**, and does not run CountGD++ or Qwen. It can expose conflicts versus the historical V14 implementation: such differences are algorithm changes, not confirmation of V14 results.

Policy defaults and thresholds are versioned in `configs/method/unicount_v1.json`; no tuning these thresholds on V14 or previously exposed FSC TEST. Output records the canonical method-policy SHA256.
