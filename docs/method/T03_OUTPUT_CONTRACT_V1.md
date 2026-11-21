# T03 — GT-blind evidence graph and output contract (local delivery, v1)

**Owner:** Thúy. **Scope:** G0 input/output software contract. **No GPU, no GT, no inference policy changes.** This add-on intentionally **does not replace or silently modify** `src/countlab/method/schema.py`, `pipeline.py`, `joint_arbitration.py`, the frozen C3 A4 algorithm, or team-owned adapters.

## Authoritative interfaces

- **Input** `countlab_candidate_pool_v1`: one image-query pair per JSON object; `image_id`, `query_id`, `query_text`, `baseline_count`, `baseline`, `rescue`, `identity_edges`, optional `metadata`. `baseline_count` must be exactly the number of baseline proposal records, **not a density integral** or known GT count. `image_id/query_id/query_text` are nonblank strings.
- **Nodes** are proposals in `baseline` or `rescue`; each has globally unique `candidate_id`, original-image `box_xyxy` with positive area, optional `view_id`, optional `confidence`/`risk_hint` in [0,1], optional `evidence_refs`, optional `provenance`, and unary `evidence` with `semantic` and `whole_instance` ∈ {`YES`,`NO`,`UNKNOWN`}. Missing unary fields default to `UNKNOWN` in the existing parser. A missing field must **never** become NO.
- **Edges** connect distinct known candidate IDs, with `relation` ∈ {`SAME`,`DIFFERENT`,`UNKNOWN`}. The same unordered pair cannot be repeated, even with the same relation. `UNKNOWN` never establishes `DIFFERENT`; geometry is not a label of physical identity.
- **Provenance** belongs under top-level `metadata` and optional node/edge `provenance`; keep original-image transforms, detector model, prompt, score threshold, hash, and evidence-call references. Other input keys are rejected at the strict T03 boundary until versioned and approved. Annotated target count, boxes, labels, absence GT, or test annotations are forbidden anywhere in inference payload, including nested metadata. This is a schema guard, not a substitute for a secure data boundary.

## State and decision meaning

- Unary acceptance requires `semantic=YES` **and** `whole_instance=YES`. Unary rejection requires dual `NO`. Other combinations are **UNKNOWN**, not negative. These are **reported/verifier evidence**, not verified physical truth.
- Identity `SAME` allows coalescence only under solver safeguards; `DIFFERENT` is needed for certified separation when the solver's policy requires it. Contradictory or absent evidence must produce an explicit unresolved/conflict trace and conservative fallback. **T03 does not resolve conflicts; T09 owns the policy.**
- `ZERO` means an **observed-candidate fallback count of 0**, not certified scene absence. `K` means fallback count is a positive integer. Both use `count_status=partial` and `query_abstained=false` in current output versions.
- `ABSTAIN` means `count=null`, `query_abstained=true`, `count_status=abstained`, with `evaluation_fallback_count` a nonnegative integer retained for fair *full-roster* evaluation. The fallback is **not an accepted physical-instance claim**.
- `count_lower_bound`, `count_upper_bound` and `count_interval_scope=observed_candidates_only_not_scene_ground_truth_bounds` are candidate-scoped diagnostics, not whole-scene recall bounds. No GT read is involved in generating them.
- Current valid output schemas: `countlab_unified_method_output_v1` and `countlab_joint_arbitration_experimental_v1`. In both, `count` is null iff abstained, otherwise equals fallback; 0 ≤ lower ≤ fallback ≤ upper. Experimental joint traces must not be advertised as deployed B7 results.

## Acceptance paths and verification

| Deliverable | File |
|---|---|
| Evidence graph schema | `docs/method/evidence_graph_schema_v1.json` |
| Output contract | `docs/method/T03_OUTPUT_CONTRACT_V1.md` |
| Strict validation module | `src/countlab/method/t03_contract_v1.py` |
| Runnable reproducible fixtures | `docs/method/T03_example_input_output.json` |
| Tests | `tests/test_thuy_t03_contract_v1.py` |
| Audit CLI | `tools/thuy/t03_validate_v1.py` |

The strict validator **supplements** legacy `parse_input`; it does not change the default parser entry points. Team owners should explicitly call `validate_candidate_pool` at producer/consumer boundaries after agreeing to the stricter closed-key contract. Until integration with the official T04/T05/T07 payloads and PR/CI is confirmed, task status is **LOCAL PATCH TESTED / INTEGRATION HOLD**, not team G0 GO.

## Reproducibility (Windows local)

Run from the Windows repository checkout root, after unpacking this ZIP to the same relative paths:

```powershell
$env:PYTHONPATH = "src"
py -m pytest -q tests/test_thuy_t03_contract_v1.py
py tools/thuy/t03_validate_v1.py --examples docs/method/T03_example_input_output.json
```

No Git command, SSH command, model download, or server mutation is needed to validate this T03 handoff. Git/PR work must happen **only on Windows**.

## Limitations

1. This contract rejects known GT-bearing keys and unexpected fields; it cannot prove data provenance or prevent an adversary from hiding a label inside an unrelated string value.
2. A validator passing does not certify that Qwen evidence is correct, that a box is whole-instance, or that all objects in a scene were proposed.
3. The current main pipeline has permissive `parse_input`; moving to strict boundary validation requires an explicit review of all upstream input producer schemas and a separate CI integration decision.
4. Real T06 pilot remains dependent on verified T04/T05/T07 handoffs, and research claims remain out of scope.
