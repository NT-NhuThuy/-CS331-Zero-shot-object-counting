# C3 T09 set-level non-match experimental contract v0

**Scope**: An independent experimental *evidence-contract audit*, not a new count estimator. The module does **not** change A4/A4+ operational count, does **not** call a model, and does **not** use ground truth.

## Research hypothesis

Instead of B individual pairwise DIFFERENT verdicts between a rescue candidate r and B baseline proposals, test a new verifier observation `SET_IDENTITY(r, subset)` asking whether any member is the same physical target instance as r. Under the assumption that each accepted `NONE_MATCH` observation is correct, the union of gated `NONE_MATCH` subsets covering all baseline candidates supports conditional exclusion of existing baseline identities. A hard group-size cap 8 implies at least ceil(B/8) batch observations for this *particular protocol*, not necessarily a reduction in wall-clock latency, error, or tokens.

**Warning**: There is no claim that a vision-language model can answer this primitive reliably. A T06 gate can attest that a response occurred and binds to a prompt and source; it cannot prove visual correctness. The implementation intentionally does not emit an operational or forced count. There is no claim of A* novelty without an extensive literature search and real paired evaluation.

## Integration requirements

1. Keep frozen A4 untouched. Store this as a separate source module.
2. Extend C2/T06 only if the team agrees to an additional `SET_IDENTITY` observation schema. The model instruction must define `NONE_MATCH`, `SOME_MATCH`, and `UNKNOWN` precisely. Include the complete set of IDs/boxes in the actual prompt and preserve raw response, image SHA, model revision, token/latency logs.
3. Verify raw response and provenance in T06; pass **only externally verified evidence refs** as `gated_evidence_refs`. Passing refs derived from the event list itself would bypass the intended gate.
4. Bind the query/image/candidate universe using `universe_sha256(manifest, source_image_sha256)`. An independent source-image provenance check still must happen outside this module.
5. Compare matched image-query cohorts at equal call, token, latency and candidate budgets. Partition by **source image**, not by four repeated query records per image. Do not use exposed 8-image E0/E1 cohort for final generalization.
6. Do not count multiple rescues independently unless rescue-rescue identity has separately been resolved. This prototype does not authorize any joint count correction.

## API

`audit_set_nonmatch(manifest, events, set_observations, source_image_sha256=..., gated_evidence_refs=set(...), max_group_size=8)`

`events`: existing C3 `REGION_SEMANTIC`, `REGION_WHOLE`, `PAIR_IDENTITY` observed events.

`set_observations`: additional externally observed objects, e.g.:

```json
{
  "op": "SET_IDENTITY",
  "rescue_candidate_id": "r1",
  "baseline_candidate_ids": ["b1", "b2"],
  "verdict": "NONE_MATCH",
  "evidence_ref": "qwen_set:evidence_ref",
  "universe_sha256": "<computed SHA256>",
  "source_image_sha256": "<actual source SHA256>",
  "query_id": "fsc147:query:..."
}
```

`conditional_set_separation_supported` means only that the **observed, gated events** cover the baseline and meet the contract, conditional on their truth. The function always returns `forced_count: null` and `operational_count_changed: false`.

## Developer smoke test

From the repository root, with the project's Python environment:

`python -m pytest -q tests/test_thuy_t09_set_nonmatch_contract.py`

Run the full suite separately. This v0 contains only unit-level contract tests, **not** real Qwen evidence or method-efficacy tests.
