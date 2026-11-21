"""C3/T03+T09: one entrypoint for event-grounded joint instance reasoning.

Only observations delivered as events can affect C3. Unobserved proposals have
UNKNOWN unary/identity values. No detector, router, GT or Qwen calls occur here.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Any

from .schema import parse_input
from .t09_event_grounded_solver import _decode_events, infer_event_grounded_count


DEFAULT_SOLVER_CONFIG = {
    "max_search_states": 20000,
    "max_candidates": 512,
    "max_hints": 32,
    "min_same_iou": 0.05,
}


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_solver_config(config: dict | None) -> dict:
    cfg = dict(DEFAULT_SOLVER_CONFIG) if config is None else dict(config)
    if set(cfg) != set(DEFAULT_SOLVER_CONFIG):
        raise ValueError("solver config must contain exactly the frozen C3 keys")
    for name in ("max_search_states", "max_candidates", "max_hints"):
        if type(cfg[name]) is not int or cfg[name] < (1 if name == "max_candidates" else 0):
            raise ValueError(f"invalid {name}")
    value = cfg["min_same_iou"]
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError("invalid min_same_iou")
    cfg["min_same_iou"] = float(value)
    return cfg


def validate_event_contract(events: list[dict]) -> None:
    """Exact-key event payloads: event metadata/GT stays outside C3 inference."""
    if not isinstance(events, list):
        raise ValueError("observed_events must be a list")
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"event {index}: must be a JSON object")
        op = event.get("op")
        expected = ({"op", "a", "b", "relation", "evidence_ref"}
                    if op == "PAIR_IDENTITY" else
                    {"op", "candidate_id", "verdict", "evidence_ref"}
                    if op in ("REGION_SEMANTIC", "REGION_WHOLE") else None)
        if expected is None or set(event) != expected:
            raise ValueError(f"event {index}: unknown operation or extraneous fields")


def project_observed_evidence(manifest: dict, observed_events: list[dict]) -> tuple[dict, list[str]]:
    """Produce an exact-key manifest with no hidden answers from input.

    If verifier retries disagree, the affected field is conservatively UNKNOWN
    in controls; the raw event-grounded C3 solver receives the conflict and
    abstains. This projection never treats UNKNOWN as NO.
    """
    validate_event_contract(observed_events)
    baseline, rescue, _ = parse_input(manifest)
    candidates = baseline + rescue
    sem, whole, pair, conflicts = _decode_events(observed_events, {p.candidate_id for p in candidates})
    conflicted = set(conflicts)
    refs: dict[str, list[str]] = defaultdict(list)
    for event in observed_events:
        if event["op"] == "PAIR_IDENTITY":
            for cid in (event["a"], event["b"]):
                refs[cid].append(event["evidence_ref"])
        else:
            refs[event["candidate_id"]].append(event["evidence_ref"])

    def candidate(p: Any) -> dict:
        cid = p.candidate_id
        sv = sem.get(cid, ("UNKNOWN", None))[0]
        wv = whole.get(cid, ("UNKNOWN", None))[0]
        if f"verifier_conflict:REGION_SEMANTIC:{cid}" in conflicted:
            sv = "UNKNOWN"
        if f"verifier_conflict:REGION_WHOLE:{cid}" in conflicted:
            wv = "UNKNOWN"
        out = {"candidate_id": cid, "view_id": p.view_id, "box_xyxy": list(p.box),
               "evidence": {"semantic": sv, "whole_instance": wv},
               "evidence_refs": sorted(set(refs[cid]))}
        if p.confidence is not None:
            out["confidence"] = p.confidence
        if p.risk_hint is not None:
            out["risk_hint"] = p.risk_hint
        return out

    masked_edges = []
    for (a, b), (relation, ref) in sorted(pair.items()):
        if f"verifier_conflict:PAIR_IDENTITY:{a},{b}" in conflicted:
            relation = "UNKNOWN"
        masked_edges.append({"a": a, "b": b, "relation": relation,
                             "evidence_ref": ref})
    metadata = manifest.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    output = {"schema": "countlab_candidate_pool_v1",
              "image_id": manifest["image_id"], "query_id": manifest["query_id"],
              "query_text": manifest["query_text"], "baseline_count": len(baseline),
              "baseline": [candidate(p) for p in baseline],
              "rescue": [candidate(p) for p in rescue],
              "identity_edges": masked_edges,
              "metadata": {key: metadata[key] for key in ("image_sha256", "backend_protocol")
                           if key in metadata}}
    parse_input(output)
    return output, conflicts


def solve_joint_v2(manifest: dict, observed_events: list[dict], *,
                   fallback_count: int, config: dict | None = None) -> dict[str, Any]:
    """Run C3 without ever reading hidden precomputed verifier answers.

    A positive count is conditional on the supplied observations and incomplete
    candidate-pool coverage. No zero is issued from empty candidate evidence.
    The fallback must be separately defined by the frozen comparator policy.
    """
    cfg = validate_solver_config(config)
    validate_event_contract(observed_events)
    result = infer_event_grounded_count(manifest, observed_events,
                                        fallback_count=fallback_count, **cfg)
    outcome = {
        "schema": "countlab_joint_solver_v2",
        "image_id": str(manifest["image_id"]),
        "query_id": str(manifest["query_id"]),
        "query_text": str(manifest["query_text"]),
        "status": result["status"],
        "count": result["count"],
        "evaluation_fallback_count": result["evaluation_fallback_count"],
        "original_fallback_count": fallback_count,
        "count_changed": result["decision_changes_fallback"],
        "count_lower_bound": result["observed_count_lower_bound"],
        "count_upper_bound": result["observed_count_upper_bound"],
        "bounds_exact": result["bounds_exact"],
        "count_scope": result["scope"],
        "physical_instances": result["physical_instances"],
        "unresolved_candidate_ids": result["unresolved_candidate_ids"],
        "excluded_candidate_ids": result["excluded_candidate_ids"],
        "blocking_issues": result["blocking_issues"],
        "evidence_requests_to_C1": result["missing_evidence_hints"],
        "source_visible_sha256": result["source_visible_sha256"],
        "observed_events_sha256": result["observed_events_sha256"],
        "solver_config_sha256": _sha(cfg),
        "observed_event_count": result["observed_event_count"],
        "executed_model_calls_by_C3": 0,
        "scene_zero_certified": False,
        "limitations": result["limitations"],
    }
    if outcome["status"] == "PROVISIONAL_POSITIVE":
        if outcome["count"] is None or outcome["count"] <= 0:
            raise AssertionError("positive count must be > 0")
        if len(outcome["physical_instances"]) != outcome["count"]:
            raise AssertionError("positive count lacks instance witnesses")
    else:
        if outcome["status"] != "ABSTAIN" or outcome["count"] is not None:
            raise AssertionError("unsupported solver decision")
        if outcome["evaluation_fallback_count"] != fallback_count:
            raise AssertionError("abstention must preserve frozen fallback")
    return outcome
