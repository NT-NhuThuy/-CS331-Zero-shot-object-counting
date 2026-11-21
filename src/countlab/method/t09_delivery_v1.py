"""T09 delivery boundary: strict, event-grounded evidence/identity trace.

This additive module does not modify the frozen joint solver or authorize a
CountGD++/Qwen call, a scene-zero claim, or an operational count correction.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .joint_solver_v2 import solve_joint_v2, validate_event_contract, validate_solver_config
from .t03_contract_v1 import validate_candidate_pool


def _verdict(values: set[str]) -> str:
    decisive = values - {"UNKNOWN"}
    if len(decisive) > 1:
        return "CONFLICT"
    if decisive:
        return next(iter(decisive))
    return "UNKNOWN"


def run_t09_delivery(manifest: dict[str, Any], observed_events: list[dict[str, Any]], *,
                     fallback_count: int, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate T03 before T09 and expose candidate-level causal evidence refs.

    Positive K is conditional on the observed pool and veracity of verifier
    events. An empty, negative-only or ambiguous pool must not certify ZERO.
    """
    source_audit = validate_candidate_pool(manifest)
    validate_event_contract(observed_events)
    cfg = validate_solver_config(config)
    strict = solve_joint_v2(manifest, observed_events,
                            fallback_count=fallback_count, config=cfg)
    baseline_ids = {p["candidate_id"] for p in manifest["baseline"]}
    rescue_ids = {p["candidate_id"] for p in manifest["rescue"]}
    ids = baseline_ids | rescue_ids
    unary: dict[str, dict[str, set[str]]] = {
        cid: {"semantic": set(), "whole_instance": set()} for cid in ids
    }
    refs: dict[str, set[str]] = defaultdict(set)
    identity: dict[str, list[dict[str, str]]] = defaultdict(list)
    all_refs: set[str] = set()
    for event in observed_events:
        ref = event["evidence_ref"]
        if ref in all_refs:
            raise ValueError("duplicate observed evidence_ref")
        all_refs.add(ref)
        if event["op"] == "PAIR_IDENTITY":
            a, b = event["a"], event["b"]
            if a == b or a not in ids or b not in ids or event["relation"] not in {"SAME", "DIFFERENT", "UNKNOWN"}:
                raise ValueError("invalid observed pair")
            for cid, other in ((a, b), (b, a)):
                refs[cid].add(ref)
                identity[cid].append({"peer_candidate_id": other,
                                      "relation": event["relation"], "evidence_ref": ref})
        else:
            cid = event["candidate_id"]
            field = "semantic" if event["op"] == "REGION_SEMANTIC" else "whole_instance"
            if cid not in ids or event["verdict"] not in {"YES", "NO", "UNKNOWN"}:
                raise ValueError("invalid observed unary")
            unary[cid][field].add(event["verdict"])
            refs[cid].add(ref)
    accepted: dict[str, str] = {}
    for instance in strict["physical_instances"]:
        for ref in (instance["unary_evidence_refs"] + instance["same_identity_evidence_refs"] +
                    instance["different_identity_evidence_refs"]):
            if ref not in all_refs:
                raise AssertionError("physical-instance witness references unobserved evidence")
        for cid in instance["member_candidate_ids"]:
            if cid not in ids or cid in accepted:
                raise AssertionError("invalid or duplicated physical-instance member")
            accepted[cid] = instance["instance_id"]
    if strict["status"] == "PROVISIONAL_POSITIVE":
        if type(strict["count"]) is not int or strict["count"] <= 0:
            raise AssertionError("nonpositive conditional K")
        if len(strict["physical_instances"]) != strict["count"]:
            raise AssertionError("conditional K lacks physical-instance witnesses")
        decision = "K"
        reason = "ALL_INCLUDED_INSTANCES_HAVE_OBSERVED_WITNESSES"
    elif strict["status"] == "ABSTAIN":
        if strict["count"] is not None or strict["evaluation_fallback_count"] != fallback_count:
            raise AssertionError("unsafe abstain/fallback mismatch")
        decision = "ABSTAIN"
        if not ids:
            reason = "EMPTY_POOL_CANNOT_PROVE_SCENE_ABSENCE"
        elif strict["blocking_issues"]:
            reason = "OBSERVED_IDENTITY_UNARY_OR_RESOURCE_BLOCKER"
        else:
            reason = "INSUFFICIENT_EVIDENCE_OR_NONUNIQUE_OBSERVED_COUNT"
    else:
        raise AssertionError("unknown strict decision status")

    unresolved = set(strict["unresolved_candidate_ids"])
    excluded = set(strict["excluded_candidate_ids"])
    audit = []
    for cid in sorted(ids):
        audit.append({
            "candidate_id": cid,
            "role": "baseline" if cid in baseline_ids else "rescue",
            "view_id": next(p.get("view_id", "full") for p in
                            (manifest["baseline"] + manifest["rescue"]) if p["candidate_id"] == cid),
            "observed_semantic": _verdict(unary[cid]["semantic"]),
            "observed_whole_instance": _verdict(unary[cid]["whole_instance"]),
            "observed_identity_edges": sorted(identity[cid], key=lambda x: (x["peer_candidate_id"], x["evidence_ref"])),
            "observed_evidence_refs": sorted(refs[cid]),
            "strict_instance_id": accepted.get(cid),
            "strict_excluded": cid in excluded,
            "strict_unresolved": cid in unresolved,
        })
    if strict["original_fallback_count"] != fallback_count:
        raise AssertionError("frozen fallback was mutated")
    if strict["scene_zero_certified"] is not False:
        raise AssertionError("T09 may not certify scene ZERO")
    if decision == "K":
        observed_pool_decision, observed_pool_count = "K", strict["count"]
    elif ids and not strict["blocking_issues"] and all(
            x["observed_semantic"] == "NO" and
            x["observed_whole_instance"] == "NO" for x in audit):
        observed_pool_decision, observed_pool_count = "ZERO", 0
    else:
        observed_pool_decision, observed_pool_count = "ABSTAIN", None
    return {
        "schema": "countlab_t09_handoff_v1",
        "image_id": source_audit["image_id"],
        "query_id": source_audit["query_id"],
        "decision": decision,
        "decision_reason": reason,
        "observed_pool_decision": observed_pool_decision,
        "observed_pool_count": observed_pool_count,
        "observed_pool_zero_not_scene_certified": True,
        "count": strict["count"],
        "evaluation_fallback_count": strict["evaluation_fallback_count"],
        "scene_zero_certified": False,
        "count_scope": "OBSERVED_POOL_CONDITIONAL_NOT_SCENE_CERTIFIED",
        "strict_result": strict,
        "candidate_trace": audit,
        "blocking_issues": strict["blocking_issues"],
        "source_visible_sha256": strict["source_visible_sha256"],
        "observed_events_sha256": strict["observed_events_sha256"],
        "solver_config_sha256": strict["solver_config_sha256"],
        "executed_model_calls_by_T09": 0,
        "gt_used": False,
        "operational_count_correction_authorized": False,
    }
