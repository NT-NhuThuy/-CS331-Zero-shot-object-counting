"""Single v1 method contract: detector outputs -> routing -> verification -> arbitration.

Runs on frozen candidate/evidence manifests. NOT a GPU implementation.
Module ownership: Thư / routing, Lộc / candidate+verifier adapters,
Thúy / identity and reconciliation. CLI and tests are shared infrastructure.
"""
from __future__ import annotations
import hashlib
import json
import math
from typing import Any
from .schema import Proposal, box_iou, parse_input
from .router import select_for_verification
from .verifier import PrecomputedUnaryVerifier, decision
from .identity import decide_rescue_components

POLICY_DEFAULTS = {
    "baseline_rescue_match_iou": 0.5,
    "identity_min_same_iou": 0.10,
    "identity_overlap_ambiguity_iou": 0.30,
}


def validate_policy(policy: dict | None) -> dict[str, float]:
    if policy is None:
        return dict(POLICY_DEFAULTS)
    if not isinstance(policy, dict) or set(policy) != set(POLICY_DEFAULTS):
        raise ValueError("policy must have EXACTLY the frozen threshold keys")
    for key, value in policy.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not (0 < value <= 1):
            raise ValueError(f"invalid policy threshold: {key}")
    if policy["identity_overlap_ambiguity_iou"] < policy["identity_min_same_iou"]:
        raise ValueError("identity ambiguity IoU must be >= minimum SAME IoU")
    return {key: float(value) for key, value in policy.items()}


def run_method(data: dict[str, Any], *, budget: int | None = None,
               enable_recovery: bool = True, enable_identity: bool = True,
               abstain_on_conflict: bool = False, policy: dict | None = None) -> dict[str, Any]:
    policy = validate_policy(policy)
    baseline, rescues, edges = parse_input(data)
    original_hash = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    candidates = baseline + (rescues if enable_recovery else [])
    selected = select_for_verification(candidates, budget)
    verifier = PrecomputedUnaryVerifier()
    decisions = {p.candidate_id: decision(*verifier.verdict(p, selected=(p.candidate_id in selected)))
                 for p in candidates}
    base_state = {p.candidate_id: decisions[p.candidate_id] for p in baseline}
    pending = []
    rejected_rescue = []
    matched = []
    conflicts = []

    # No rescue is counted if it geometrically matches an existing baseline;
    # a matched YES rescue conflicting with a NO baseline keeps that baseline.
    for p in rescues if enable_recovery else []:
        d = decisions[p.candidate_id]
        matches = sorted(((box_iou(p.box, b.box), b.candidate_id) for b in baseline), reverse=True)
        overlap, base_id = matches[0] if matches else (0., None)
        if overlap >= policy["baseline_rescue_match_iou"]:
            matched.append({"rescue_id": p.candidate_id, "baseline_id": base_id, "iou": round(overlap, 6),
                            "note": "IoU match is duplicate-prevention heuristic, NOT identity proof"})
            if base_state[base_id] == "REJECTED" and d == "ACCEPTED":
                base_state[base_id] = "UNKNOWN"
                conflicts.append(f"contradictory_evidence:{base_id}/{p.candidate_id}")
            continue
        if d == "ACCEPTED":
            pending.append(p)
        elif d == "REJECTED":
            rejected_rescue.append(p)

    groups, identity_unknown, identity_reasons = decide_rescue_components(
        pending, edges, use_identity=enable_identity,
        min_same_iou=policy["identity_min_same_iou"],
        ambiguous_iou=policy["identity_overlap_ambiguity_iou"])
    promoted = [group[0] for group in groups]
    rejected_base = [p for p in baseline if base_state[p.candidate_id] == "REJECTED"]
    accepted_base = [p for p in baseline if base_state[p.candidate_id] == "ACCEPTED"]
    unknown_base = [p for p in baseline if base_state[p.candidate_id] == "UNKNOWN"]
    unknown_rescue = [p for p in rescues if enable_recovery and decisions[p.candidate_id] == "UNKNOWN"]
    unknown_rescue += identity_unknown
    # Matched rescue cannot independently enter accepted/unknown physical count.
    matched_ids = {x["rescue_id"] for x in matched}
    unknown_rescue = [p for p in unknown_rescue if p.candidate_id not in matched_ids]
    point_count = len(baseline) - len(rejected_base) + len(groups)
    lower = len(accepted_base) + len(groups)
    upper = max(point_count, lower) + len(unknown_rescue)
    if not lower <= point_count <= upper:
        raise AssertionError("point count inconsistent with observed-candidate interval")
    reasons = conflicts + identity_reasons
    abstained = bool(abstain_on_conflict and reasons)
    def record(p: Proposal, decision: str) -> dict:
        return p.to_dict(decision)
    out = {
        "schema": "countlab_unified_method_output_v1",
        "image_id": str(data["image_id"]), "query_id": str(data["query_id"]),
        "count": None if abstained else point_count,
        "evaluation_fallback_count": point_count,
        "count_lower_bound": lower, "count_upper_bound": upper,
        "count_interval_scope": "observed_candidates_only_not_scene_ground_truth_bounds",
        "count_status": "abstained" if abstained else "partial",
        "query_abstained": abstained,
        "accepted_instances": [record(x, "ACCEPTED") for x in accepted_base] + [
            {"component_id": "+".join(p.candidate_id for p in group),
             "members": [p.candidate_id for p in group],
             "representative": record(group[0], "ACCEPTED")} for group in groups],
        "rejected_regions": [record(x, "REJECTED") for x in rejected_base + rejected_rescue],
        "unknown_regions": [record(x, "UNKNOWN") for x in unknown_base + unknown_rescue],
        "matched_rescue_to_baseline": matched,
        "evidence_trace": [{"stage": "frozen_baseline", "baseline_count": len(baseline)},
                           {"stage": "risk_router_replay", "budget": budget, "selected": sorted(selected)},
                           {"stage": "unary_verifier_replay", "explicit_rejections": len(rejected_base)},
                           {"stage": "safe_identity", "promoted_components": len(groups), "unresolved": identity_reasons},
                           {"stage": "bidirectional_reconcile", "count": point_count, "conflicts": conflicts}],
        "method_options": {"enable_recovery": enable_recovery,
                           "enable_identity": enable_identity, "abstain_on_conflict": abstain_on_conflict},
        "input_sha256": original_hash,
        "method_policy": policy,
        "method_policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
        "limitations": ["Evidence was supplied by precomputed artifact, not generated online",
                        "YES/NO provenance must be verified in frozen source experiment",
                        "Not an exact scene count or certified absence", "Only current candidate-set coverage is represented"],
    }
    return out
