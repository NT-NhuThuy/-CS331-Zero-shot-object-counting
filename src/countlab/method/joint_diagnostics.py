"""Decision-quality diagnostics for experimental joint arbitration.

No ground truth enters this module. It reports differences from v1 replay so
that real evaluation can later determine whether a difference was beneficial.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from .joint_arbitration import run_joint_arbitration
from .pipeline import run_method
from .schema import parse_input


def diagnose_joint_decision(data: dict[str, Any], *, budget: int | None = None,
                            baseline_policy: dict[str, Any] | None = None,
                            collision_iou: float = .5,
                            identity_min_same_iou: float = .10,
                            abstain_on_unresolved: bool = False) -> dict[str, Any]:
    """Compare v1 and joint outputs on exactly one frozen evidence manifest."""
    parse_input(data)
    v1_policy = baseline_policy or {
        "baseline_rescue_match_iou": collision_iou,
        "identity_min_same_iou": identity_min_same_iou,
        "identity_overlap_ambiguity_iou": collision_iou,
    }
    v1 = run_method(data, budget=budget, policy=v1_policy)
    joint = run_joint_arbitration(data, budget=budget, collision_iou=collision_iou,
                                  identity_min_same_iou=identity_min_same_iou,
                                  abstain_on_unresolved=abstain_on_unresolved)
    if v1["input_sha256"] != joint["input_sha256"]:
        raise AssertionError("diagnostic inputs diverged")
    actions = Counter(row["action"] for row in joint["joint_trace"])
    reasons = Counter(row["reason"] for row in joint["joint_trace"])
    v1_matched = {row["rescue_id"] for row in v1["matched_rescue_to_baseline"]}
    joint_added = {row["rescue_component"][0] for row in joint["joint_trace"]
                   if row["action"] == "ADD" and "rescue_component" in row}
    return {
        "schema": "countlab_joint_decision_diagnostic_v1",
        "input_sha256": v1["input_sha256"],
        "budget": budget,
        "v1_evaluation_fallback_count": v1["evaluation_fallback_count"],
        "joint_evaluation_fallback_count": joint["evaluation_fallback_count"],
        "fallback_count_delta_joint_minus_v1": (
            joint["evaluation_fallback_count"] - v1["evaluation_fallback_count"]),
        "v1_matched_rescue_ids": sorted(v1_matched),
        "joint_added_component_ids": sorted(joint_added),
        "joint_action_counts": dict(sorted(actions.items())),
        "joint_reason_counts": dict(sorted(reasons.items())),
        "joint_unresolved_count": len(joint["unresolved_candidates"]),
        "joint_query_abstained": joint["query_abstained"],
        "interpretation": ("A count delta or changed action is not an improvement claim; "
                           "evaluate only with fresh GT-separated annotations at matched evidence budget."),
    }
