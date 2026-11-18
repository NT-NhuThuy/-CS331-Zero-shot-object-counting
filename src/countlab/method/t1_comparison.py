"""Matched CPU replay for T1 controls and experimental joint arbitration."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

from .baselines import ARM_IDS, BaselinePolicy, run_baseline_arm
from .joint_arbitration import run_joint_arbitration
from .schema import parse_input


def run_matched_t1_comparison(data: dict[str, Any], *, budget: int | None = None,
                              pairwise_budget: int | None = None,
                              policy: dict[str, Any] | None = None,
                              arms: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Run requested controls plus joint experimental arm on one frozen input.

    This is a replay harness, not an evaluator: it cannot choose a winner
    without GT-separated annotations. Its purpose is to make pool/evidence
    equality inspectable before real DEV becomes available.
    """
    parse_input(data)
    frozen_policy = BaselinePolicy.parse(policy)
    selected_arms = tuple(sorted(ARM_IDS)) if arms is None else tuple(arms)
    if not selected_arms or any(arm not in ARM_IDS for arm in selected_arms):
        raise ValueError("arms must be non-empty registered baseline IDs")
    outputs = {arm: run_baseline_arm(data, arm, budget=budget, policy=frozen_policy.to_dict())
               for arm in selected_arms}
    joint = run_joint_arbitration(data, budget=budget, pairwise_budget=pairwise_budget,
                                  pairwise_strategy="impact", collision_iou=frozen_policy.nms_iou,
                                  identity_min_same_iou=frozen_policy.identity_min_same_iou)
    geometry_control = run_joint_arbitration(data, budget=budget, pairwise_budget=pairwise_budget,
                                             pairwise_strategy="geometry", collision_iou=frozen_policy.nms_iou,
                                             identity_min_same_iou=frozen_policy.identity_min_same_iou)
    risk_control = run_joint_arbitration(data, budget=budget, pairwise_budget=pairwise_budget,
                                         pairwise_strategy="risk", collision_iou=frozen_policy.nms_iou,
                                         identity_min_same_iou=frozen_policy.identity_min_same_iou)
    input_hashes = ({output["input_sha256"] for output in outputs.values()} |
                    {joint["input_sha256"], geometry_control["input_sha256"], risk_control["input_sha256"]})
    if len(input_hashes) != 1:
        raise AssertionError("matched comparison lost frozen input identity")
    return {
        "schema": "countlab_t1_matched_comparison_v1",
        "input_sha256": input_hashes.pop(),
        "budget": budget,
        "pairwise_budget": pairwise_budget,
        "policy": frozen_policy.to_dict(),
        "arms": outputs,
        "experimental_joint": joint,
        "pairwise_geometry_control": geometry_control,
        "pairwise_risk_control": risk_control,
        "limitations": ["No GT metric or winner selection is computed", "CPU evidence replay is not billed verifier cost",
                        "Threshold policy must be calibrated only on an approved CAL split"],
    }


def run_matched_t1_batch(manifests: list[dict[str, Any]], *, budget: int | None = None,
                         pairwise_budget: int | None = None,
                         policy: dict[str, Any] | None = None,
                         arms: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Replay T1 controls over a fixed, GT-blind batch and summarize changes.

    This is deliberately a *decision* summary, not an accuracy evaluator. It
    exposes how often the joint rule differs from incumbent B6 before offline
    instance labels are joined. Every row is tied to its immutable input hash,
    so a later evaluator can verify that no arm changed candidate pools.
    """
    if not manifests:
        raise ValueError("matched T1 batch must contain at least one manifest")
    seen_keys: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    actions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for manifest in manifests:
        baseline, rescues, _ = parse_input(manifest)
        key = (str(manifest["image_id"]), str(manifest["query_id"]))
        if key in seen_keys:
            raise ValueError("matched T1 batch contains duplicate image/query")
        seen_keys.add(key)
        comparison = run_matched_t1_comparison(
            manifest, budget=budget, pairwise_budget=pairwise_budget,
            policy=policy, arms=arms)
        incumbent = comparison["arms"].get("B6")
        if incumbent is None:
            # B6 is the named incumbent. If callers restrict replay arms, run
            # it solely for diagnostics; it does not alter the selected arms.
            incumbent = run_baseline_arm(manifest, "B6", budget=budget,
                                         policy=comparison["policy"])
        joint = comparison["experimental_joint"]
        joint_count = joint["evaluation_fallback_count"]
        incumbent_count = incumbent["count"]
        if incumbent_count is None:
            raise AssertionError("incumbent B6 must expose fallback count")
        joint_actions = [item["action"] for item in joint["joint_trace"]]
        joint_reasons = [item["reason"] for item in joint["joint_trace"]]
        actions.update(joint_actions)
        reasons.update(joint_reasons)
        rows.append({
            "image_id": key[0], "query_id": key[1],
            "input_sha256": comparison["input_sha256"],
            "n_candidates": len(baseline) + len(rescues),
            "incumbent_b6_fallback_count": incumbent_count,
            "joint_fallback_count": joint_count,
            "fallback_count_delta_joint_minus_b6": joint_count - incumbent_count,
            "joint_unresolved_count": len(joint["unresolved_candidates"]),
            "joint_query_abstained": joint["query_abstained"],
            "pairwise_planned": joint["pairwise_allocation_summary"]["n_planned_pairs"],
            "joint_actions": joint_actions,
        })
    rows.sort(key=lambda row: (row["image_id"], row["query_id"]))
    deltas = [row["fallback_count_delta_joint_minus_b6"] for row in rows]
    digest = hashlib.sha256(
        json.dumps([row["input_sha256"] for row in rows], separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "countlab_t1_matched_batch_summary_v1",
        "status": "FROZEN_EVIDENCE_DECISION_SUMMARY_NO_CLAIM",
        "batch_input_sha256": digest,
        "budget": budget, "pairwise_budget": pairwise_budget,
        "n_image_query_pairs": len(rows),
        "summary": {
            "incumbent_b6_fallback_count_sum": sum(row["incumbent_b6_fallback_count"] for row in rows),
            "joint_fallback_count_sum": sum(row["joint_fallback_count"] for row in rows),
            "joint_minus_b6_count_sum": sum(deltas),
            "n_changed_pairs": sum(delta != 0 for delta in deltas),
            "n_count_increases": sum(delta > 0 for delta in deltas),
            "n_count_decreases": sum(delta < 0 for delta in deltas),
            "n_pairs_with_unresolved": sum(row["joint_unresolved_count"] > 0 for row in rows),
            "n_abstained": sum(row["joint_query_abstained"] for row in rows),
            "n_pairwise_queries_planned": sum(row["pairwise_planned"] for row in rows),
            "joint_action_counts": dict(sorted(actions.items())),
            "joint_reason_counts": dict(sorted(reasons.items())),
        },
        "rows": rows,
        "limitations": [
            "No ground truth is read or used for this summary",
            "A count change is not evidence of improvement",
            "CPU replay is not measured verifier cost",
        ],
    }
