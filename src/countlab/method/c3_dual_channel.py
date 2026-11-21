"""C3 dual-channel inference: operational evidence correction and strict decision.

Research integration arm, not full V14 parity or scene-count certification.
"""

from countlab.method.joint_solver_v2 import solve_joint_v2
from countlab.method.v14_rejection_shadow import run_v14_rejection_shadow
from countlab.method.c3_rescue_arbitration import reconcile_rescues
from countlab.method.c3_evidence_refinement import project_decisive_events

_UNARY = {
    "REGION_SEMANTIC": "semantic",
    "REGION_WHOLE": "whole_instance",
}


def _state(values):
    if not values:
        return "MISSING"
    distinct = set(values)
    if len(distinct) > 1:
        return "CONFLICT"
    return next(iter(distinct))


def solve_c3_dual_channel(manifest, observed_events, *, fallback_count, config=None, rescue_policy="rejection_only"):
    if rescue_policy not in ("rejection_only", "guarded_identity"):
        raise ValueError("unsupported rescue_policy")
    strict = solve_joint_v2(
        manifest,
        observed_events,
        fallback_count=fallback_count,
        config=config,
    )

    decision_events, ignored_unknown_refs = (
        project_decisive_events(observed_events)
        if rescue_policy == "guarded_identity"
        else (observed_events, [])
    )

    shadow = run_v14_rejection_shadow(
        manifest,
        decision_events,
        fallback_count=fallback_count,
    )

    if strict["original_fallback_count"] != fallback_count:
        raise AssertionError("strict solver changed original fallback")

    if strict["status"] == "ABSTAIN":
        if strict["evaluation_fallback_count"] != fallback_count:
            raise AssertionError("abstention changed frozen fallback")

    rejected = shadow["rejected"]
    operational_count = shadow["operational_count"]
    identity_arbitration = None
    if rescue_policy == "guarded_identity":
        identity_arbitration = reconcile_rescues(manifest, observed_events, shadow)
        effective_ids = set(identity_arbitration["effective_rejected"])
        rejected = [item for item in rejected if item["candidate_id"] in effective_ids]
        operational_count = identity_arbitration["operational_count"]
    rescue_additions = (len(identity_arbitration["admitted_rescue_groups"])
                        if identity_arbitration is not None else 0)
    baseline_merges = (
        identity_arbitration["baseline_identity_merge_reduction"]
        if identity_arbitration is not None else 0
    )

    if operational_count != (
        len(manifest["baseline"]) - len(rejected)
        - baseline_merges + rescue_additions
    ):
        raise AssertionError("operational correction invariant violated")

    if operational_count < 0:
        raise AssertionError("negative operational count")

    rescue_ids = {
        item["candidate_id"] for item in manifest.get("rescue", [])
    }

    unary = {
        cid: {"semantic": [], "whole_instance": [], "refs": []}
        for cid in rescue_ids
    }

    for event in observed_events:
        field = _UNARY.get(event.get("op"))
        cid = event.get("candidate_id")
        if field is None or cid not in unary:
            continue
        unary[cid][field].append(event["verdict"])
        unary[cid]["refs"].append(event["evidence_ref"])

    pending_rescues = []

    for cid in sorted(rescue_ids):
        evidence = unary[cid]
        semantic = _state(evidence["semantic"])
        whole = _state(evidence["whole_instance"])

        if semantic == "YES" and whole == "YES":
            pending_rescues.append({
                "candidate_id": cid,
                "semantic": semantic,
                "whole_instance": whole,
                "evidence_refs": sorted(set(evidence["refs"])),
                "admission": "DEFERRED_IDENTITY_ARBITRATION",
                "possible_delta_if_distinct": 1,
            })

    hints = []

    for item in shadow["unresolved"]:
        hints.append({
            "candidate_id": item["candidate_id"],
            "reason": "BASELINE_UNRESOLVED_UNARY",
            "semantic": item["semantic"],
            "whole_instance": item["whole_instance"],
            "possible_delta_if_new_dual_no": -1,
        })

    for item in pending_rescues:
        hints.append({
            "candidate_id": item["candidate_id"],
            "reason": "RESCUE_IDENTITY_UNRESOLVED",
            "possible_delta_if_distinct": 1,
        })

    if identity_arbitration is not None:
        pending_rescues = identity_arbitration["pending_rescues"]
        hints = [h for h in hints if h.get("reason") != "RESCUE_IDENTITY_UNRESOLVED"]
        hints.extend(identity_arbitration["pair_hints"])
        hints.extend({"candidate_id": x["candidate_id"],
                      "reason": "RECHECK_CONTRADICTORY_REJECTION"}
                     for x in identity_arbitration["protected_rejections"])

    return {
        "schema": "countlab_c3_dual_channel_v1",
        "policy": ("V14_REJECTION_ONLY_PLUS_T09_STRICT"
                   if identity_arbitration is None else
                   "V14_DUAL_NO_GUARDED_RESCUE_PLUS_T09_STRICT"),
        "scope": "OBSERVED_CANDIDATE_POOL_NOT_FULL_SCENE",
        "strict_decision": strict,
        "operational_prediction": {
            "count": operational_count,
            "baseline_count": fallback_count,
            "delta": operational_count - fallback_count,
            "rejection_count": len(rejected),
            "rescue_additions": rescue_additions,
            "baseline_identity_merge_reduction": baseline_merges,
            "scene_zero_certified": False,
            "full_scene_count_certified": False,
        },
        "correction_ledger": {
            "rejected_baseline": rejected,
            "retained_baseline_candidate_ids": (
                sorted(set(shadow["retained_candidate_ids"]) |
                       {x["candidate_id"] for x in identity_arbitration["protected_rejections"]})
                if identity_arbitration is not None else shadow["retained_candidate_ids"]),
            "unresolved_baseline": shadow["unresolved"],
            "pending_rescues": pending_rescues,
            "identity_arbitration": identity_arbitration,
            "coalesced_baseline_groups": (
                identity_arbitration["coalesced_baseline_groups"]
                if identity_arbitration is not None else []
            ),
        },
        "count_impact_hints_to_C1": hints,
        "evidence_provenance": {
            "manifest_sha256": shadow["manifest_sha256"],
            "observed_events_sha256": (
                strict["observed_events_sha256"]
                if rescue_policy == "guarded_identity"
                else shadow["observed_events_sha256"]
            ),
            "decision_projection_sha256": shadow["observed_events_sha256"],
            "ignored_unknown_evidence_refs": ignored_unknown_refs,
            "strict_solver_config_sha256": strict["solver_config_sha256"],
        },
        "evaluation_policy_unchanged": True,
        "executed_model_calls_by_C3": 0,
        "limitations": [
            "Operational count is a provisional evidence-based estimate.",
            "No scene-level zero or exhaustive count certification.",
            ("Rescue admission requires a separate identity-arbitration policy."
             if identity_arbitration is None else
             "Rescue admission is conditional on observed pairwise evidence, not full-scene completeness."),
            "The original T09 strict decision and fallback remain unchanged.",
        ],
    }
