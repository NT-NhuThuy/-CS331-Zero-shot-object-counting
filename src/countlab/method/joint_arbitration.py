"""Experimental GT-blind baseline/rescue arbitration (Thuy / T1).

This is intentionally separate from ``pipeline.run_method``.  It tests the
smallest joint decision missing from v1: an observed baseline--rescue relation
is considered before geometry suppresses a rescue.  It is not a learned
solver, is not the default method, and does not claim real-data improvement.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .schema import Proposal, box_iou, parse_input
from .router import select_for_verification
from .verifier import PrecomputedUnaryVerifier, decision
from .identity_graph import arbitrate_identity_graph
from .pairwise_allocation import select_pairwise_queries


def _canonical_relation(edges: dict[tuple[str, str], str], left: str, right: str) -> str:
    return edges.get(tuple(sorted((left, right))), "UNKNOWN")


def run_joint_arbitration(data: dict[str, Any], *, budget: int | None = None,
                          collision_iou: float = .5,
                          identity_min_same_iou: float = .10,
                          pairwise_budget: int | None = None,
                          pairwise_strategy: str = "impact",
                          abstain_on_unresolved: bool = False,
                          enable_recovery: bool = True,
                          enable_identity: bool = True) -> dict[str, Any]:
    """Run a conservative joint baseline/rescue decision on frozen evidence.

    Only an *observed* DIFFERENT relation permits an overlapping accepted
    rescue to add a physical instance beside a baseline.  SAME coalesces it
    with the baseline contribution. UNKNOWN keeps the baseline fallback but
    withholds rescue admission. This protects against both IoU-as-identity and
    unproven rescue additions while exposing exactly where pair evidence has
    decision value.
    """
    if not 0 < identity_min_same_iou <= collision_iou <= 1:
        raise ValueError("joint identity IoU policy is invalid")
    baseline, rescues, edges = parse_input(data)
    if not enable_recovery:
        rescues = []
    if not enable_identity:
        edges = {}
    candidates = baseline + rescues
    selected = select_for_verification(candidates, budget)
    verifier = PrecomputedUnaryVerifier()
    states = {item.candidate_id: decision(*verifier.verdict(item, selected=item.candidate_id in selected))
              for item in candidates}
    pairwise_plan = select_pairwise_queries(baseline, rescues, states, budget=pairwise_budget,
                                            collision_iou=collision_iou,
                                            identity_min_same_iou=identity_min_same_iou,
                                            strategy=pairwise_strategy)
    # In the fixed-evidence setting, an edge answer is visible only when its
    # pair was selected. The planner itself did not inspect edge outcomes.
    visible_pairs = {tuple(row["pair"]) for row in pairwise_plan}
    visible_edges = edges if pairwise_budget is None else {
        pair: relation for pair, relation in edges.items() if pair in visible_pairs}
    allocation_summary = {
        "n_planned_pairs": len(pairwise_plan),
        "n_collision_pairs": sum(row["purpose"] == "baseline_rescue_collision" for row in pairwise_plan),
        "n_duplicate_pairs": sum(row["purpose"] == "rescue_duplicate_resolution" for row in pairwise_plan),
        "relation_outcomes_used_for_planning": False,
    }

    rejected_baseline = {item.candidate_id for item in baseline if states[item.candidate_id] == "REJECTED"}
    baseline_contribution = {item.candidate_id for item in baseline if item.candidate_id not in rejected_baseline}
    added_rescues: set[str] = set()
    unresolved: dict[str, str] = {}
    traces: list[dict[str, Any]] = []
    # Physical partition used by the offline identity evaluator.  A rescue
    # coalesced with a baseline must be represented in the same component;
    # count equality alone is insufficient because it could hide a false
    # split in the audit output.
    predicted_components: list[list[str]] = [
        [item.candidate_id] for item in baseline
        if states[item.candidate_id] == "ACCEPTED"
    ]

    eligible_rescues = []
    for rescue in sorted(rescues, key=lambda item: item.candidate_id):
        state = states[rescue.candidate_id]
        if state == "REJECTED":
            traces.append({"rescue_id": rescue.candidate_id, "action": "REJECT", "reason": "dual_no"})
        elif state != "ACCEPTED":
            unresolved[rescue.candidate_id] = "unary_unknown_or_mixed"
            traces.append({"rescue_id": rescue.candidate_id, "action": "UNRESOLVED", "reason": "unary_unknown_or_mixed"})
        else:
            eligible_rescues.append(rescue)

    # First deduplicate eligible rescues by the pre-existing complete-link
    # safety rule. Baseline--rescue edges are intentionally excluded here:
    # this graph has a rescue-only API and joint handling follows below.
    eligible_ids = {item.candidate_id for item in eligible_rescues}
    rescue_edges = {pair: relation for pair, relation in visible_edges.items()
                    if pair[0] in eligible_ids and pair[1] in eligible_ids}
    graph = arbitrate_identity_graph(eligible_rescues, rescue_edges,
                                     min_same_iou=identity_min_same_iou, ambiguity_iou=collision_iou)
    for component in graph.unresolved_components:
        for candidate_id in component:
            unresolved[candidate_id] = "rescue_identity_unresolved"
        traces.append({"rescue_component": list(component), "action": "UNRESOLVED",
                       "reason": "rescue_identity_unresolved"})

    by_id = {item.candidate_id: item for item in eligible_rescues}
    for component in graph.accepted_components:
        members = [by_id[candidate_id] for candidate_id in component]
        component_id = "+".join(component)
        overlaps = [base for base in baseline
                    if any(box_iou(rescue.box, base.box) >= collision_iou for rescue in members)]
        if not overlaps:
            added_rescues.add(component_id)
            predicted_components.append(list(component))
            traces.append({"rescue_component": list(component), "action": "ADD", "reason": "no_baseline_collision"})
            continue

        # All member-to-baseline relations must support DIFFERENT before this
        # component may add one instance. A missing relation remains UNKNOWN.
        relation_sets = {base.candidate_id: {_canonical_relation(visible_edges, base.candidate_id, rescue.candidate_id)
                                             for rescue in members} for base in overlaps}
        relations = {candidate_id: sorted(values) for candidate_id, values in relation_sets.items()}
        # DIFFERENT is sufficient only if every colliding baseline is explicitly
        # different. A mixture with SAME/UNKNOWN is a real conflict, not a vote.
        if all(values == {"DIFFERENT"} for values in relation_sets.values()):
            added_rescues.add(component_id)
            predicted_components.append(list(component))
            traces.append({"rescue_component": list(component), "action": "ADD", "reason": "observed_different",
                           "relations": relations})
        elif any("SAME" in values for values in relation_sets.values()) and all(
                values <= {"SAME", "DIFFERENT"} for values in relation_sets.values()):
            traces.append({"rescue_component": list(component), "action": "COALESCE", "reason": "observed_same",
                           "relations": relations})
            same_baselines = [base.candidate_id for base in overlaps
                              if "SAME" in relation_sets[base.candidate_id]
                              and states[base.candidate_id] == "ACCEPTED"]
            if same_baselines:
                target = next(group for group in predicted_components
                              if same_baselines[0] in group)
                target.extend(candidate_id for candidate_id in component
                              if candidate_id not in target)
            for base in overlaps:
                if "SAME" in relation_sets[base.candidate_id] and states[base.candidate_id] != "ACCEPTED":
                    unresolved[base.candidate_id] = "same_identity_unary_conflict"
                    # A rejected baseline and accepted rescue claiming SAME is
                    # an unresolved physical instance, never a silent deletion.
                    baseline_contribution.add(base.candidate_id)
        else:
            for rescue in members:
                unresolved[rescue.candidate_id] = "collision_relation_unknown_or_conflict"
            traces.append({"rescue_component": list(component), "action": "UNRESOLVED",
                           "reason": "collision_relation_unknown_or_conflict", "relations": relations})

    # A baseline remains the conservative fallback unless it has dual-NO.
    # This deliberately preserves v1's no-evidence monotonicity.
    point_count = len(baseline_contribution) + len(added_rescues)
    # Fallback contribution and verified acceptance are intentionally separate:
    # an UNKNOWN baseline can preserve the conservative count but is not an
    # accepted physical-instance claim.
    accepted = sorted(({item.candidate_id for item in baseline if states[item.candidate_id] == "ACCEPTED"}
                       | added_rescues) - set(unresolved))
    # Keep the physical-instance partition separate from the public shorthand
    # candidate IDs.  In particular, a rescue component has a synthetic
    # representative such as ``r1+r2`` and must not be re-expanded later by
    # an evaluator as two singleton claims.
    predicted_components = [
        [candidate_id for candidate_id in component if candidate_id not in unresolved]
        for component in predicted_components if any(candidate_id not in unresolved for candidate_id in component)
    ]
    candidate_ids = {item.candidate_id for item in candidates}
    flat_partition = [candidate_id for component in predicted_components for candidate_id in component]
    if any(not component for component in predicted_components):
        raise AssertionError("joint physical partition cannot contain empty components")
    if any(candidate_id not in candidate_ids for candidate_id in flat_partition):
        raise AssertionError("joint physical partition contains an unknown candidate")
    if len(flat_partition) != len(set(flat_partition)):
        raise AssertionError("joint physical partition duplicates a candidate")
    # These values are deliberately candidate-scoped diagnostics. They cannot
    # certify a full-scene count because a proposal may be absent altogether.
    lower = len(accepted)
    upper = point_count + sum(candidate_id not in baseline_contribution for candidate_id in unresolved)
    if upper < point_count:
        raise AssertionError("joint observed-candidate interval is inconsistent")
    abstained = bool(abstain_on_unresolved and unresolved)
    return {
        "schema": "countlab_joint_arbitration_experimental_v1",
        "experimental": True,
        "joint_policy": {"collision_iou": collision_iou,
                         "identity_min_same_iou": identity_min_same_iou,
                         "pairwise_budget": pairwise_budget,
                         "pairwise_strategy": pairwise_strategy,
                         "abstain_on_unresolved": abstain_on_unresolved,
                         "enable_recovery": enable_recovery,
                         "enable_identity": enable_identity},
        "input_sha256": hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "count": None if abstained else point_count,
        "evaluation_fallback_count": point_count,
        "count_lower_bound": lower,
        "count_upper_bound": upper,
        "count_interval_scope": "observed_candidates_only_not_scene_ground_truth_bounds",
        "count_status": "abstained" if abstained else "partial",
        "query_abstained": abstained,
        "fallback_baseline_candidate_ids": sorted(baseline_contribution),
        "selected_candidate_ids": sorted(selected),
        "pairwise_query_plan": pairwise_plan,
        "pairwise_allocation_summary": allocation_summary,
        "visible_pairwise_edge_count": len(visible_edges),
        "accepted_candidate_ids": accepted,
        "accepted_components": [list(component) for component in graph.accepted_components
                                 if all(candidate_id not in unresolved for candidate_id in component)],
        "predicted_components": predicted_components,
        "rejected_candidate_ids": sorted(rejected_baseline | {item.candidate_id for item in rescues if states[item.candidate_id] == "REJECTED"}),
        "unresolved_candidates": [{"candidate_id": key, "reason": unresolved[key]} for key in sorted(unresolved)],
        "joint_trace": traces,
        "limitations": ["Synthetic/CPU experimental arbitration only", "No calibration or real-data effectiveness claim",
                        "Only observed pairwise relations change collision handling", "Does not replace v1 pipeline"],
    }
