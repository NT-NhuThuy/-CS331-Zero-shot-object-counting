"""T09 experimental, bounded, GT-blind observed-candidate count reasoning.

All assertions about membership/identity are CONDITIONAL on supplied verifier
answers being correct. Absent scene completeness cannot be established here.
This module has no inference/backend/controller side effects.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

from .schema import box_iou, parse_input
from .t03_integrity import validate_joint_t1_integrity


def _visible(manifest: dict, prediction: dict) -> dict[tuple[str, str], str]:
    _, _, edges = parse_input(manifest)
    policy = prediction["joint_policy"]
    if not policy["enable_identity"]:
        return {}
    if policy["pairwise_budget"] is None:
        return edges
    planned = {tuple(row["pair"]) for row in prediction["pairwise_query_plan"]}
    return {key: value for key, value in edges.items() if key in planned}


def _candidate_state(proposal: Any, selected: set[str]) -> str:
    if proposal.candidate_id not in selected:
        return "UNKNOWN"
    if proposal.semantic == proposal.whole == "YES":
        return "YES"
    if proposal.semantic == proposal.whole == "NO":
        return "NO"
    return "UNKNOWN"


def _feasible_counts(nodes: list[tuple[str, ...]], status: list[str], different: set[tuple[int, int]]) -> list[int]:
    """Enumerate possible distinct-instance counts, without pairwise transitivity shortcuts."""
    possible: set[int] = set()
    # Each group is an explicit SAME component, not an automatically accepted object.
    def visit(index: int, buckets: list[set[int]]) -> None:
        if index == len(nodes):
            possible.add(len(buckets))
            return
        if status[index] == "NO":
            visit(index + 1, buckets)
            return
        if status[index] == "UNKNOWN":
            visit(index + 1, buckets)
        for bucket in buckets:
            if all(tuple(sorted((index, j))) not in different for j in bucket):
                bucket.add(index)
                visit(index + 1, buckets)
                bucket.remove(index)
        buckets.append({index})
        visit(index + 1, buckets)
        buckets.pop()
    visit(0, [])
    return sorted(possible)


def reason_observed_instances(manifest: dict, t1_prediction: dict, *, max_groups: int = 8) -> dict:
    """Produce a conditional count set and safe decision without rewriting T1.

    SAME is treated as an equality constraint, DIFFERENT as an inequality.
    UNKNOWN leaves both identities possible. A missing direct SAME across a
    transitive chain triggers review rather than automatically merging count.
    This is bounded feasibility, not a probabilistic or visually certified solver.
    """
    if type(max_groups) is not int or not 1 <= max_groups <= 10:
        raise ValueError("max_groups must be an integer in [1,10]")
    integrity = validate_joint_t1_integrity(manifest, t1_prediction)
    baseline, rescue, _ = parse_input(manifest)
    if not t1_prediction["joint_policy"]["enable_recovery"]:
        rescue = []
    candidates = baseline + rescue
    by_id = {p.candidate_id: p for p in candidates}
    selected = set(t1_prediction["selected_candidate_ids"])
    visible = {pair: rel for pair, rel in _visible(manifest, t1_prediction).items()
               if pair[0] in by_id and pair[1] in by_id}

    parent = {cid: cid for cid in by_id}
    def root(cid: str) -> str:
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid
    def union(a: str, b: str) -> None:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    for (a, b), relation in sorted(visible.items()):
        if relation == "SAME":
            union(a, b)

    grouped: dict[str, list[str]] = {}
    for cid in sorted(by_id):
        grouped.setdefault(root(cid), []).append(cid)
    nodes = [tuple(value) for _, value in sorted(grouped.items())]
    lookup = {cid: i for i, members in enumerate(nodes) for cid in members}
    reasons: list[str] = []
    states: list[str] = []
    for members in nodes:
        observed = {_candidate_state(by_id[cid], selected) for cid in members}
        if "YES" in observed and "NO" in observed:
            reasons.append("semantic_identity_conflict:" + ",".join(members))
        states.append("YES" if "YES" in observed else "NO" if "NO" in observed else "UNKNOWN")
        for a, b in combinations(members, 2):
            if visible.get((a, b)) != "SAME":
                reasons.append("unconfirmed_transitive_same:" + a + "," + b)

    different: set[tuple[int, int]] = set()
    for (a, b), relation in sorted(visible.items()):
        if relation != "DIFFERENT":
            continue
        i, j = lookup[a], lookup[b]
        if i == j:
            reasons.append("same_different_contradiction:" + a + "," + b)
        else:
            different.add(tuple(sorted((i, j))))

    # Legacy T1's physical partition can contradict already-observed identity.
    reasons.extend("legacy_partition_risk:" + entry for entry in integrity["structural_risks"])
    issues = sorted(set(reasons))
    statuses = {"INCONSISTENT" if any(x.startswith(("semantic_identity_conflict:",
                      "same_different_contradiction:")) for x in issues) else "CONSISTENT"}
    feasible: list[int] | None = None
    if len(nodes) > max_groups:
        issues.append("resource_limit:too_many_identity_groups")
    elif "INCONSISTENT" in statuses:
        feasible = []
    else:
        feasible = _feasible_counts(nodes, states, different)
        if not feasible:
            issues.append("no_admissible_assignment")

    # Diagnostic action hints only. Execution and budgets belong to C1/C2.
    hints: list[dict] = []
    for reason in issues:
        if reason.startswith("unconfirmed_transitive_same:"):
            a, b = reason.split(":", 1)[1].split(",", 1)
            hints.append({"type": "PAIR_IDENTITY", "candidate_ids": [a, b],
                          "reason": "confirm_implied_same", "executed": False})
    if not candidates:
        hints.append({"type": "RECOVER_VIEW", "candidate_ids": [],
                      "reason": "no_candidate_does_not_certify_absence", "executed": False})
    if feasible is not None and len(feasible) > 1 and not hints:
        mandatory = [members[0] for members, state in zip(nodes, states) if state == "YES"]
        for a, b in combinations(mandatory, 2):
            key = tuple(sorted((a, b)))
            if key not in visible:
                hints.append({"type": "PAIR_IDENTITY", "candidate_ids": list(key),
                              "reason": "ambiguous_count", "executed": False})
                break

    # An observed-pool zero is never proof of whole-scene target absence.
    safe_positive = bool(feasible and len(feasible) == 1 and feasible[0] > 0 and not issues)
    count = feasible[0] if safe_positive else None
    return {
        "schema": "countlab_t09_admissible_worlds_experimental_v0",
        "scope": "observed_candidates_conditional_on_verifier_evidence_not_scene_bounds",
        "status": "POSITIVE" if safe_positive else "ABSTAIN",
        "count": count,
        "evaluation_fallback_count": t1_prediction["evaluation_fallback_count"],
        "feasible_observed_counts": feasible,
        "issues": sorted(set(issues)),
        "same_components": [list(x) for x in nodes],
        "component_membership_states": states,
        "suggested_evidence_actions": hints,
        "input_sha256": integrity["input_sha256"],
        "t1_prediction_sha256": integrity["prediction_sha256"],
        "budget_visibility": "all_supplied_edges_unmetered" if t1_prediction["joint_policy"]["pairwise_budget"] is None else "selected_pairwise_plan_only",
        "limitations": ["Unary/pairwise verifier answers are not visual truth",
                        "No coverage certificate for missing scene objects",
                        "Feasible counts may shrink spuriously when model answers are wrong",
                        "No empirical claim of improvement or calibrated risk"],
    }
