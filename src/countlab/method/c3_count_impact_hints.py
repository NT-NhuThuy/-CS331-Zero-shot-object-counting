"""GT-blind hypothetical pairwise count-impact analysis.

Counterfactual relations are never observed verifier evidence.
This module only supplies diagnostic hints to C1.
"""

from collections import defaultdict

from countlab.method.joint_solver_v2 import validate_event_contract
from countlab.method.v14_rejection_shadow import run_v14_rejection_shadow
from countlab.method.c3_rescue_arbitration import reconcile_rescues
from countlab.method.c3_evidence_refinement import project_decisive_events


def analyze_count_critical_pairs(
    manifest,
    observed_events,
    *,
    fallback_count,
    max_pairs=12,
    max_candidates=512,
    max_search_states=20000,
):
    if (
        isinstance(max_pairs, bool)
        or not isinstance(max_pairs, int)
        or not 0 <= max_pairs <= 64
    ):
        raise ValueError("max_pairs must be an integer in [0,64]")

    if (
        isinstance(max_search_states, bool)
        or not isinstance(max_search_states, int)
        or max_search_states < 1
    ):
        raise ValueError("invalid max_search_states")

    validate_event_contract(observed_events)
    decision_events, ignored_unknown_refs = project_decisive_events(observed_events)

    shadow = run_v14_rejection_shadow(
        manifest,
        decision_events,
        fallback_count=fallback_count,
    )

    options = {
        "max_candidates": max_candidates,
        "max_search_states": max_search_states,
        "max_hints": max(96, max_pairs * 4),
    }

    current = reconcile_rescues(
        manifest,
        decision_events,
        shadow,
        **options,
    )

    output = {
        "schema": "countlab_c3_count_impact_hints_v1",
        "policy": "COUNTERFACTUAL_DIRECT_IDENTITY_LEVERAGE",
        "query_id": str(manifest["query_id"]),
        "current_operational_count": current["operational_count"],
        "current_search_exact": current["search_exact"],
        "ranked_pair_hints": [],
        "skipped_existing_pairs": [],
        "skipped_search_limited_pairs": [],
        "analysis_only": True,
        "counterfactual_relations_observed": False,
        "new_model_calls": 0,
        "requires_C1_action_selection": True,
        "ground_truth_used": False,
        "manifest_sha256": shadow["manifest_sha256"],
        "observed_events_sha256": shadow["observed_events_sha256"],
        "ignored_unknown_evidence_refs": ignored_unknown_refs,
        "ranking_complete": False,
    }

    if not current["search_exact"]:
        output["status"] = "HOLD_CURRENT_SEARCH_NOT_EXACT"
        return output

    if current["hints_truncated"]:
        output["status"] = "HOLD_INCOMPLETE_HINT_COHORT"
        return output

    observed_pairs = set()

    for event in decision_events:
        if event["op"] == "PAIR_IDENTITY":
            observed_pairs.add(
                tuple(sorted((event["a"], event["b"])))
            )

    reasons = defaultdict(set)

    for hint in current["pair_hints"]:
        if hint.get("op") != "PAIR_IDENTITY":
            continue

        a, b = sorted((hint["a"], hint["b"]))
        if a == b:
            continue

        reasons[(a, b)].add(
            hint.get("reason", "identity_uncertainty")
        )

    scored = []

    for a, b in sorted(reasons):
        if (a, b) in observed_pairs:
            output["skipped_existing_pairs"].append([a, b])
            continue

        counterfactual_counts = {}
        exact = True

        for relation in ("SAME", "DIFFERENT"):
            hypothetical = {
                "op": "PAIR_IDENTITY",
                "a": a,
                "b": b,
                "relation": relation,
                "evidence_ref": (
                    "counterfactual-not-observed:"
                    + a + ":" + b + ":" + relation
                ),
            }

            simulated = reconcile_rescues(
                manifest,
                decision_events + [hypothetical],
                shadow,
                **options,
            )

            if not simulated["search_exact"]:
                exact = False
                break

            counterfactual_counts[relation] = (
                simulated["operational_count"]
            )

        if not exact:
            output["skipped_search_limited_pairs"].append([a, b])
            continue

        actual = current["operational_count"]
        same = counterfactual_counts["SAME"]
        different = counterfactual_counts["DIFFERENT"]

        scored.append({
            "a": a,
            "b": b,
            "reasons": sorted(reasons[(a, b)]),
            "current_count": actual,
            "if_same_count": same,
            "if_different_count": different,
            "if_same_delta": same - actual,
            "if_different_delta": different - actual,
            "maximum_absolute_count_change": max(
                abs(same - actual),
                abs(different - actual),
            ),
            "outcome_count_span": abs(different - same),
            "hypotheses_only": True,
        })

    output["ranked_pair_hints"] = sorted(
        scored,
        key=lambda x: (
            -x["maximum_absolute_count_change"],
            -x["outcome_count_span"],
            x["a"],
            x["b"],
        ),
    )[:max_pairs]

    output["ranking_complete"] = not bool(output["skipped_search_limited_pairs"])
    output["status"] = (
        "COUNTERFACTUAL_HINTS_COMPUTED"
        if output["ranking_complete"]
        else "PARTIAL_SEARCH_LIMIT"
    )
    output["source_pair_hint_count"] = len(reasons)
    output["pairs_scored"] = len(scored)
    output["pair_limit_reached"] = len(scored) > max_pairs

    return output
