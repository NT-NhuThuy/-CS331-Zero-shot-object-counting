import copy

import pytest

from countlab.method.c3_dual_channel import solve_c3_dual_channel
from countlab.method.c3_count_impact_hints import analyze_count_critical_pairs


def manifest(base=(), rescue=()):
    def candidate(cid, view):
        return {
            "candidate_id": cid,
            "view_id": view,
            "box_xyxy": [0, 0, 20, 20],
            "evidence": {
                "semantic": "UNKNOWN",
                "whole_instance": "UNKNOWN",
            },
            "evidence_refs": [],
        }

    return {
        "schema": "countlab_candidate_pool_v1",
        "image_id": "test-image",
        "query_id": "test-query",
        "query_text": "object",
        "baseline_count": len(base),
        "baseline": [candidate(x, "full") for x in base],
        "rescue": [candidate(x, "tile") for x in rescue],
        "identity_edges": [],
    }


def unary(cid, semantic="YES", whole="YES"):
    return [
        {
            "op": "REGION_SEMANTIC",
            "candidate_id": cid,
            "verdict": semantic,
            "evidence_ref": f"semantic:{cid}",
        },
        {
            "op": "REGION_WHOLE",
            "candidate_id": cid,
            "verdict": whole,
            "evidence_ref": f"whole:{cid}",
        },
    ]


def pair(a, b, relation):
    return {
        "op": "PAIR_IDENTITY",
        "a": a,
        "b": b,
        "relation": relation,
        "evidence_ref": f"observed:{a}:{b}:{relation}",
    }


def analyze(m, events):
    return analyze_count_critical_pairs(
        m, events, fallback_count=len(m["baseline"])
    )


def test_m8_distinct_rescue_changes_operational_count():
    m = manifest(("b",), ("r",))
    events = unary("r") + [pair("b", "r", "DIFFERENT")]
    result = solve_c3_dual_channel(
        m, events, fallback_count=1,
        rescue_policy="guarded_identity",
    )
    assert result["operational_prediction"]["count"] == 2
    assert result["operational_prediction"]["rescue_additions"] == 1
    assert result["operational_prediction"]["scene_zero_certified"] is False


def test_m8_semantic_identity_contradiction_protects_rejection():
    m = manifest(("b",), ("r",))
    events = (
        unary("b", "NO", "NO")
        + unary("r")
        + [pair("b", "r", "SAME")]
    )
    result = solve_c3_dual_channel(
        m, events, fallback_count=1,
        rescue_policy="guarded_identity",
    )
    assert result["operational_prediction"]["count"] == 1
    assert len(
        result["correction_ledger"]["identity_arbitration"]["protected_rejections"]
    ) == 1


def test_m9_baseline_rescue_identity_is_count_critical():
    m = manifest(("b",), ("r",))
    result = analyze(m, unary("r"))
    assert result["status"] == "COUNTERFACTUAL_HINTS_COMPUTED"
    assert result["current_operational_count"] == 1
    assert result["pairs_scored"] == 1
    row = result["ranked_pair_hints"][0]
    assert (row["a"], row["b"]) == ("b", "r")
    assert row["if_same_count"] == 1
    assert row["if_different_count"] == 2
    assert row["maximum_absolute_count_change"] == 1
    assert row["hypotheses_only"] is True


def test_m9_two_rescue_identity_changes_count():
    m = manifest((), ("r1", "r2"))
    result = analyze(m, unary("r1") + unary("r2"))
    assert result["current_operational_count"] == 1
    row = result["ranked_pair_hints"][0]
    assert row["if_same_count"] == 1
    assert row["if_different_count"] == 2


def test_m9_unknown_pair_remains_retryable_without_overwriting_evidence():
    m = manifest(("b",), ("r",))
    events = unary("r") + [pair("b", "r", "UNKNOWN")]
    before_manifest = copy.deepcopy(m)
    before_events = copy.deepcopy(events)

    result = analyze(m, events)

    assert result["status"] == "COUNTERFACTUAL_HINTS_COMPUTED"
    assert result["ranking_complete"] is True
    assert result["pairs_scored"] == 1
    assert result["skipped_existing_pairs"] == []

    row = result["ranked_pair_hints"][0]
    assert (row["a"], row["b"]) == ("b", "r")
    assert row["current_count"] == 1
    assert row["if_same_count"] == 1
    assert row["if_different_count"] == 2
    assert row["maximum_absolute_count_change"] == 1
    assert row["hypotheses_only"] is True

    assert result["counterfactual_relations_observed"] is False
    assert result["ground_truth_used"] is False
    assert result["new_model_calls"] == 0
    assert m == before_manifest
    assert events == before_events


def test_m9_no_pair_hint_without_rescue():
    m = manifest(("b",), ())
    result = analyze(m, unary("b", "NO", "NO"))
    assert result["current_operational_count"] == 0
    assert result["ranked_pair_hints"] == []
    assert result["counterfactual_relations_observed"] is False


def test_m9_preserves_inputs():
    m = manifest(("b",), ("r",))
    events = unary("r")
    before_manifest = copy.deepcopy(m)
    before_events = copy.deepcopy(events)
    analyze(m, events)
    assert m == before_manifest
    assert events == before_events


def test_m9_rejects_invalid_event_contract():
    m = manifest(("b",), ("r",))
    events = unary("r")
    events[0]["query_id"] = "unexpected-field"
    with pytest.raises(ValueError):
        analyze(m, events)


def test_t12_baseline_same_coalescing():
    m = manifest(("b1", "b2"), ())
    events = unary("b1") + unary("b2") + [
        pair("b1", "b2", "SAME")
    ]

    result = solve_c3_dual_channel(
        m, events, fallback_count=2,
        rescue_policy="guarded_identity",
    )

    assert result["operational_prediction"]["count"] == 1
    assert result["operational_prediction"]["baseline_identity_merge_reduction"] == 1
    assert result["strict_decision"]["count"] == 1
    assert result["strict_decision"]["status"] == "PROVISIONAL_POSITIVE"
    assert len(result["correction_ledger"]["coalesced_baseline_groups"]) == 1


def test_t12_q31_style_same_without_unary_is_conditional_only():
    m = manifest(("q31:b1", "q31:b2"), ())
    events = [pair("q31:b1", "q31:b2", "SAME")]

    guarded = solve_c3_dual_channel(
        m, events, fallback_count=2,
        rescue_policy="guarded_identity",
    )
    original = solve_c3_dual_channel(
        m, events, fallback_count=2
    )

    assert guarded["operational_prediction"]["count"] == 1
    assert guarded["strict_decision"]["status"] == "ABSTAIN"
    assert guarded["strict_decision"]["evaluation_fallback_count"] == 2
    assert original["operational_prediction"]["count"] == 2
    assert guarded["operational_prediction"]["scene_zero_certified"] is False


def test_t12_distinct_witness_survives_same_refinement():
    m = manifest((), ("a", "b", "c"))
    events = unary("a") + unary("b") + unary("c")
    before = events + [pair("b", "c", "DIFFERENT")]
    after = before + [pair("a", "b", "SAME")]

    first = solve_c3_dual_channel(
        m, before, fallback_count=0,
        rescue_policy="guarded_identity",
    )
    second = solve_c3_dual_channel(
        m, after, fallback_count=0,
        rescue_policy="guarded_identity",
    )

    assert first["operational_prediction"]["count"] == 2
    assert second["operational_prediction"]["count"] == 2
    assert second["strict_decision"]["count"] == 2
    assert len(
        second["correction_ledger"]["identity_arbitration"]["admitted_rescue_groups"]
    ) == 2


def test_t12_one_distinct_witness_between_baseline_group_and_rescue():
    m = manifest(("b1", "b2"), ("r",))
    events = unary("r") + [
        pair("b1", "b2", "SAME"),
        pair("b2", "r", "DIFFERENT"),
    ]

    result = solve_c3_dual_channel(
        m, events, fallback_count=2,
        rescue_policy="guarded_identity",
    )

    assert result["operational_prediction"]["count"] == 2
    assert result["operational_prediction"]["baseline_identity_merge_reduction"] == 1
    assert result["operational_prediction"]["rescue_additions"] == 1
    assert result["strict_decision"]["status"] == "ABSTAIN"


def test_t12_missing_identity_does_not_coalesce_baselines():
    m = manifest(("b1", "b2"), ())
    result = solve_c3_dual_channel(
        m, [], fallback_count=2,
        rescue_policy="guarded_identity",
    )

    assert result["operational_prediction"]["count"] == 2
    assert result["operational_prediction"]["baseline_identity_merge_reduction"] == 0


def test_t12_incomplete_or_contradictory_identity_does_not_merge():
    m = manifest(("a", "b", "c"), ())
    events = [
        pair("a", "b", "SAME"),
        pair("b", "c", "SAME"),
        pair("a", "c", "DIFFERENT"),
    ]

    result = solve_c3_dual_channel(
        m, events, fallback_count=3,
        rescue_policy="guarded_identity",
    )

    assert result["operational_prediction"]["count"] == 3
    assert result["operational_prediction"]["baseline_identity_merge_reduction"] == 0
    assert result["strict_decision"]["status"] == "ABSTAIN"
    assert result["correction_ledger"]["identity_arbitration"]["conflicts"]


def test_t12_same_yes_no_conflict_protects_baseline():
    m = manifest(("b1", "b2"), ())
    events = (
        unary("b1", "NO", "NO")
        + unary("b2")
        + [pair("b1", "b2", "SAME")]
    )

    result = solve_c3_dual_channel(
        m, events, fallback_count=2,
        rescue_policy="guarded_identity",
    )

    assert result["operational_prediction"]["count"] == 2
    assert result["operational_prediction"]["baseline_identity_merge_reduction"] == 0
    assert len(
        result["correction_ledger"]["identity_arbitration"]["protected_rejections"]
    ) == 1


def test_t12_rescue_requires_distinctness_from_every_baseline_group():
    m = manifest(("b1", "b2", "b3"), ("r",))
    events = unary("r") + [
        pair("b1", "b2", "SAME"),
        pair("b1", "r", "DIFFERENT"),
    ]

    result = solve_c3_dual_channel(
        m, events, fallback_count=3,
        rescue_policy="guarded_identity",
    )

    assert result["operational_prediction"]["count"] == 2
    assert result["operational_prediction"]["baseline_identity_merge_reduction"] == 1
    assert result["operational_prediction"]["rescue_additions"] == 0
