from countlab.method.c3_dual_channel import solve_c3_dual_channel
from countlab.method.c3_count_impact_hints import analyze_count_critical_pairs
from countlab.method.joint_solver_v2 import solve_joint_v2
from countlab.method.v14_rejection_shadow import run_v14_rejection_shadow


def manifest(base=("b",), rescue=()):
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
        "image_id": "M10-test",
        "query_id": "M10-q",
        "query_text": "target object",
        "baseline_count": len(base),
        "baseline": [candidate(cid, "full") for cid in base],
        "rescue": [candidate(cid, "tile") for cid in rescue],
        "identity_edges": [],
    }


def unary(cid, semantic="YES", whole="YES"):
    return [
        {
            "op": "REGION_SEMANTIC",
            "candidate_id": cid,
            "verdict": semantic,
            "evidence_ref": "semantic:" + cid,
        },
        {
            "op": "REGION_WHOLE",
            "candidate_id": cid,
            "verdict": whole,
            "evidence_ref": "whole:" + cid,
        },
    ]


def pair(a, b, relation):
    return {
        "op": "PAIR_IDENTITY",
        "a": a,
        "b": b,
        "relation": relation,
        "evidence_ref": f"pair:{a}:{b}:{relation}",
    }


def solve(m, events):
    return solve_c3_dual_channel(
        m,
        events,
        fallback_count=len(m["baseline"]),
        rescue_policy="guarded_identity",
    )


def test_m10_unary_unknown_is_neutral_for_guarded_policy():
    m = manifest()
    events = unary("b", "NO", "NO")
    retry = {
        "op": "REGION_SEMANTIC",
        "candidate_id": "b",
        "verdict": "UNKNOWN",
        "evidence_ref": "semantic:b:unknown",
    }
    direct = solve(m, events)
    refined = solve(m, events + [retry])
    assert direct["operational_prediction"]["count"] == 0
    assert refined["operational_prediction"]["count"] == 0
    assert refined["strict_decision"] == solve_joint_v2(
        m, events + [retry], fallback_count=1
    )
    assert "semantic:b:unknown" in refined["evidence_provenance"][
        "ignored_unknown_evidence_refs"
    ]


def test_m10_v14_shadow_control_remains_unchanged():
    m = manifest()
    events = unary("b", "NO", "NO")
    retry = {
        "op": "REGION_SEMANTIC",
        "candidate_id": "b",
        "verdict": "UNKNOWN",
        "evidence_ref": "semantic:b:unknown",
    }
    original = run_v14_rejection_shadow(m, events, fallback_count=1)
    untouched = run_v14_rejection_shadow(
        m, events + [retry], fallback_count=1
    )
    assert original["operational_count"] == 0
    assert untouched["operational_count"] == 1


def test_m10_unknown_then_different_adds_rescue():
    m = manifest(("b",), ("r",))
    events = unary("r") + [
        pair("b", "r", "UNKNOWN"),
        pair("b", "r", "DIFFERENT"),
    ]
    result = solve(m, events)
    assert result["operational_prediction"]["count"] == 2
    assert result["operational_prediction"]["rescue_additions"] == 1


def test_m10_unknown_then_same_does_not_duplicate_rescue():
    m = manifest(("b",), ("r",))
    events = unary("r") + [
        pair("b", "r", "UNKNOWN"),
        pair("b", "r", "SAME"),
    ]
    assert solve(m, events)["operational_prediction"]["count"] == 1


def test_m10_decisive_identity_conflict_still_blocks():
    m = manifest(("b",), ("r",))
    events = unary("r") + [
        pair("b", "r", "SAME"),
        pair("b", "r", "DIFFERENT"),
    ]
    result = solve(m, events)
    assert result["operational_prediction"]["count"] == 1
    assert result["correction_ledger"]["identity_arbitration"]["conflicts"]


def test_m10_decisive_unary_conflict_still_blocks_rejection():
    m = manifest()
    events = unary("b", "NO", "NO")
    events.append({
        "op": "REGION_SEMANTIC",
        "candidate_id": "b",
        "verdict": "YES",
        "evidence_ref": "semantic:b:opposite",
    })
    assert solve(m, events)["operational_prediction"]["count"] == 1


def test_m10_unknown_pair_can_be_retried_by_router():
    m = manifest(("b",), ("r",))
    events = unary("r") + [pair("b", "r", "UNKNOWN")]
    result = analyze_count_critical_pairs(
        m, events, fallback_count=1
    )
    assert result["status"] == "COUNTERFACTUAL_HINTS_COMPUTED"
    assert result["ranking_complete"] is True
    assert result["pairs_scored"] == 1
    row = result["ranked_pair_hints"][0]
    assert row["if_same_count"] == 1
    assert row["if_different_count"] == 2
    assert row["maximum_absolute_count_change"] == 1


def test_m10_ranking_limit_applied_after_scoring():
    m = manifest(("b",), ("r1", "r2"))
    events = unary("r1") + unary("r2")
    result = analyze_count_critical_pairs(
        m, events, fallback_count=1, max_pairs=1
    )
    assert result["status"] == "COUNTERFACTUAL_HINTS_COMPUTED"
    assert result["pairs_scored"] == 2
    assert len(result["ranked_pair_hints"]) == 1
    assert result["pair_limit_reached"] is True
    assert result["ranking_complete"] is True


def test_m10_strict_and_fallback_do_not_change():
    m = manifest(("b",), ("r",))
    events = unary("r") + [
        pair("b", "r", "UNKNOWN"),
        pair("b", "r", "DIFFERENT"),
    ]
    result = solve(m, events)
    strict = solve_joint_v2(m, events, fallback_count=1)
    assert result["strict_decision"] == strict
    assert result["strict_decision"]["original_fallback_count"] == 1
    assert result["operational_prediction"]["scene_zero_certified"] is False
