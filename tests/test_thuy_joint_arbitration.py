import json
import pytest

from countlab.method.joint_arbitration import run_joint_arbitration
from countlab.method.pipeline import run_method


def p(cid, box, semantic="UNKNOWN", whole="UNKNOWN", view="full"):
    return {"candidate_id": cid, "box_xyxy": box, "view_id": view,
            "evidence": {"semantic": semantic, "whole_instance": whole}}


def m(baseline, rescue, edges=()):
    return {"schema": "countlab_candidate_pool_v1", "image_id": "i", "query_id": "q", "query_text": "x",
            "baseline": baseline, "baseline_count": len(baseline), "rescue": rescue, "identity_edges": list(edges)}


def test_observed_different_prevents_v1_iou_suppression_of_distinct_rescue():
    data = m([p("b", [0, 0, 10, 10], "YES", "YES")], [p("r", [1, 0, 11, 10], "YES", "YES", "tile")],
             [{"a": "b", "b": "r", "relation": "DIFFERENT"}])
    assert run_method(data)["count"] == 1
    out = run_joint_arbitration(data)
    assert out["count"] == 2
    assert out["joint_trace"][0]["reason"] == "observed_different"


def test_same_relation_coalesces_without_double_counting():
    data = m([p("b", [0, 0, 10, 10], "YES", "YES")], [p("r", [1, 0, 11, 10], "YES", "YES", "tile")],
             [{"a": "b", "b": "r", "relation": "SAME"}])
    out = run_joint_arbitration(data)
    assert out["count"] == 1 and out["joint_trace"][0]["action"] == "COALESCE"
    assert out["predicted_components"] == [["b", "r"]]


def test_unknown_collision_never_promotes_rescue():
    data = m([p("b", [0, 0, 10, 10])], [p("r", [1, 0, 11, 10], "YES", "YES", "tile")])
    out = run_joint_arbitration(data)
    assert out["count"] == 1
    assert out["unresolved_candidates"] == [{"candidate_id": "r", "reason": "collision_relation_unknown_or_conflict"}]


def test_dual_no_baseline_and_different_rescue_is_bidirectional_correction():
    data = m([p("b", [0, 0, 10, 10], "NO", "NO")], [p("r", [1, 0, 11, 10], "YES", "YES", "tile")],
             [{"a": "b", "b": "r", "relation": "DIFFERENT"}])
    out = run_joint_arbitration(data)
    assert out["count"] == 1
    assert out["accepted_candidate_ids"] == ["r"]
    assert out["rejected_candidate_ids"] == ["b"]


def test_budget_masks_evidence_to_unknown():
    data = m([p("b", [0, 0, 10, 10], "NO", "NO")], [], [])
    out = run_joint_arbitration(data, budget=0)
    assert out["count"] == 1 and out["rejected_candidate_ids"] == []


def test_duplicate_rescues_are_counted_once_before_joint_baseline_handling():
    data = m([], [p("r1", [20, 0, 30, 10], "YES", "YES", "tile1"),
                  p("r2", [21, 0, 31, 10], "YES", "YES", "tile2")],
             [{"a": "r1", "b": "r2", "relation": "SAME"}])
    out = run_joint_arbitration(data)
    assert out["count"] == 1
    assert out["accepted_candidate_ids"] == ["r1+r2"]


def test_unsupported_rescue_bridge_is_unresolved_not_three_additions():
    data = m([], [p("a", [0, 0, 10, 10], "YES", "YES", "v1"),
                  p("b", [1, 0, 11, 10], "YES", "YES", "v2"),
                  p("c", [2, 0, 12, 10], "YES", "YES", "v3")],
             [{"a": "a", "b": "b", "relation": "SAME"}, {"a": "b", "b": "c", "relation": "SAME"}])
    out = run_joint_arbitration(data)
    assert out["count"] == 0
    assert {row["candidate_id"] for row in out["unresolved_candidates"]} == {"a", "b", "c"}


def test_same_conflict_restores_rejected_baseline_as_unresolved_fallback():
    data = m([p("b", [0, 0, 10, 10], "NO", "NO")], [p("r", [1, 0, 11, 10], "YES", "YES", "tile")],
             [{"a": "b", "b": "r", "relation": "SAME"}])
    out = run_joint_arbitration(data)
    assert out["count"] == 1
    assert out["unresolved_candidates"] == [{"candidate_id": "b", "reason": "same_identity_unary_conflict"}]


def test_joint_output_is_deterministic_and_json_serializable():
    data = m([p("b", [0, 0, 10, 10], "YES", "YES")],
             [p("r2", [1, 0, 11, 10], "YES", "YES", "v2"),
              p("r1", [1, 0, 11, 10], "YES", "YES", "v1")],
             [{"a": "b", "b": "r1", "relation": "DIFFERENT"},
              {"a": "b", "b": "r2", "relation": "DIFFERENT"},
              {"a": "r1", "b": "r2", "relation": "SAME"}])
    first = run_joint_arbitration(data)
    second = run_joint_arbitration({**data, "rescue": list(reversed(data["rescue"]))})
    assert {key: value for key, value in first.items() if key != "input_sha256"} == {
        key: value for key, value in second.items() if key != "input_sha256"}
    json.dumps(first, sort_keys=True)


def test_joint_partition_is_a_unique_known_candidate_partition():
    out = run_joint_arbitration(
        m([p("b", [0, 0, 10, 10], "YES", "YES")],
          [p("r", [1, 0, 11, 10], "YES", "YES", "tile")],
          [{"a": "b", "b": "r", "relation": "SAME"}])
    )
    flat = [candidate_id for component in out["predicted_components"] for candidate_id in component]
    assert flat == ["b", "r"]
    assert len(flat) == len(set(flat))


def test_unresolved_collision_can_explicitly_abstain_with_fallback():
    data = m([p("b", [0, 0, 10, 10])], [p("r", [1, 0, 11, 10], "YES", "YES", "tile")])
    out = run_joint_arbitration(data, abstain_on_unresolved=True)
    assert out["count"] is None and out["query_abstained"]
    assert out["evaluation_fallback_count"] == 1
    assert out["count_status"] == "abstained"
    assert out["count_interval_scope"] == "observed_candidates_only_not_scene_ground_truth_bounds"


def test_no_unresolved_case_does_not_abstain():
    out = run_joint_arbitration(m([], [p("r", [20, 0, 30, 10], "YES", "YES", "tile")]),
                                abstain_on_unresolved=True)
    assert out["count"] == 1 and not out["query_abstained"]


def test_joint_identity_policy_is_explicit_and_validated():
    data = m([], [p("r", [20, 0, 30, 10], "YES", "YES", "tile")])
    assert run_joint_arbitration(data, collision_iou=.4, identity_min_same_iou=.2)["joint_policy"]["collision_iou"] == .4
    with pytest.raises(ValueError, match="policy"):
        run_joint_arbitration(data, collision_iou=.2, identity_min_same_iou=.3)
    with pytest.raises(ValueError, match="strategy"):
        run_joint_arbitration(data, pairwise_strategy="future_learned_router")


def test_pairwise_budget_reveals_only_relation_blind_selected_collision():
    data = m([p("b", [0, 0, 10, 10], "YES", "YES")],
             [p("r", [1, 0, 11, 10], "YES", "YES", "tile")],
             [{"a": "b", "b": "r", "relation": "DIFFERENT"}])
    no_pairwise = run_joint_arbitration(data, pairwise_budget=0)
    queried = run_joint_arbitration(data, pairwise_budget=1)
    assert no_pairwise["count"] == 1
    assert queried["count"] == 2
    row = queried["pairwise_query_plan"]
    assert len(row) == 1 and row[0]["pair"] == ["b", "r"]
    assert row[0]["priority"] == 4 and row[0]["iou"] == pytest.approx(9 / 11)
    assert row[0]["strategy"] == "impact"
    assert queried["pairwise_allocation_summary"] == {"n_planned_pairs": 1, "n_collision_pairs": 1,
                                                        "n_duplicate_pairs": 0,
                                                        "relation_outcomes_used_for_planning": False}


def test_pairwise_plan_does_not_depend_on_hidden_relation_answer():
    common = m([p("b", [0, 0, 10, 10], "YES", "YES")],
               [p("r", [1, 0, 11, 10], "YES", "YES", "tile")])
    same = {**common, "identity_edges": [{"a": "b", "b": "r", "relation": "SAME"}]}
    different = {**common, "identity_edges": [{"a": "b", "b": "r", "relation": "DIFFERENT"}]}
    assert run_joint_arbitration(same, pairwise_budget=1)["pairwise_query_plan"] == \
        run_joint_arbitration(different, pairwise_budget=1)["pairwise_query_plan"]


def test_impact_policy_beats_geometry_on_count_changing_collision_priority():
    # r1/r2 are a near-duplicate pair, but b/r1 is the lower-IoU collision
    # whose relation can decide whether a second physical instance is added.
    data = m([p("b", [0, 0, 10, 10], "YES", "YES")],
             [p("r1", [2, 0, 12, 10], "YES", "YES", "t1"),
              p("r2", [2.1, 0, 12.1, 10], "YES", "YES", "t2")],
             [{"a": "b", "b": "r1", "relation": "DIFFERENT"},
              {"a": "r1", "b": "r2", "relation": "SAME"}])
    impact = run_joint_arbitration(data, pairwise_budget=1, pairwise_strategy="impact")
    geometry = run_joint_arbitration(data, pairwise_budget=1, pairwise_strategy="geometry")
    assert impact["pairwise_query_plan"][0]["pair"] == ["b", "r1"]
    assert geometry["pairwise_query_plan"][0]["pair"] == ["r1", "r2"]
