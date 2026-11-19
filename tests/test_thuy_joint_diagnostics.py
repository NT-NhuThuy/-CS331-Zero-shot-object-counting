from countlab.method.joint_diagnostics import diagnose_joint_decision


def p(cid, box, semantic="YES", whole="YES", view="full"):
    return {"candidate_id": cid, "box_xyxy": box, "view_id": view,
            "evidence": {"semantic": semantic, "whole_instance": whole}}


def m():
    return {"schema": "countlab_candidate_pool_v1", "image_id": "i", "query_id": "q", "query_text": "x",
            "baseline": [p("b", [0, 0, 10, 10])], "baseline_count": 1,
            "rescue": [p("r", [1, 0, 11, 10], view="tile")],
            "identity_edges": [{"a": "b", "b": "r", "relation": "DIFFERENT"}]}


def test_diagnostic_exposes_joint_change_without_claiming_improvement():
    out = diagnose_joint_decision(m())
    assert out["v1_evaluation_fallback_count"] == 1
    assert out["joint_evaluation_fallback_count"] == 2
    assert out["fallback_count_delta_joint_minus_v1"] == 1
    assert out["v1_matched_rescue_ids"] == ["r"]
    assert out["joint_action_counts"] == {"ADD": 1}
    assert "not an improvement claim" in out["interpretation"]


def test_diagnostic_reports_unknown_collision_as_unresolved_not_addition():
    data = m()
    data["identity_edges"] = []
    out = diagnose_joint_decision(data, abstain_on_unresolved=True)
    assert out["joint_action_counts"] == {"UNRESOLVED": 1}
    assert out["joint_unresolved_count"] == 1
    assert out["joint_query_abstained"]
