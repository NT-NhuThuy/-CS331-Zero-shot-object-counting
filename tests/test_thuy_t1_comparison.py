import pytest

from countlab.method.t1_comparison import run_matched_t1_batch, run_matched_t1_comparison


def p(cid, box, semantic="YES", whole="YES", view="full"):
    return {"candidate_id": cid, "box_xyxy": box, "view_id": view,
            "confidence": .7, "evidence": {"semantic": semantic, "whole_instance": whole}}


def manifest():
    return {"schema": "countlab_candidate_pool_v1", "image_id": "i", "query_id": "q", "query_text": "x",
            "baseline": [p("b", [0, 0, 10, 10])], "baseline_count": 1,
            "rescue": [p("r", [1, 0, 11, 10], view="tile")],
            "identity_edges": [{"a": "b", "b": "r", "relation": "DIFFERENT"}]}


def test_comparison_locks_one_input_and_exposes_joint_difference():
    output = run_matched_t1_comparison(manifest(), arms=("B0", "B4", "B8"), pairwise_budget=1)
    assert set(output["arms"]) == {"B0", "B4", "B8"}
    assert all(row["input_sha256"] == output["input_sha256"] for row in output["arms"].values())
    assert output["arms"]["B0"]["count"] == 1
    assert output["experimental_joint"]["count"] == 2
    assert output["experimental_joint"]["joint_policy"]["pairwise_strategy"] == "impact"
    assert output["pairwise_geometry_control"]["joint_policy"]["pairwise_strategy"] == "geometry"
    assert output["pairwise_risk_control"]["joint_policy"]["pairwise_strategy"] == "risk"


def test_comparison_rejects_unregistered_or_empty_arm_selection():
    with pytest.raises(ValueError, match="registered"):
        run_matched_t1_comparison(manifest(), arms=())
    with pytest.raises(ValueError, match="registered"):
        run_matched_t1_comparison(manifest(), arms=("B99",))


def test_batch_summary_reports_decision_delta_without_ground_truth_claim():
    different = manifest()
    same = manifest()
    same["image_id"] = "i2"
    same["identity_edges"][0]["relation"] = "SAME"
    result = run_matched_t1_batch([different, same], pairwise_budget=1, arms=("B0", "B4"))
    assert result["status"] == "FROZEN_EVIDENCE_DECISION_SUMMARY_NO_CLAIM"
    assert result["summary"]["joint_minus_b6_count_sum"] == 1
    assert result["summary"]["n_changed_pairs"] == 1
    assert result["summary"]["joint_action_counts"] == {"ADD": 1, "COALESCE": 1}
    assert "not evidence of improvement" in result["limitations"][1]
