import pytest

from countlab.method.baselines import run_baseline_arm


def proposal(candidate_id, box, *, score=.5, semantic="UNKNOWN", whole="UNKNOWN", view="full", risk=None):
    item = {"candidate_id": candidate_id, "box_xyxy": box, "view_id": view, "confidence": score,
            "evidence": {"semantic": semantic, "whole_instance": whole}}
    if risk is not None:
        item["risk_hint"] = risk
    return item


def manifest(baseline, rescue=(), edges=()):
    return {"schema": "countlab_candidate_pool_v1", "image_id": "image", "query_id": "query",
            "query_text": "object", "baseline": list(baseline), "baseline_count": len(baseline),
            "rescue": list(rescue), "identity_edges": list(edges)}


def test_b0_preserves_frozen_baseline_without_postprocessing():
    data = manifest([proposal("a", [0, 0, 10, 10], semantic="NO", whole="NO")],
                    [proposal("r", [20, 0, 30, 10], semantic="YES", whole="YES")])
    output = run_baseline_arm(data, "B0")
    assert output["count"] == 1
    assert output["accepted_candidate_ids"] == ["a"]


def test_unary_controls_do_not_turn_unknown_into_rejection():
    data = manifest([proposal("no", [0, 0, 10, 10], semantic="NO", whole="YES"),
                     proposal("unknown", [20, 0, 30, 10])])
    assert run_baseline_arm(data, "B2")["accepted_candidate_ids"] == ["unknown"]
    assert run_baseline_arm(data, "B3")["accepted_candidate_ids"] == ["no", "unknown"]
    assert run_baseline_arm(data, "B4")["accepted_candidate_ids"] == ["no", "unknown"]


def test_complete_link_blocks_unknown_overlap_while_naive_union_merges_same_chain():
    rescue = [proposal("a", [0, 0, 10, 10], semantic="YES", whole="YES", view="tile-a"),
              proposal("b", [1, 0, 11, 10], semantic="YES", whole="YES", view="tile-b"),
              proposal("c", [2, 0, 12, 10], semantic="YES", whole="YES", view="tile-c")]
    edges = [{"a": "a", "b": "b", "relation": "SAME"}, {"a": "b", "b": "c", "relation": "SAME"}]
    data = manifest([], rescue, edges)
    assert run_baseline_arm(data, "B7")["count"] == 1
    strict = run_baseline_arm(data, "B8")
    assert strict["count"] == 0
    assert strict["unresolved_candidate_ids"] == ["a", "b", "c"]


def test_b9_replay_masks_unselected_evidence_and_is_deterministic():
    data = manifest([proposal("low", [0, 0, 10, 10], semantic="NO", whole="NO", risk=.1),
                     proposal("high", [20, 0, 30, 10], semantic="NO", whole="NO", risk=.9)])
    one = run_baseline_arm(data, "B9", budget=1)
    assert one["selected_candidate_ids"] == ["high"]
    assert one["accepted_candidate_ids"] == ["low"]
    assert one == run_baseline_arm(data, "B9", budget=1)


def test_b4_budgeted_and_b5_all_verify_are_distinct_controls():
    data = manifest([proposal("low", [0, 0, 10, 10], semantic="NO", whole="NO", risk=.1),
                     proposal("high", [20, 0, 30, 10], semantic="NO", whole="NO", risk=.9)])
    budgeted = run_baseline_arm(data, "B4", budget=1)
    fully_observed = run_baseline_arm(data, "B5", budget=1)
    assert budgeted["trace"]["evidence_mode"] == "budget_masked"
    assert fully_observed["trace"]["evidence_mode"] == "all_precomputed"
    assert budgeted["accepted_candidate_ids"] == ["low"]
    assert fully_observed["accepted_candidate_ids"] == []


def test_invalid_arm_and_policy_are_rejected():
    data = manifest([])
    with pytest.raises(ValueError, match="unknown baseline arm"):
        run_baseline_arm(data, "B10")
    with pytest.raises(ValueError, match="EXACTLY"):
        run_baseline_arm(data, "B1", policy={"nms_iou": .5})
