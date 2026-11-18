import json

from countlab.method.t1_batch_evaluate import evaluate_batch


def dump(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def test_offline_batch_evaluator_reports_count_and_identity_without_selection(tmp_path):
    manifest = {
        "schema": "countlab_candidate_pool_v1", "image_id": "i", "query_id": "q",
        "query_text": "object", "baseline_count": 1,
        "baseline": [{"candidate_id": "b", "box_xyxy": [0, 0, 10, 10],
                      "evidence": {"semantic": "YES", "whole_instance": "YES"}}],
        "rescue": [{"candidate_id": "r", "box_xyxy": [1, 0, 11, 10], "view_id": "tile",
                    "evidence": {"semantic": "YES", "whole_instance": "YES"}}],
        "identity_edges": [{"a": "b", "b": "r", "relation": "DIFFERENT"}],
    }
    dump(tmp_path / "manifest.json", manifest)
    dump(tmp_path / "annotation.json", {
        "schema": "countlab_t1_instance_truth_v1", "image_id": "i", "query_id": "q",
        "ground_truth_count": 2, "truth_by_candidate": {"b": "i1", "r": "i2"},
    })
    index = dump(tmp_path / "index.json", {
        "schema": "countlab_t1_manifest_index_v1", "manifest_paths": ["manifest.json"],
    })
    annotations_index = dump(tmp_path / "annotations_index.json", {
        "schema": "countlab_t1_annotation_index_v1",
        "annotations": [{"image_id": "i", "query_id": "q", "path": "annotation.json"}],
    })
    config = dump(tmp_path / "config.json", {
        "schema": "countlab_t1_experiment_config_v1", "arms": ["B6"],
        "verification_budget": None, "pairwise_verification_budget": 1,
        "policy": {"score_threshold": .5, "nms_iou": .5,
                   "identity_min_same_iou": .1, "identity_ambiguity_iou": .3},
        "claims_allowed": False,
    })
    result = evaluate_batch(index_path=index, annotations_index_path=annotations_index, config_path=config)
    assert result["status"] == "OFFLINE_EVALUATED_NO_POLICY_SELECTION"
    assert result["uses_gt_for_inference_or_selection"] is False
    assert result["metrics_by_arm"]["JOINT_T1"]["fallback_mae"] == 0
    assert result["metrics_by_arm"]["JOINT_T1"]["false_merge"] == 0
    assert result["metrics_by_arm"]["JOINT_T1"]["false_split"] == 0
