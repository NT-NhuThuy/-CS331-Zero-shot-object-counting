import json

import pytest

from countlab.method.t1_replay import evaluate_count_annotation, evaluate_instance_annotations, replay


def dump(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def test_offline_identity_evaluator_known_answer_and_unscored():
    result = evaluate_instance_annotations(
        {"image_id": "i", "query_id": "q", "predicted_components": [["a", "b"], ["c"], ["u"]]},
        {"schema": "countlab_t1_instance_truth_v1", "image_id": "i", "query_id": "q",
         "truth_by_candidate": {"a": "x", "b": "y", "c": "x"}},
    )
    assert result["pair_false_merge"] == 1
    assert result["pair_false_split"] == 1
    assert result["pair_precision"] == 0
    assert result["pair_recall"] == 0
    assert result["unscored_candidate_ids"] == ["u"]


def test_offline_evaluator_requires_exact_pool_and_unique_partition():
    ann = {"schema": "countlab_t1_instance_truth_v1", "image_id": "other", "query_id": "q",
           "truth_by_candidate": {}}
    with pytest.raises(ValueError, match="identity mismatch"):
        evaluate_instance_annotations({"image_id": "i", "query_id": "q", "predicted_components": []}, ann)
    ann["image_id"] = "i"
    with pytest.raises(ValueError, match="multiple components"):
        evaluate_instance_annotations({"image_id": "i", "query_id": "q", "predicted_components": [["a"], ["a"]]}, ann)


def test_offline_evaluator_counts_reviewed_candidates_omitted_by_selection():
    result = evaluate_instance_annotations(
        {"image_id": "i", "query_id": "q", "candidate_universe_ids": ["a", "b", "c"],
         "predicted_components": [["a"]]},
        {"schema": "countlab_t1_instance_truth_v1", "image_id": "i", "query_id": "q",
         "truth_by_candidate": {"a": "x", "b": "x", "c": "y"}},
    )
    assert result["omitted_reviewed_candidate_ids"] == ["b", "c"]
    assert result["pair_missed_same"] == 1
    assert result["instance_candidate_coverage"] == .5


def test_create_only_replay_locks_all_arms_and_never_accepts_claim_flag(tmp_path):
    manifest = dump(tmp_path / "manifest.json", {
        "schema": "countlab_candidate_pool_v1", "image_id": "i", "query_id": "q", "query_text": "object",
        "baseline_count": 1,
        "baseline": [{"candidate_id": "b", "box_xyxy": [0, 0, 10, 10], "confidence": .8,
                      "evidence": {"semantic": "YES", "whole_instance": "YES"}}],
        "rescue": [], "identity_edges": [],
    })
    config = dump(tmp_path / "config.json", {
        "schema": "countlab_t1_experiment_config_v1", "experiment_id": "test",
        "arms": [f"B{i}" for i in range(10)], "include_experimental_joint": True,
        "verification_budget": 0, "policy": {"score_threshold": .5, "nms_iou": .5,
            "identity_min_same_iou": .1, "identity_ambiguity_iou": .3}, "claims_allowed": False,
    })
    out_dir = tmp_path / "run"
    lock = replay(manifest_path=manifest, config_path=config, out_dir=out_dir)
    result = json.loads((out_dir / "t1_replay.json").read_text())
    assert set(result["replay"]["arms"]) == {f"B{i}" for i in range(10)}
    assert result["replay"]["experimental_joint"]["count"] == 1
    assert lock["uses_gt_for_replay"] is False
    with pytest.raises(FileExistsError):
        replay(manifest_path=manifest, config_path=config, out_dir=out_dir)


def test_replay_joins_instance_truth_only_after_all_gt_blind_arms_run(tmp_path):
    manifest = dump(tmp_path / "manifest.json", {
        "schema": "countlab_candidate_pool_v1", "image_id": "i", "query_id": "q", "query_text": "object",
        "baseline_count": 1,
        "baseline": [{"candidate_id": "b", "box_xyxy": [0, 0, 10, 10],
                      "evidence": {"semantic": "YES", "whole_instance": "YES"}}],
        "rescue": [{"candidate_id": "r", "box_xyxy": [1, 0, 11, 10], "view_id": "tile",
                    "evidence": {"semantic": "YES", "whole_instance": "YES"}}],
        "identity_edges": [{"a": "b", "b": "r", "relation": "DIFFERENT"}],
    })
    config = dump(tmp_path / "config.json", {
        "schema": "countlab_t1_experiment_config_v1", "experiment_id": "offline-eval",
        "arms": ["B0", "B6"], "verification_budget": None, "pairwise_verification_budget": 1,
        "policy": {"score_threshold": .5, "nms_iou": .5,
                   "identity_min_same_iou": .1, "identity_ambiguity_iou": .3}, "claims_allowed": False,
    })
    annotations = dump(tmp_path / "annotations.json", {
        "schema": "countlab_t1_instance_truth_v1", "image_id": "i", "query_id": "q",
        "ground_truth_count": 2,
        "truth_by_candidate": {"b": "instance-1", "r": "instance-2"},
    })
    replay(manifest_path=manifest, config_path=config, out_dir=tmp_path / "run", annotation_path=annotations)
    result = json.loads((tmp_path / "run" / "t1_replay.json").read_text())
    assert result["status"] == "OFFLINE_EVALUATED_NO_POLICY_SELECTION"
    assert result["uses_gt_for_replay"] is False
    assert result["offline_evaluation"]["uses_gt_for_inference_or_selection"] is False
    assert result["offline_evaluation"]["joint_metrics"]["pair_false_merge"] == 0
    assert result["offline_evaluation"]["joint_count_metrics"]["fallback_abs_error"] == 0


def test_offline_count_metrics_keep_abstention_and_fallback_separate():
    annotation = {"schema": "countlab_t1_instance_truth_v1", "image_id": "i", "query_id": "q",
                  "ground_truth_count": 2, "truth_by_candidate": {}}
    metrics = evaluate_count_annotation(
        {"image_id": "i", "query_id": "q", "count": None,
         "evaluation_fallback_count": 1, "query_abstained": True}, annotation)
    assert metrics["predicted_count"] is None
    assert metrics["public_abs_error"] is None
    assert metrics["fallback_abs_error"] == 1
    assert metrics["fallback_exact"] is False
