import json

import pytest

from countlab.method.t1_batch_replay import replay_batch


def dump(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def manifest(image_id, relation):
    return {
        "schema": "countlab_candidate_pool_v1", "image_id": image_id, "query_id": "q", "query_text": "object",
        "baseline_count": 1,
        "baseline": [{"candidate_id": "b", "box_xyxy": [0, 0, 10, 10],
                      "evidence": {"semantic": "YES", "whole_instance": "YES"}}],
        "rescue": [{"candidate_id": "r", "box_xyxy": [1, 0, 11, 10], "view_id": "tile",
                    "evidence": {"semantic": "YES", "whole_instance": "YES"}}],
        "identity_edges": [{"a": "b", "b": "r", "relation": relation}],
    }


def test_batch_replay_is_create_only_hashed_and_gt_blind(tmp_path):
    dump(tmp_path / "one.json", manifest("one", "DIFFERENT"))
    dump(tmp_path / "two.json", manifest("two", "SAME"))
    index = dump(tmp_path / "index.json", {
        "schema": "countlab_t1_manifest_index_v1", "manifest_paths": ["one.json", "two.json"],
    })
    config = dump(tmp_path / "config.json", {
        "schema": "countlab_t1_experiment_config_v1", "experiment_id": "test-batch",
        "arms": ["B0", "B6"], "verification_budget": None, "pairwise_verification_budget": 1,
        "policy": {"score_threshold": .5, "nms_iou": .5,
                   "identity_min_same_iou": .1, "identity_ambiguity_iou": .3}, "claims_allowed": False,
    })
    lock = replay_batch(index_path=index, config_path=config, out_dir=tmp_path / "run")
    artifact = json.loads((tmp_path / "run" / "t1_batch_replay.json").read_text())
    assert lock["uses_gt_for_replay"] is False
    assert artifact["status"] == "FROZEN_EVIDENCE_BATCH_REPLAY_NO_CLAIM"
    assert artifact["summary"]["summary"]["n_changed_pairs"] == 1
    assert len(artifact["manifest_records"]) == 2
    with pytest.raises(FileExistsError):
        replay_batch(index_path=index, config_path=config, out_dir=tmp_path / "run")
