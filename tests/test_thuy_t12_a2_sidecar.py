"""A2 integration tests against the original GT-blind T12 validation path."""

import copy
import pytest

from countlab.method.t12_a2_sidecar import evaluate_a2_sidecar


def candidate(cid, box):
    return {
        "candidate_id": cid,
        "view_id": "full",
        "box_xyxy": box,
        "evidence": {
            "semantic": "UNKNOWN",
            "whole_instance": "UNKNOWN",
        },
        "evidence_refs": [],
    }


def record(baseline, events):
    return {
        "manifest": {
            "schema": "countlab_candidate_pool_v1",
            "image_id": "fixture-image",
            "query_id": "fixture-query",
            "query_text": "target",
            "baseline_count": len(baseline),
            "baseline": baseline,
            "rescue": [],
            "identity_edges": [],
        },
        "observed_events": events,
        "source_group_id": "fixture-image",
        "cohort_role": "synthetic",
        "split_protocol_id": "synthetic",
        "frozen_pool_run_id": "synthetic",
        "pool_scope": "unverified",
    }


def test_nms_is_not_identity_proof():
    row = record([
        candidate("a", [0, 0, 10, 10]),
        candidate("b", [0, 0, 10, 10]),
    ], [])

    original = copy.deepcopy(row)
    out = evaluate_a2_sidecar(row, 1, "diagnostic")

    assert out["a0_frozen_count"] == 2
    assert out["a1_rejection_only_count"] == 2
    assert out["A2_NMS"]["forced_count"] == 1
    assert out["A2_COMPLETE_LINK"]["forced_count"] == 2
    assert out["gt_used"] is False
    assert row == original


def test_complete_link_requires_same_not_iou():
    row = record([
        candidate("a", [0, 0, 10, 10]),
        candidate("b", [20, 0, 30, 10]),
    ], [{
        "op": "PAIR_IDENTITY",
        "a": "a",
        "b": "b",
        "relation": "SAME",
        "evidence_ref": "qwen:pair:1",
    }])

    out = evaluate_a2_sidecar(row, 1, "diagnostic")

    assert out["A2_NMS"]["forced_count"] == 2
    assert out["A2_COMPLETE_LINK"]["forced_count"] == 1
    assert out["A2_COMPLETE_LINK"]["details"]["direct_same_evidence_refs"] == [
        "qwen:pair:1"
    ]


def test_a1_rejected_ids_derived_from_original_exporter():
    row = record([
        candidate("a", [0, 0, 10, 10]),
        candidate("b", [20, 0, 30, 10]),
    ], [
        {
            "op": "REGION_SEMANTIC",
            "candidate_id": "a",
            "verdict": "NO",
            "evidence_ref": "qwen:s:1",
        },
        {
            "op": "REGION_WHOLE",
            "candidate_id": "a",
            "verdict": "NO",
            "evidence_ref": "qwen:w:1",
        },
    ])

    out = evaluate_a2_sidecar(row, 1, "diagnostic")

    assert out["a1_rejection_only_count"] == 1
    assert out["a1_rejected_candidate_ids"] == ["a"]
    assert out["A2_NMS"]["forced_count"] == 1
    assert out["A2_COMPLETE_LINK"]["forced_count"] == 1


def test_gt_leak_is_rejected():
    row = record([candidate("a", [0, 0, 10, 10])], [])
    row["manifest"]["gt_count"] = 1

    with pytest.raises(ValueError, match="GT-bearing"):
        evaluate_a2_sidecar(row, 1, "diagnostic")
