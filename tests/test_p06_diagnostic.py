"""Synthetic tests for P06 GT-free diagnostic features."""

from copy import deepcopy

import numpy as np
import pytest

from countlab.integrations.p06_diagnostic import (
    extract_diagnostic_features,
)


def make_fixture(tmp_path):
    manifest = {
        "schema": "countlab_candidate_pool_v1",
        "image_id": "synthetic-image-001",
        "query_id": "synthetic-query-001",
        "query_text": "chair",
        "baseline_count": 1,
        "baseline": [
            {
                "candidate_id": "b1",
                "view_id": "full",
                "box_xyxy": [0, 0, 10, 20],
                "confidence": 0.8,
                "evidence": {
                    "semantic": "UNKNOWN",
                    "whole_instance": "UNKNOWN",
                },
                "evidence_refs": [],
            }
        ],
        "rescue": [
            {
                "candidate_id": "r1",
                "view_id": "g2_r0_c0",
                "box_xyxy": [20, 10, 26, 14],
                "confidence": 0.3,
                "evidence": {
                    "semantic": "UNKNOWN",
                    "whole_instance": "UNKNOWN",
                },
                "evidence_refs": [],
            }
        ],
        "identity_edges": [],
        "metadata": {"source": "synthetic_fixture"},
    }

    mapping = {
        "b1": {"node_graph_id": 9, "raw_query_index": 3},
        "r1": {"node_graph_id": 10, "raw_query_index": 4},
    }

    npz = tmp_path / "synthetic_features.npz"

    np.savez(
        npz,
        node_graph_id=np.array([9, 10]),
        raw_query_index=np.array([3, 4]),
        v4_contribution=np.array([1.02, -0.2]),
        a3_score=np.array([0.7, 0.25]),
    )

    return manifest, npz, mapping


def test_exact_key_join(tmp_path):
    manifest, npz, mapping = make_fixture(tmp_path)

    result = extract_diagnostic_features(
        manifest, npz, mapping
    )

    assert result["schema"] == "countlab_p06_diagnostic_features_v1"
    assert result["n_candidates"] == 2
    assert result["g0_decision"] == "HOLD"

    rows = {
        r["candidate_id"]: r
        for r in result["candidates"]
    }

    assert set(rows) == {"b1", "r1"}

    assert rows["b1"]["role"] == "baseline"
    assert rows["r1"]["role"] == "rescue"

    assert rows["b1"]["node_graph_id"] == 9
    assert rows["b1"]["raw_query_index"] == 3
    assert rows["b1"]["v4_contribution"] == pytest.approx(1.02)
    assert rows["b1"]["a3_score"] == pytest.approx(0.7)

    assert rows["r1"]["node_graph_id"] == 10
    assert rows["r1"]["a3_score"] == pytest.approx(0.25)

    assert rows["b1"]["area_px2"] == pytest.approx(200)
    assert rows["b1"]["aspect_ratio"] == pytest.approx(0.5)

    assert rows["b1"]["confidence_uncertainty_baseline"] == pytest.approx(0.2)

    assert result["risk_hint_generated"] is False
    assert result["calibration_used"] is False
    assert result["eligible_for_training"] is False


def test_original_manifest_unchanged(tmp_path):
    manifest, npz, mapping = make_fixture(tmp_path)

    before_manifest = deepcopy(manifest)
    before_mapping = deepcopy(mapping)

    extract_diagnostic_features(manifest, npz, mapping)

    assert manifest == before_manifest
    assert mapping == before_mapping
    assert "risk_hint" not in manifest["baseline"][0]


def test_deterministic_output(tmp_path):
    manifest, npz, mapping = make_fixture(tmp_path)

    a = extract_diagnostic_features(
        manifest, npz, mapping
    )
    b = extract_diagnostic_features(
        manifest, npz, mapping
    )

    assert a == b


def test_missing_candidate_mapping_rejected(tmp_path):
    manifest, npz, mapping = make_fixture(tmp_path)

    mapping.pop("r1")

    with pytest.raises(ValueError, match="candidate_map"):
        extract_diagnostic_features(
            manifest, npz, mapping
        )


def test_duplicate_node_mapping_rejected(tmp_path):
    manifest, npz, mapping = make_fixture(tmp_path)

    mapping["r1"] = {
        "node_graph_id": 9,
        "raw_query_index": 3,
    }

    with pytest.raises(ValueError, match="same frozen node"):
        extract_diagnostic_features(
            manifest, npz, mapping
        )


def test_missing_npz_feature_rejected(tmp_path):
    manifest, _, mapping = make_fixture(tmp_path)

    bad_npz = tmp_path / "bad_features.npz"

    np.savez(
        bad_npz,
        node_graph_id=np.array([9, 10]),
        raw_query_index=np.array([3, 4]),
        v4_contribution=np.array([1.0, 0.5]),
    )

    with pytest.raises(ValueError, match="feature keys"):
        extract_diagnostic_features(
            manifest, bad_npz, mapping
        )


def test_gt_leakage_rejected(tmp_path):
    manifest, npz, mapping = make_fixture(tmp_path)

    manifest["metadata"]["gt_count"] = 2

    with pytest.raises(ValueError, match="GT/label"):
        extract_diagnostic_features(
            manifest, npz, mapping
        )


def test_invalid_original_box_rejected(tmp_path):
    manifest, npz, mapping = make_fixture(tmp_path)

    manifest["baseline"][0]["box_xyxy"] = [10, 10, 2, 2]

    with pytest.raises(ValueError, match="invalid original-coordinate box"):
        extract_diagnostic_features(
            manifest, npz, mapping
        )


def test_unexposed_claim_rejected(tmp_path):
    manifest, npz, mapping = make_fixture(tmp_path)

    with pytest.raises(ValueError, match="exposure"):
        extract_diagnostic_features(
            manifest,
            npz,
            mapping,
            exposure_status="UNEXPOSED",
        )
