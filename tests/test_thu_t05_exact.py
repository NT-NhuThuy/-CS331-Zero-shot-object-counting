"""C1 T05 contract tests; synthetic evidence, never a GPU or clean-DEV claim."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from countlab.integrations.thu_t05_exact import (
    bind_exact, build_binding_request, canonical_sha, file_sha, main,
)


def fixtures(tmp_path):
    manifest = {
        "schema": "countlab_candidate_pool_v1",
        "image_id": "synthetic-image-1", "query_id": "q-neutral-1", "query_text": "strawberry",
        "baseline_count": 1,
        "baseline": [{"candidate_id": "cgd-pseudo3:one", "view_id": "full",
                      "box_xyxy": [1, 2, 9, 10], "native_query_index": 7,
                      "confidence": 0.9, "evidence": {"semantic": "UNKNOWN", "whole_instance": "UNKNOWN"},
                      "evidence_refs": ["real-id-placeholder-synthetic"]}],
        "rescue": [{"candidate_id": "cgd-pseudo3:two", "view_id": "tile-A",
                    "box_xyxy": [21, 22, 29, 30], "native_query_index": 0,
                    "confidence": 0.2, "evidence_refs": []}],
        "identity_edges": [],
        "metadata": {"image_sha256": "a" * 64, "backend_protocol": "CGD_OFFICIAL_QUERY_ONLY_PSEUDO3_V1"},
    }
    npz = tmp_path / "frozen.npz"
    np.savez(npz, node_graph_id=np.array([101, 202]), raw_query_index=np.array([17, 23]),
             v4_contribution=np.array([1.2, -0.3]), a3_score=np.array([0.8, 0.1]))
    req = build_binding_request(manifest)
    mapped = deepcopy(req["candidate_rows"])
    for row, key in zip(mapped, [(101, 17), (202, 23)]):
        row["node_graph_id"], row["raw_query_index"] = key
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"image_sha256": "a" * 64, "query_text": "strawberry",
                               "protocol_id": "CGD_OFFICIAL_QUERY_ONLY_PSEUDO3_V1"}), encoding="utf-8")
    binding = {
        "schema": "thu_t05_exact_binding_v1",
        "source_cohort_status": "EXPOSED_REFERENCE",
        "source": req["source"],
        "candidate_pool_sha256": req["candidate_pool_sha256"],
        "candidate_identity_sha256": req["candidate_identity_sha256"],
        "feature_npz_sha256": file_sha(npz),
        "source_run_manifest_sha256": file_sha(run),
        "candidate_rows": mapped,
    }
    cross = tmp_path / "crosswalk.json"
    cross.write_text(json.dumps({"schema": "thu_t05_crosswalk_evidence_v1",
                                 "source": req["source"],
                                 "candidate_identity_sha256": req["candidate_identity_sha256"],
                                 "feature_npz_sha256": file_sha(npz),
                                 "candidate_rows": mapped}), encoding="utf-8")
    binding["crosswalk_evidence_sha256"] = file_sha(cross)
    return manifest, binding, npz, cross, run


def test_request_does_not_invent_node_crosswalk(tmp_path):
    manifest, _, _, _, _ = fixtures(tmp_path)
    req = build_binding_request(manifest)
    assert req["binding_status"].startswith("BLOCKED")
    assert req["n_candidates"] == 2
    assert all(r["node_graph_id"] is None and r["raw_query_index"] is None for r in req["candidate_rows"])
    assert set(r["candidate_id"] for r in req["candidate_rows"]) == {"cgd-pseudo3:one", "cgd-pseudo3:two"}
    assert req["candidate_rows"][0]["native_query_index"] == 7
    assert "gt" not in str(req).lower()


def test_bind_exact_real_npz_join_no_routing_or_count(tmp_path):
    manifest, binding, npz, cross, run = fixtures(tmp_path)
    original = deepcopy(manifest)
    joined, report = bind_exact(manifest, binding, npz, cross, run)
    assert manifest == original
    assert joined["baseline"] == manifest["baseline"] and joined["rescue"] == manifest["rescue"]
    assert report["technical_status"] == "PASS_EXACT_JOIN"
    assert report["risk_hint_generated"] is False and report["count_changed"] is False
    assert report["g0_data_decision"] == "HOLD"
    diag = joined["metadata"]["thu_frozen_features"]["per_candidate"]
    assert diag["cgd-pseudo3:one"]["a3_score"] == pytest.approx(0.8)
    assert diag["cgd-pseudo3:two"]["v4_contribution"] == pytest.approx(-0.3)
    assert "risk_hint" not in joined["baseline"][0]
    assert canonical_sha(original) == report["candidate_pool_sha256"]


@pytest.mark.parametrize("change,reason", [
    (lambda b: b["source"].update(image_id="other"), "source/namespace"),
    (lambda b: b.update(candidate_pool_sha256="0" * 64), "pool mismatch"),
    (lambda b: b.update(candidate_identity_sha256="0" * 64), "pool mismatch"),
    (lambda b: b.update(source_cohort_status="FINAL_SEALED"), "exposed diagnostic"),
    (lambda b: b.update(feature_npz_sha256="0" * 64), "feature_npz_sha256"),
    (lambda b: b.update(crosswalk_evidence_sha256="0" * 64), "crosswalk_evidence_sha256"),
    (lambda b: b.update(source_run_manifest_sha256="0" * 64), "source_run_manifest_sha256"),
    (lambda b: b["candidate_rows"][0].update(candidate_id="different"), "candidate ID namespace"),
    (lambda b: b["candidate_rows"][0].update(box_xyxy=[0, 0, 10, 10]), "box_xyxy mismatch"),
    (lambda b: b["candidate_rows"][0].update(native_query_index=999), "native_query_index mismatch"),
    (lambda b: b["candidate_rows"][0].update(node_graph_id=None), "frozen node keys"),
    (lambda b: b["candidate_rows"][1].update(node_graph_id=101, raw_query_index=17), "same frozen node"),
    (lambda b: b["candidate_rows"].pop(), "must cover exactly"),
    (lambda b: b.update(gt_count=3), "GT/label"),
])
def test_modified_binding_rejected(tmp_path, change, reason):
    manifest, binding, npz, cross, run = fixtures(tmp_path)
    change(binding)
    with pytest.raises(ValueError, match=reason):
        bind_exact(manifest, binding, npz, cross, run)


def test_source_pool_modifications_rejected(tmp_path):
    manifest, binding, npz, cross, run = fixtures(tmp_path)
    manifest["query_text"] = "apple"
    with pytest.raises(ValueError, match="source/namespace/pool mismatch"):
        bind_exact(manifest, binding, npz, cross, run)


def test_gt_field_in_pool_rejected(tmp_path):
    manifest, *_ = fixtures(tmp_path)
    manifest["metadata"]["presence_label"] = "PRESENT"
    with pytest.raises(ValueError, match="GT/label"):
        build_binding_request(manifest)


def test_gt_label_encoded_in_query_id_rejected(tmp_path):
    manifest, *_ = fixtures(tmp_path)
    manifest["query_id"] = "POS__2011__strawberry"
    with pytest.raises(ValueError, match="encodes presence label"):
        build_binding_request(manifest)


def test_crosswalk_evidence_semantically_binds_rows(tmp_path):
    manifest, binding, npz, cross, run = fixtures(tmp_path)
    cross_doc = json.loads(cross.read_text())
    cross_doc["candidate_rows"][0]["node_graph_id"] = 555
    cross.write_text(json.dumps(cross_doc))
    binding["crosswalk_evidence_sha256"] = file_sha(cross)
    with pytest.raises(ValueError, match="does not bind to source"):
        bind_exact(manifest, binding, npz, cross, run)


def test_run_manifest_image_scope_rejected_even_with_updated_hash(tmp_path):
    manifest, binding, npz, cross, run = fixtures(tmp_path)
    run.write_text(json.dumps({"image_sha256": "b" * 64, "query_text": "strawberry",
                               "protocol_id": "CGD_OFFICIAL_QUERY_ONLY_PSEUDO3_V1"}))
    binding["source_run_manifest_sha256"] = file_sha(run)
    with pytest.raises(ValueError, match="run manifest image/query/protocol mismatch"):
        bind_exact(manifest, binding, npz, cross, run)


def test_invalid_npz_key_blocked(tmp_path):
    manifest, binding, npz, cross, run = fixtures(tmp_path)
    np.savez(npz, node_graph_id=np.array([101]), raw_query_index=np.array([17]),
             v4_contribution=np.array([1.]), fake_score=np.array([.5]))
    binding["feature_npz_sha256"] = file_sha(npz)
    cross_doc = json.loads(cross.read_text())
    cross_doc["feature_npz_sha256"] = file_sha(npz)
    cross.write_text(json.dumps(cross_doc))
    binding["crosswalk_evidence_sha256"] = file_sha(cross)
    with pytest.raises(ValueError, match="feature keys"):
        bind_exact(manifest, binding, npz, cross, run)


def test_request_cli_is_create_only(tmp_path, monkeypatch, capsys):
    manifest, _, _, _, _ = fixtures(tmp_path)
    pool = tmp_path / "pool.json"; pool.write_text(json.dumps(manifest))
    out = tmp_path / "request_001"
    monkeypatch.setattr("sys.argv", ["countlab-thu-t05", "--pool", str(pool), "--out-dir", str(out)])
    main()
    assert (out / "binding_request.json").exists()
    assert "BLOCKED_CROSSWALK" in capsys.readouterr().out
    with pytest.raises(FileExistsError, match="create-only"):
        main()


def test_request_empty_pool_includes_query_level_source(tmp_path):
    manifest, *_ = fixtures(tmp_path)
    manifest["baseline"] = []
    manifest["rescue"] = []
    manifest["baseline_count"] = 0
    req = build_binding_request(manifest)
    assert req["n_candidates"] == 0
    assert req["binding_status"].startswith("BLOCKED")


def test_bind_cli_produces_immutable_pair(tmp_path, monkeypatch):
    manifest, binding, npz, cross, run = fixtures(tmp_path)
    pool = tmp_path / "pool.json"; pool.write_text(json.dumps(manifest))
    bind = tmp_path / "binding.json"; bind.write_text(json.dumps(binding))
    out = tmp_path / "bound_001"
    monkeypatch.setattr("sys.argv", ["countlab-thu-t05", "--mode", "bind", "--pool", str(pool),
                                         "--binding", str(bind), "--feature-npz", str(npz),
                                         "--crosswalk-evidence", str(cross), "--run-manifest", str(run),
                                         "--out-dir", str(out)])
    main()
    frozen = json.loads((out / "frozen_candidates_t05.json").read_text())
    report = json.loads((out / "t05_binding_report.json").read_text())
    assert report["technical_status"] == "PASS_EXACT_JOIN"
    assert frozen["baseline"] == manifest["baseline"]
    assert len(frozen["metadata"]["thu_frozen_features"]["per_candidate"]) == 2


def test_duplicate_json_key_rejected(tmp_path, monkeypatch):
    pool = tmp_path / "pool.json"; pool.write_text('{"schema":"countlab_candidate_pool_v1","schema":"bad"}')
    monkeypatch.setattr("sys.argv", ["countlab-thu-t05", "--pool", str(pool),
                                         "--out-dir", str(tmp_path / "x")])
    with pytest.raises(ValueError, match="duplicate JSON key"):
        main()


def test_source_run_protocol_mismatch_rejected(tmp_path):
    manifest, binding, npz, cross, run = fixtures(tmp_path)
    run.write_text(json.dumps({"image_sha256": "a" * 64, "query_text": "strawberry", "protocol_id": "OTHER"}))
    binding["source_run_manifest_sha256"] = file_sha(run)
    with pytest.raises(ValueError, match="run manifest image/query/protocol mismatch"):
        bind_exact(manifest, binding, npz, cross, run)
