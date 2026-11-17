"""GT-free tests for P01-D; no private dataset access."""
import json
from pathlib import Path

from countlab.integrations.historical_experiment_lineage_audit import (
    rel, v14_rows,
)


def test_frozen_manifest_is_conservative():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "manifests/historical_experiment_lineage_audit_v1.json").read_text())
    assert len(manifest["experiments"]) == 13
    assert len(manifest["missing_provenance_entries"]) == 54
    assert manifest["verified_entries"] == []
    assert manifest["uses_gt"] is False
    rows = manifest["missing_provenance_entries"]
    assert len({r["query_id"] for r in rows}) == 54
    assert all(r["experiment_id"] == "V14" for r in rows)
    assert all(r["image_sha256"] is None for r in rows)
    assert all(r["source_group"] is None for r in rows)
    assert all(not p.startswith("/") for r in rows for p in r["artifact_path"])


def test_recover_v14_synthetic(tmp_path):
    private = tmp_path / "UNICOUNT_PRIVATE"
    orig = private / "outputs/v14_original_cpu_repro_20261114/bidirectional_output.json"
    man = private / "outputs/v14_unicount_import_20261114/manifests/q_test.json"
    res = private / "outputs/v14_unicount_import_20261114/results/q_test.json"
    for p in (orig, man, res):
        p.parent.mkdir(parents=True, exist_ok=True)
    orig.write_text(json.dumps({"queries": {"1": {
        "accepted_instances": [{"source_image_id": 7, "query_caption": "bird"}],
        "rejected_regions": [], "unknown_regions": [],
    }}, "input_hashes": {}}))
    man.write_text(json.dumps({"query_id": "1", "image_id": 7, "query_text": "bird"}))
    res.write_text(json.dumps({"query_id": "1"}))
    rows, summary = v14_rows(private=private, workspace=tmp_path)
    assert len(rows) == 1
    assert rows[0]["image_sha256"] is None
    assert rows[0]["source_group"] is None
    assert rows[0]["image_id"] == "7"
    assert rows[0]["evidence"]["per_query_result_present"] is True
    assert summary["evidence_status"] == "PARTIAL_RECOVERED"


def test_outside_workspace_path_rejected(tmp_path):
    outside = tmp_path.parent / "outside-private-file.json"
    import pytest
    with pytest.raises(ValueError, match="outside workspace"):
        rel(outside, tmp_path)
