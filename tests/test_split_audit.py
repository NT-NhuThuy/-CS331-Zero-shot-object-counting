import pytest
from countlab.split_audit import audit_rows


def row(source, query, split, exposure="UNEXPOSED"):
    return {
        "dataset": "TOY",
        "source_image_id": source,
        "query_id": query,
        "split": split,
        "gt_available": True,
        "exposure_status": exposure,
        "artifact_sha256": "a" * 64,
        "protocol_id": "P03_TOY",
    }


def test_source_disjoint_pass():
    rows = [
        row("image1", "q1", "TRAIN"),
        row("image1", "q2", "TRAIN"),
        row("image2", "q3", "DEV"),
        row("image3", "q4", "CAL"),
        row("image4", "q5", "FINAL_SEALED"),
    ]
    assert audit_rows(rows)["status"] == "PASS"


def test_same_image_in_train_and_dev_fails():
    rows = [
        row("image1", "q1", "TRAIN"),
        row("image1", "q2", "DEV"),
    ]
    result = audit_rows(rows)
    assert result["status"] == "FAIL"
    assert any("source-image leakage" in x for x in result["issues"])


def test_exposed_data_cannot_enter_cal():
    rows = [row("image1", "q1", "CAL", "EXPOSED")]
    assert audit_rows(rows)["status"] == "FAIL"


def test_exposed_reference_is_allowed():
    rows = [
        row("old_image", "q1", "EXPOSED_REFERENCE", "EXPOSED"),
        row("new_image", "q2", "TRAIN"),
    ]
    assert audit_rows(rows)["status"] == "PASS"


def test_duplicate_query_rejected():
    rows = [
        row("image1", "q1", "TRAIN"),
        row("image2", "q1", "TRAIN"),
    ]
    assert audit_rows(rows)["status"] == "FAIL"


def test_ground_truth_value_rejected():
    x = row("image1", "q1", "TRAIN")
    x["gt_count"] = 10
    with pytest.raises(ValueError, match="GT values forbidden"):
        audit_rows([x])


def test_identical_image_hash_across_splits_fails():
    a = row("datasetA_img", "q1", "TRAIN")
    b = row("datasetB_img", "q2", "DEV")
    a["source_image_sha256"] = "b" * 64
    b["source_image_sha256"] = "b" * 64

    result = audit_rows([a, b])
    assert result["status"] == "FAIL"
    assert any("image-hash leakage" in x for x in result["issues"])
