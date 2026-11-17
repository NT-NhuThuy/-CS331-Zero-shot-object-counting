from countlab.split_policy import audit_split_proposal


def row(split="EXPOSED_REFERENCE", source="img1", group="g1",
        digest="a" * 64, query="q1", exposure="EXPOSED",
        lineage="HISTORICAL_EXPOSED"):
    return {
        "dataset": "synthetic",
        "source_image_id": source,
        "source_group_id": group,
        "source_image_sha256": digest,
        "query_id": query,
        "proposed_split": split,
        "exposure_status": exposure,
        "lineage_status": lineage,
    }


def clean(split, source, group, digest, query):
    r = row(
        split, source, group, digest, query,
        "UNEXPOSED", "VERIFIED"
    )
    r["provenance_evidence_sha256"] = "b" * 64
    return r


def test_historical_only_passes_but_g0_hold():
    result = audit_split_proposal([row()])
    assert result["status"] == "PASS"
    assert result["g0_decision"] == "HOLD"


def test_source_group_overlap_fails():
    rows = [
        clean("TRAIN", "i1", "g", "a" * 64, "q1"),
        clean("DEV", "i2", "g", "c" * 64, "q2"),
    ]
    assert audit_split_proposal(rows)["status"] == "FAIL"


def test_duplicate_bytes_across_groups_fail():
    rows = [
        clean("TRAIN", "i1", "g1", "a" * 64, "q1"),
        clean("CAL", "i2", "g2", "a" * 64, "q2"),
    ]
    assert audit_split_proposal(rows)["status"] == "FAIL"


def test_exposed_source_cannot_enter_cal():
    rows = [
        row(),
        clean("CAL", "i2", "g1", "a" * 64, "q2"),
    ]
    assert audit_split_proposal(rows)["status"] == "FAIL"


def test_unknown_cannot_enter_final_sealed():
    r = row(
        "FINAL_SEALED", "i2", "g2",
        "c" * 64, "q2", "UNKNOWN", "UNKNOWN"
    )
    assert audit_split_proposal([r])["status"] == "FAIL"


def test_distinct_verified_metadata_passes_but_g0_hold():
    rows = [
        clean("TRAIN", "i1", "g1", "a" * 64, "q1"),
        clean("CAL", "i2", "g2", "c" * 64, "q2"),
    ]
    result = audit_split_proposal(rows)
    assert result["status"] == "PASS"
    assert result["g0_decision"] == "HOLD"


def test_provenance_evidence_required():
    r = clean("DEV", "i1", "g1", "a" * 64, "q1")
    r.pop("provenance_evidence_sha256")
    assert audit_split_proposal([r])["status"] == "FAIL"
