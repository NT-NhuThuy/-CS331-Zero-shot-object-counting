import csv
import hashlib
import json
from pathlib import Path

import pytest

from countlab.source_identity_audit import collect, write_outputs


def setup_fixture(tmp_path, missing=False, same_bytes=False):

    sources = {
        "fsc": (
            "generated/fsc147_presence_v4/test_pairs_1301.csv",
            [{"file_name": "a.jpg", "image_path": "data/fsc/a.jpg"}],
        ),
        "train": (
            "generated/joint_semantic_counting_v1/lvis_train_semantic_pairs_v1.csv",
            [{"image_id": "1", "file_name": "train.jpg"}],
        ),
        "dev": (
            "generated/joint_semantic_counting_v1/lvis_semantic_dev_pairs_v1.csv",
            [{"image_id": "2", "file_name": "dev.jpg"}],
        ),
        "test": (
            "generated/joint_semantic_counting_v1/lvis_semantic_test_pairs_v1.csv",
            [{"image_id": "3", "file_name": "test.jpg"}],
        ),
    }

    for cohort, (relative, rows) in sources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    fsc = tmp_path / "data/fsc/a.jpg"
    fsc.parent.mkdir(parents=True, exist_ok=True)
    fsc.write_bytes(b"fsc")

    for cohort, filename, folder in [
        ("train", "train.jpg", "train"),
        ("dev", "dev.jpg", "val"),
        ("test", "test.jpg", "val"),
    ]:
        path = (
            tmp_path
            / "data/lvis_v1/images/semantic_v1"
            / folder
            / filename
        )

        path.parent.mkdir(parents=True, exist_ok=True)

        if not (missing and cohort == "dev"):
            content = (
                b"fsc"
                if same_bytes and cohort == "test"
                else cohort.encode()
            )
            path.write_bytes(content)

    return tmp_path


def test_complete_unique_file_audit(tmp_path):
    root = setup_fixture(tmp_path)

    ledger, report = collect(root)

    assert report["n_pairs"] == 4
    assert report["n_unique_declared_source_ids"] == 4
    assert report["n_images_hashed"] == 4
    assert report["n_missing_images"] == 0
    assert report["n_byte_duplicate_groups_across_distinct_source_ids"] == 0
    assert report["file_integrity_status"] == "PASS"
    assert report["g0_decision"] == "HOLD"
    assert all(len(r["image_sha256"]) == 64 for r in ledger)


def test_missing_image_holds(tmp_path):
    root = setup_fixture(tmp_path, missing=True)

    ledger, report = collect(root)

    assert len(ledger) == 4
    assert report["n_missing_images"] == 1
    assert report["file_integrity_status"] == "HOLD"


def test_duplicate_content_detected(tmp_path):
    root = setup_fixture(tmp_path, same_bytes=True)

    _, report = collect(root)

    assert report["n_byte_duplicate_groups_across_distinct_source_ids"] == 1

    ids = set(
        report["byte_duplicate_examples"][0]["source_image_ids"]
    )

    assert ids == {"FSC147:a.jpg", "LVIS:3"}


def test_create_only_outputs(tmp_path):
    root = setup_fixture(tmp_path / "images")

    ledger, report = collect(root)

    out = tmp_path / "out"
    csv_path, json_path = write_outputs(ledger, report, out)

    assert csv_path.exists()
    assert json_path.exists()

    actual = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    saved = json.loads(json_path.read_text())["ledger_sha256"]

    assert saved == actual

    with pytest.raises(FileExistsError):
        write_outputs(ledger, report, out)
