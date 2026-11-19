"""T02 audit tests; synthetic fixtures only (not research results)."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import pytest
from countlab.thu_t02 import build, file_sha256


def csv_file(path: Path, data: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader(); w.writerows(data)
    return path


def fixtures(tmp_path: Path):
    fsc = [
        {"pair_id":"P1", "file_name":"a.jpg", "image_path":"/images/a.jpg", "query":"cat",
         "pair_type":"VERIFIED_PRESENT_FSC147","presence_label":"1", "count_target":"2",
         "num_exemplars":"3", "annotation_source":"FSC147_GT"},
        {"pair_id":"P2", "file_name":"b.jpg", "image_path":"/images/b.jpg", "query":"dog",
         "pair_type":"VERIFIED_PRESENT_FSC147","presence_label":"1", "count_target":"1",
         "num_exemplars":"3", "annotation_source":"FSC147_GT"},
        {"pair_id":"A1", "file_name":"a.jpg", "image_path":"/images/a.jpg", "query":"sheep",
         "pair_type":"VERIFIED_ABSENT_HARD_NEGATIVE","presence_label":"0", "count_target":"0",
         "num_exemplars":"0", "annotation_source":"historical_manual"},
        {"pair_id":"A2", "file_name":"c.jpg", "image_path":"/images/c.jpg", "query":"horse",
         "pair_type":"VERIFIED_ABSENT_HARD_NEGATIVE","presence_label":"0", "count_target":"0",
         "num_exemplars":"0", "annotation_source":"historical_manual"},
    ]
    loc111 = [{"pair_id":r["pair_id"], "image_name":r["file_name"],
               "query_target":r["query"], "presence_label":"0", "count_target":"0",
               "assistant_self_check":"PASS_ABSENT", "final_decision":"KEEP_NEGATIVE"}
              for r in fsc if r["presence_label"] == "0"]
    fresh160 = [
        {"candidate_id":"X1", "image_name":"x.jpg", "target":"sofa", "presence_label":"ABSENT", "label_source":"single_human_reviewer"},
        {"candidate_id":"X2", "image_name":"y.jpg", "target":"chair", "presence_label":"PRESENT", "label_source":"single_human_reviewer"},
    ]
    return (csv_file(tmp_path/"fsc.csv",fsc),csv_file(tmp_path/"loc111.csv",loc111),
            csv_file(tmp_path/"fresh160.csv",fresh160),csv_file(tmp_path/"fresh126.csv",fresh160[:1]))


def test_build_exposed_roster_no_gt_and_no_duplicate_sources(tmp_path):
    paths = fixtures(tmp_path)
    out = tmp_path/"out"
    report = build(*paths, out, pilot_each=2)
    assert report["technical_status"] == "PASS"
    assert report["g0_data_decision"] == "HOLD"
    assert report["g2_decision"] == "HOLD"
    assert report["n_records"] == 6
    assert report["n_historical_absent_mirror_matched"] == 2
    assert report["pilot"]["n_pairs"] == 3  # a.jpg shares present+absent; no overlap
    assert report["pilot"]["n_source_groups"] == 3
    inp = json.loads((out/"t02_pilot_inference_inputs_GT_FREE.json").read_text())
    assert inp["role"] == "EXPOSED_REFERENCE"
    for item in inp["items"]:
        assert "presence_label" not in item and "count_target" not in item
        assert "pair_id" not in item and item["query_id"].startswith("q_")
        assert "label_status" not in item and "gt" not in json.dumps(item).lower()
    with (out/"t02_pilot_labels_GT_ONLY.csv").open() as f:
        eval_rows = list(csv.DictReader(f))
    assert {row["query_id"] for row in eval_rows} == {item["query_id"] for item in inp["items"]}
    with (out/"t02_label_ledger_GT_ONLY.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert {r["proposed_split"] for r in rows} == {"EXPOSED_REFERENCE"}
    assert next(r for r in rows if r["record_id"] == "fresh160:X1")["label_status"] == "SINGLE_REVIEW_ABSENT"
    assert not next(r for r in rows if r["record_id"] == "fresh160:X1")["pilot_eligible"] == "True"


def test_strict_mirror_identity_blocks_bad_parity(tmp_path):
    paths = list(fixtures(tmp_path))
    with paths[1].open() as f:
        rows = list(csv.DictReader(f))
    rows[0]["query_target"] = "different"
    csv_file(paths[1], rows)
    with pytest.raises(ValueError, match="mirror pair identity mismatch"):
        build(*paths, tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_fresh_subset_label_conflict_blocks(tmp_path):
    paths = list(fixtures(tmp_path))
    with paths[3].open() as f:
        rows = list(csv.DictReader(f))
    rows[0]["presence_label"] = "PRESENT"
    csv_file(paths[3], rows)
    with pytest.raises(ValueError, match="fresh126 mismatch"):
        build(*paths, tmp_path/"out")


def test_report_attested_hash_required_for_byte_group(tmp_path):
    paths = fixtures(tmp_path)
    ledger = csv_file(tmp_path/"ledger.csv", [
        {"source_image_id":f"FSC147:{n}.jpg", "exists":"True", "image_sha256":c*64}
        for n,c in [("a","a"),("b","b"),("c","c")]])
    report_path=tmp_path/"identity.json"
    report_path.write_text(json.dumps({"ledger_sha256":file_sha256(ledger), "file_integrity_status":"PASS"}))
    report = build(*paths, tmp_path/"out", source_ledger=ledger, source_identity_report=report_path)
    assert report["p03_ledger_verification"] == "P03_SHA_ATTESTED"
    assert report["source_identity_proof_counts"]["P03_SHA_ATTESTED"] == 4
    assert report["g0_data_decision"] == "HOLD"


def test_invalid_p03_sha_blocks(tmp_path):
    paths = fixtures(tmp_path)
    ledger = csv_file(tmp_path/"ledger.csv", [
        {"source_image_id":"FSC147:a.jpg", "exists":"True", "image_sha256":"a"*64}])
    rp = tmp_path/"identity.json"
    rp.write_text(json.dumps({"ledger_sha256":"0"*64,"file_integrity_status":"PASS"}))
    with pytest.raises(ValueError, match="ledger SHA"):
        build(*paths, tmp_path/"out", source_ledger=ledger, source_identity_report=rp)


def test_missing_evidence_is_hold_and_create_only(tmp_path):
    paths = fixtures(tmp_path)
    out=tmp_path/"out"
    result=build(*paths,out,pilot_each=1)
    assert result["clean_proposal_audit"]["g2_decision"] == "HOLD"
    with pytest.raises(FileExistsError,match="create-only"):
        build(*paths,out)


def test_bad_positive_count_blocks(tmp_path):
    paths=list(fixtures(tmp_path))
    with paths[0].open() as f:
        rows = list(csv.DictReader(f))
    rows[0]["count_target"]="0"
    csv_file(paths[0],rows)
    with pytest.raises(ValueError,match="inconsistent FSC presence/count"):
        build(*paths,tmp_path/"out")


def test_zero_pilot_budget_is_empty(tmp_path):
    paths = fixtures(tmp_path)
    out = tmp_path/"out"
    report = build(*paths, out, pilot_each=0)
    assert report["pilot"]["n_pairs"] == 0
    assert not json.loads((out/"t02_pilot_inference_inputs_GT_FREE.json").read_text())["items"]


@pytest.mark.parametrize("num_exemplars", [3, 4, 5])
def test_real_fsc_positive_exemplar_values_are_preserved(tmp_path, num_exemplars):
    paths = list(fixtures(tmp_path))
    with paths[0].open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["num_exemplars"] = str(num_exemplars)
    csv_file(paths[0], rows)
    result = build(*paths, tmp_path / "out", pilot_each=1)
    assert result["fsc_exemplar_distribution"]["present"][str(num_exemplars)] >= 1
    with (tmp_path / "out" / "t02_label_ledger_GT_ONLY.csv").open(encoding="utf-8", newline="") as f:
        ledger = list(csv.DictReader(f))
    assert next(row for row in ledger if row["pair_id"] == "P1")["num_exemplars"] == str(num_exemplars)


def test_invalid_historical_exemplars_are_rejected(tmp_path):
    paths = list(fixtures(tmp_path))
    with paths[0].open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[2]["num_exemplars"] = "1"  # absent must not have exemplars
    csv_file(paths[0], rows)
    with pytest.raises(ValueError, match="historical FSC exemplar value unexpected"):
        build(*paths, tmp_path / "out")
    assert not (tmp_path / "out").exists()
