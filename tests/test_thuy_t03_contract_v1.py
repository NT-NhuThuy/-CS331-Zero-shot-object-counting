"""CPU-only T03 acceptance suite. No test labels or model artifacts are needed."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from countlab.method.schema import parse_input
from countlab.method.t03_contract_v1 import (
    T03ContractError, validate_candidate_pool, validate_method_output,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs/method/T03_example_input_output.json"


def fixtures():
    return json.loads(EXAMPLES.read_text(encoding="utf-8"))["examples"]


def base():
    return copy.deepcopy(fixtures()[1]["manifest"])


def assert_bad(manifest, fragment):
    with pytest.raises(T03ContractError, match=fragment):
        validate_candidate_pool(manifest)


@pytest.mark.parametrize("index", range(6))
def test_all_six_frozen_examples_are_valid(index):
    row = fixtures()[index]
    a = validate_candidate_pool(row["manifest"])
    b = validate_method_output(row["manifest"], row["prediction"])
    assert a["valid"] and b["valid"]
    assert a["input_sha256"] == b["input_sha256"]
    assert b["decision"] in {"ZERO", "K", "ABSTAIN"}


def test_empty_pool_is_valid_zero_not_certified_absence():
    row = fixtures()[0]
    a = validate_candidate_pool(row["manifest"])
    b = validate_method_output(row["manifest"], row["prediction"])
    assert (a["baseline_count"], a["rescue_count"], b["fallback_count"]) == (0, 0, 0)
    assert b["decision"] == "ZERO"
    assert row["prediction"]["count_status"] == "partial"


def test_unknown_baseline_preserves_fallback():
    row = fixtures()[1]
    assert row["prediction"]["evaluation_fallback_count"] == 1
    assert validate_method_output(row["manifest"], row["prediction"])["decision"] == "K"


def test_abstain_has_null_count_and_int_fallback():
    row = fixtures()[3]
    assert row["prediction"]["count"] is None
    assert row["prediction"]["evaluation_fallback_count"] == 1
    assert validate_method_output(row["manifest"], row["prediction"])["decision"] == "ABSTAIN"


def test_default_unary_is_unknown_not_no():
    m = base()
    del m["baseline"][0]["evidence"]
    validate_candidate_pool(m)
    proposal, = parse_input(m)[0]
    assert proposal.semantic == proposal.whole == "UNKNOWN"


def test_top_level_typo_rejected():
    m = base()
    m["baseline_counts"] = 1
    assert_bad(m, "unexpected key")


def test_missing_required_key_rejected():
    m = base()
    del m["baseline_count"]
    assert_bad(m, "missing key")


def test_candidate_unrecognized_field_rejected():
    m = base()
    m["baseline"][0]["scor"] = 0.5
    assert_bad(m, "unexpected key")


def test_unary_typo_rejected():
    m = base()
    m["baseline"][0]["evidence"]["whole"] = "YES"
    assert_bad(m, "unexpected key")


@pytest.mark.parametrize("key", ["gt_count", "GT_Count", "GroundTruthCount", "ground_truth_boxes", "annotations", "label_count"])
def test_recursive_gt_annotation_leak_rejected(key):
    m = base()
    m["metadata"] = {"run": [{"provenance": {key: 7}}]}
    assert_bad(m, "GT/annotation")


def test_unexpected_gt_even_if_false_is_rejected():
    m = base()
    m["uses_gt"] = False
    assert_bad(m, "unexpected key")


@pytest.mark.parametrize("bad", [True, "1", -1, 1.0, None])
def test_baseline_count_type_and_range(bad):
    m = base()
    m["baseline_count"] = bad
    assert_bad(m, "nonnegative integer")


def test_baseline_count_mismatch():
    m = base()
    m["baseline_count"] = 2
    assert_bad(m, "must equal")


@pytest.mark.parametrize("bad", [True, "0.8", float('nan'), -0.2, 2.0])
def test_reject_bad_confidence(bad):
    m = base()
    m["baseline"][0]["confidence"] = bad
    assert_bad(m, "must")


@pytest.mark.parametrize("bad", [False, "1", float('inf'), float('nan')])
def test_reject_invalid_box_coord(bad):
    m = base()
    m["baseline"][0]["box_xyxy"][0] = bad
    assert_bad(m, "finite")


def test_reject_zero_area_box():
    m = base()
    m["baseline"][0]["box_xyxy"] = [0, 0, 0, 10]
    assert_bad(m, "positive area")


def test_reject_duplicate_ids_across_roles():
    m = base()
    r = copy.deepcopy(m["baseline"][0])
    m["rescue"] = [r]
    assert_bad(m, "duplicate candidate ID")


def test_reject_duplicate_evidence_refs():
    m = base()
    m["baseline"][0]["evidence_refs"] = ["ref1", "ref1"]
    assert_bad(m, "duplicate evidence reference")


def test_unknown_unary_is_valid_but_wrong_verdict_rejected():
    m = base()
    m["baseline"][0]["evidence"]["semantic"] = "NOPE"
    assert_bad(m, "YES/NO/UNKNOWN")


def test_malformed_unary_type_rejected_with_contract_error():
    m = base()
    m["baseline"][0]["evidence"]["semantic"] = ["YES"]
    assert_bad(m, "YES/NO/UNKNOWN")


def test_identity_missing_target_rejected():
    m = base()
    m["identity_edges"] = [{"a": "b1", "b": "missing", "relation": "UNKNOWN"}]
    assert_bad(m, "distinct known candidates")


def test_identity_self_loop_rejected():
    m = base()
    m["identity_edges"] = [{"a": "b1", "b": "b1", "relation": "SAME"}]
    assert_bad(m, "distinct known candidates")


def test_identity_invalid_relation_rejected():
    m = base()
    m["baseline"].append({"candidate_id": "b2", "box_xyxy": [30,0,40,10]})
    m["baseline_count"] = 2
    m["identity_edges"] = [{"a":"b1","b":"b2","relation":"IDENTICAL"}]
    assert_bad(m, "SAME/DIFFERENT/UNKNOWN")


def test_identity_duplicate_reverse_pair_rejected():
    m = base()
    m["baseline"].append({"candidate_id": "b2", "box_xyxy": [30,0,40,10]})
    m["baseline_count"] = 2
    m["identity_edges"] = [
        {"a":"b1","b":"b2","relation":"SAME"},
        {"a":"b2","b":"b1","relation":"DIFFERENT"},
    ]
    assert_bad(m, "repeated identity pair")


def test_nested_evidence_edge_typo_rejected():
    m = base()
    m["baseline"].append({"candidate_id": "b2", "box_xyxy": [30,0,40,10]})
    m["baseline_count"] = 2
    m["identity_edges"] = [{"a":"b1","b":"b2","relation":"UNKNOWN","relatoin":"SAME"}]
    assert_bad(m, "unexpected key")


def test_output_rejects_false_abstention_count():
    row = copy.deepcopy(fixtures()[3])
    row["prediction"]["count"] = 1
    with pytest.raises(T03ContractError, match="must be null"):
        validate_method_output(row["manifest"], row["prediction"])


def test_output_rejects_hidden_mismatched_input():
    row = copy.deepcopy(fixtures()[1])
    row["manifest"]["query_text"] = "different query"
    with pytest.raises(T03ContractError, match="does not match frozen input"):
        validate_method_output(row["manifest"], row["prediction"])


def test_output_rejects_scene_wide_bounds_claim():
    row = copy.deepcopy(fixtures()[1])
    row["prediction"]["count_interval_scope"] = "exact_scene_bound"
    with pytest.raises(T03ContractError, match="scene-wide"):
        validate_method_output(row["manifest"], row["prediction"])


def test_output_rejects_bad_fallback_and_bounds():
    row = copy.deepcopy(fixtures()[1])
    row["prediction"]["count_upper_bound"] = 0
    with pytest.raises(T03ContractError, match="bounds"):
        validate_method_output(row["manifest"], row["prediction"])


def test_output_rejects_wrong_count_status():
    row = copy.deepcopy(fixtures()[0])
    row["prediction"]["count_status"] = "exact"
    with pytest.raises(T03ContractError, match="inconsistent"):
        validate_method_output(row["manifest"], row["prediction"])


def test_output_rejects_gt_fields_in_prediction():
    row = copy.deepcopy(fixtures()[0])
    row["prediction"]["eval_metadata"] = {"ground_truth_count": 0}
    with pytest.raises(T03ContractError, match="GT/annotation"):
        validate_method_output(row["manifest"], row["prediction"])


def test_output_rejects_bool_masquerading_as_one_count():
    row = copy.deepcopy(fixtures()[1])
    row["prediction"]["count"] = True
    with pytest.raises(T03ContractError, match="integer equal to fallback"):
        validate_method_output(row["manifest"], row["prediction"])


@pytest.mark.parametrize("index", range(6))
def test_solver_replay_still_agrees_with_frozen_t03_examples(index):
    from countlab.method.pipeline import run_method
    from countlab.method.joint_arbitration import run_joint_arbitration
    row = fixtures()[index]
    if row["method"] == "joint_t1":
        live = run_joint_arbitration(row["manifest"])
    else:
        live = run_method(row["manifest"], **row["options"])
    for key in ("schema", "count", "count_status", "query_abstained",
                "evaluation_fallback_count", "count_lower_bound", "count_upper_bound",
                "input_sha256"):
        assert live[key] == row["prediction"][key], (row["case"], key)
    validate_method_output(row["manifest"], live)


def test_existing_repository_toy_fixture_is_accepted():
    manifest = json.loads((ROOT / "fixtures/unified_toy.json").read_text(encoding="utf-8"))
    result = validate_candidate_pool(manifest)
    assert result["baseline_count"] == 2
    assert result["rescue_count"] == 2
    assert result["identity_edge_count"] == 1
