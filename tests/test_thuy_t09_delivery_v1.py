"""T09 G3 acceptance: closed GT-free boundary, identity, conflicts, trace.

All fixtures are synthetic; no experiment/GT or model inference is involved.
"""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from countlab.method.joint_solver_v2 import DEFAULT_SOLVER_CONFIG, solve_joint_v2
from countlab.method.t09_delivery_v1 import run_t09_delivery
from countlab.method.t03_contract_v1 import T03ContractError

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/method/solver_config.json"
EXAMPLES = ROOT / "docs/method/solver_trace_examples.jsonl"
TOOL = ROOT / "tools/thuy/t09_validate_v1.py"


def config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def examples():
    return [json.loads(line) for line in EXAMPLES.read_text(encoding="utf-8").splitlines() if line.strip()]


def example(name):
    return next(x for x in examples() if x["case"] == name)


def run(row):
    return run_t09_delivery(row["manifest"], row["observed_events"],
                            fallback_count=row["fallback_count"], config=config())


@pytest.mark.parametrize("name", [x["case"] for x in examples()])
def test_frozen_trace_exactly_replays(name):
    row = example(name)
    actual = run(row)
    assert actual == row["solver_output"]
    assert actual["decision"] == row["expected_decision"]
    assert actual["count"] == row["expected_count"]
    assert actual["executed_model_calls_by_T09"] == 0
    assert actual["gt_used"] is False
    assert actual["operational_count_correction_authorized"] is False
    assert actual["scene_zero_certified"] is False


def test_config_matches_frozen_entrypoint_defaults():
    assert config() == DEFAULT_SOLVER_CONFIG


def test_no_candidate_cannot_certify_zero():
    outcome = run(example("empty_pool"))
    assert outcome["decision"] == "ABSTAIN"
    assert outcome["count"] is None
    assert outcome["evaluation_fallback_count"] == 0
    assert outcome["decision_reason"] == "EMPTY_POOL_CANNOT_PROVE_SCENE_ABSENCE"
    assert outcome["observed_pool_decision"] == "ABSTAIN"


def test_dual_no_does_not_claim_scene_zero():
    outcome = run(example("dual_no_baseline_not_scene_zero"))
    assert outcome["decision"] == "ABSTAIN"
    assert outcome["evaluation_fallback_count"] == 1
    assert outcome["candidate_trace"][0]["observed_semantic"] == "NO"
    assert outcome["candidate_trace"][0]["observed_whole_instance"] == "NO"
    assert outcome["observed_pool_decision"] == "ZERO"
    assert outcome["observed_pool_count"] == 0
    assert outcome["observed_pool_zero_not_scene_certified"] is True


def test_explicit_different_witness_keeps_distinct_overlapping_boxes():
    out = run(example("explicit_different_k2"))
    assert out["decision"] == "K" and out["count"] == 2
    assert out["observed_pool_decision"] == "K"
    witnesses = out["strict_result"]["physical_instances"]
    assert len(witnesses) == 2
    assert all(x["different_identity_evidence_refs"] for x in witnesses)
    assert {x["role"] for x in out["candidate_trace"]} == {"baseline", "rescue"}


def test_same_dedup_requires_observed_identity():
    out = run(example("same_object_dedup"))
    assert out["decision"] == "K" and out["count"] == 1
    witnesses = out["strict_result"]["physical_instances"]
    assert set(witnesses[0]["member_candidate_ids"]) == {"b1", "r1"}
    assert witnesses[0]["same_identity_evidence_refs"]


def test_same_without_direct_pair_does_not_auto_dedup():
    row = copy.deepcopy(example("same_object_dedup"))
    row["observed_events"] = [e for e in row["observed_events"] if e["op"] != "PAIR_IDENTITY"]
    out = run(row)
    assert out["decision"] == "ABSTAIN"
    assert out["count"] is None


def test_unknown_identity_never_assumed_different():
    row = copy.deepcopy(example("missing_identity_different"))
    row["observed_events"].append({"op": "PAIR_IDENTITY", "a": "b1", "b": "r1",
                                    "relation": "UNKNOWN", "evidence_ref": "pair:unknown"})
    assert run(row)["decision"] == "ABSTAIN"


def test_same_chain_incomplete_does_not_merge_transitively():
    out = run(example("incomplete_same_clique"))
    assert out["decision"] == "ABSTAIN"
    assert any(x.startswith("incomplete_SAME_clique") for x in out["blocking_issues"])


def test_decisive_identity_conflict_blocks_count():
    out = run(example("pair_same_different_conflict"))
    assert out["decision"] == "ABSTAIN"
    assert "verifier_conflict:PAIR_IDENTITY:b1,r1" in out["blocking_issues"]


def test_decisive_unary_conflict_blocks_count():
    out = run(example("unary_yes_no_conflict"))
    assert out["decision"] == "ABSTAIN"
    assert out["candidate_trace"][0]["observed_semantic"] == "CONFLICT"


def test_unknown_then_yes_is_valid_retry():
    out = run(example("unknown_retry_then_yes"))
    assert out["decision"] == "K"
    assert out["candidate_trace"][0]["observed_semantic"] == "YES"
    assert "s-unknown" in out["candidate_trace"][0]["observed_evidence_refs"]


def test_precomputed_unobserved_answers_do_not_enter_solver():
    row = copy.deepcopy(example("hidden_precomputed_answers_ignored"))
    without_hidden = copy.deepcopy(row)
    for p in without_hidden["manifest"]["baseline"] + without_hidden["manifest"]["rescue"]:
        p["evidence"] = {"semantic": "UNKNOWN", "whole_instance": "UNKNOWN"}
    without_hidden["manifest"]["identity_edges"] = []
    actual = run(row)
    masked = run(without_hidden)
    assert actual["decision"] == masked["decision"] == "ABSTAIN"
    assert actual["source_visible_sha256"] == masked["source_visible_sha256"]
    assert actual["observed_events_sha256"] == masked["observed_events_sha256"]


def test_event_permutation_yields_same_normalized_result():
    row = copy.deepcopy(example("explicit_different_k2"))
    initial = run(row)
    row["observed_events"].reverse()
    assert run(row) == initial


def test_one_nonnegative_conditional_rescue_can_change_fallback():
    out = run(example("rescue_only_positive_conditional"))
    assert out["count"] == 1
    assert out["evaluation_fallback_count"] == 1
    assert out["strict_result"]["count_changed"] is True


def test_candidate_trace_uses_only_observed_refs():
    for row in examples():
        out = run(row)
        refs = {event["evidence_ref"] for event in row["observed_events"]}
        candidate_ids = {p["candidate_id"] for p in row["manifest"]["baseline"] + row["manifest"]["rescue"]}
        assert {p["candidate_id"] for p in out["candidate_trace"]} == candidate_ids
        for item in out["candidate_trace"]:
            assert set(item["observed_evidence_refs"]) <= refs
            for edge in item["observed_identity_edges"]:
                assert edge["evidence_ref"] in refs
                assert edge["peer_candidate_id"] in candidate_ids


@pytest.mark.parametrize("bad_key", ["gt_count", "ground_truth", "annotation"])
def test_gt_leak_is_rejected_even_in_metadata(bad_key):
    row = copy.deepcopy(example("singleton_k"))
    row["manifest"]["metadata"] = {"nested": {bad_key: 9000}}
    with pytest.raises(T03ContractError, match="GT/annotation"):
        run(row)


def test_unknown_manifest_field_rejected():
    row = copy.deepcopy(example("singleton_k"))
    row["manifest"]["future_count_label"] = 19
    with pytest.raises(T03ContractError):
        run(row)


def test_duplicate_candidate_id_rejected():
    row = copy.deepcopy(example("explicit_different_k2"))
    row["manifest"]["rescue"][0]["candidate_id"] = "b1"
    with pytest.raises(T03ContractError, match="duplicate candidate ID"):
        run(row)


@pytest.mark.parametrize("invalid", [
    {"op": "PAIR_IDENTITY", "a": "b1", "b": "r1", "relation": "DIFFERENT", "evidence_ref": "dup"},
    {"op": "REGION_SEMANTIC", "candidate_id": "r1", "verdict": "YES", "evidence_ref": "dup", "gt": 2},
])
def test_strict_event_key_contract_blocks_errors(invalid):
    row = copy.deepcopy(example("explicit_different_k2"))
    if "gt" not in invalid:
        row["observed_events"].append(dict(invalid, evidence_ref=row["observed_events"][0]["evidence_ref"]))
    else:
        row["observed_events"].append(invalid)
    with pytest.raises(ValueError):
        run(row)


def test_unknown_event_target_rejected():
    row = copy.deepcopy(example("singleton_k"))
    row["observed_events"][0]["candidate_id"] = "injected-candidate"
    with pytest.raises(ValueError):
        run(row)


@pytest.mark.parametrize("bad", [
    {"max_search_states": -1},
    {"max_candidates": 0},
    {"min_same_iou": 1.4},
    {"max_hints": True},
])
def test_invalid_solver_config_rejected(bad):
    row = example("singleton_k")
    cfg = config()
    cfg.update(bad)
    with pytest.raises(ValueError):
        run_t09_delivery(row["manifest"], row["observed_events"],
                         fallback_count=row["fallback_count"], config=cfg)


def test_candidate_cap_fails_closed():
    row = example("explicit_different_k2")
    cfg = config()
    cfg["max_candidates"] = 1
    out = run_t09_delivery(row["manifest"], row["observed_events"],
                           fallback_count=row["fallback_count"], config=cfg)
    assert out["decision"] == "ABSTAIN"
    assert out["evaluation_fallback_count"] == 1
    assert out["blocking_issues"] == ["resource_limit:candidate_cap"]


def test_bad_fallback_rejected():
    row = example("singleton_k")
    with pytest.raises(ValueError):
        run_t09_delivery(row["manifest"], row["observed_events"],
                         fallback_count=-1, config=config())


def test_cli_replays_all_frozen_examples():
    run_cli = subprocess.run([sys.executable, "-B", str(TOOL), "--config", str(CONFIG),
                              "--examples", str(EXAMPLES)], cwd=str(ROOT),
                             capture_output=True, text=True, check=True)
    assert "T09_TRACE_SUITE_PASS 13" in run_cli.stdout


def test_cli_writes_real_solver_output_without_overwriting(tmp_path):
    row = example("explicit_different_k2")
    m, e, o = [tmp_path / x for x in ("manifest.json", "events.json", "output.json")]
    m.write_text(json.dumps(row["manifest"]), encoding="utf-8")
    e.write_text(json.dumps(row["observed_events"]), encoding="utf-8")
    cmd = [sys.executable, "-B", str(TOOL), "--config", str(CONFIG), "--manifest", str(m),
           "--events", str(e), "--fallback", "1", "--out", str(o)]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=True)
    assert "T09_DECISION K 2" in result.stdout
    assert json.loads(o.read_text(encoding="utf-8")) == row["solver_output"]
    again = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert again.returncode != 0


def test_primary_solver_source_not_required_to_change():
    row = example("singleton_k")
    core = solve_joint_v2(row["manifest"], row["observed_events"], fallback_count=1, config=config())
    assert run(row)["strict_result"] == core


def test_all_negative_must_include_all_rescue_candidates_to_label_observed_zero():
    row = copy.deepcopy(example("dual_no_baseline_not_scene_zero"))
    row["manifest"]["rescue"].append({"candidate_id": "r2", "box_xyxy": [30, 30, 40, 40]})
    out = run(row)
    assert out["observed_pool_decision"] == "ABSTAIN"
    assert out["observed_pool_count"] is None


def test_all_negative_unary_conflict_must_not_label_zero():
    row = copy.deepcopy(example("dual_no_baseline_not_scene_zero"))
    row["observed_events"].append({"op": "REGION_SEMANTIC", "candidate_id": "b1",
                                   "verdict": "YES", "evidence_ref": "s-conflict"})
    out = run(row)
    assert out["observed_pool_decision"] == "ABSTAIN"
    assert out["candidate_trace"][0]["observed_semantic"] == "CONFLICT"
