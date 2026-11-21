"""T12 fixed-schedule and matched-evidence contract tests (synthetic only)."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from countlab.method.t12_fixed_baselines_v1 import (
    fixed_schedule, run_t12_matched, validate_b5_config, validate_b6_config,
)

ROOT = Path(__file__).resolve().parents[1]
B5 = json.loads((ROOT / "configs/method/B5_config.json").read_text(encoding="utf-8"))
B6 = json.loads((ROOT / "configs/method/B6_config.json").read_text(encoding="utf-8"))
INPUTS = ROOT / "docs/method/T12_B5_B6_SYNTHETIC_INPUTS.jsonl"


def candidate(cid, box, role="baseline", risk=0.7, hidden="UNKNOWN"):
    return {"candidate_id": cid, "view_id": "full" if role == "baseline" else "crop",
            "box_xyxy": box, "risk_hint": risk, "confidence": .7,
            "evidence": {"semantic": hidden, "whole_instance": hidden},
            "evidence_refs": []}


def manifest(n=2, overlap=True):
    b = [candidate("b", [0, 0, 10, 10])]
    r = [candidate("r", [1 if overlap else 100, 1 if overlap else 100,
                           11 if overlap else 110, 11 if overlap else 110], "rescue")]
    return {"schema": "countlab_candidate_pool_v1", "image_id": "i", "query_id": "q",
            "query_text": "object", "baseline_count": 1,
            "baseline": b if n else [], "rescue": r if n > 1 else [],
            "identity_edges": [], "metadata": {"backend_protocol": "SYNTHETIC"}}


def unary(cid, op, verdict):
    return {"op": op, "candidate_id": cid, "verdict": verdict,
            "evidence_ref": f"{cid}:{op}:{verdict}"}


def events(n=2, rel="DIFFERENT"):
    out = []
    for cid in ("b", "r")[:n]:
        out += [unary(cid, "REGION_SEMANTIC", "YES"), unary(cid, "REGION_WHOLE", "YES")]
    if n > 1:
        out.append({"op": "PAIR_IDENTITY", "a": "b", "b": "r", "relation": rel,
                    "evidence_ref": "pair:br:" + rel})
    return out


def run(m=None, e=None, b6=None):
    return run_t12_matched(m if m is not None else manifest(),
                           e if e is not None else events(), b5_config=B5,
                           b6_config=b6 if b6 is not None else B6)


def test_configs_match_frozen_solver_defaults():
    assert validate_b5_config(B5)["policy"]["nms_iou"] == 0.5
    assert validate_b6_config(B6)["verification_budget_calls"] == 5


@pytest.mark.parametrize("bad", [-1, 1.5, None, True, "5"])
def test_fixed_budget_rejects_invalid_values(bad):
    c = copy.deepcopy(B6)
    c["verification_budget_calls"] = bad
    with pytest.raises(ValueError):
        validate_b6_config(c)


def test_schedule_frozen_before_outcomes_and_independent_of_hidden_evidence():
    m = manifest()
    initial = fixed_schedule(m, B6)
    altered = copy.deepcopy(m)
    altered["baseline"][0]["evidence"] = {"semantic": "YES", "whole_instance": "NO"}
    altered["rescue"][0]["evidence"] = {"semantic": "NO", "whole_instance": "YES"}
    altered["identity_edges"] = [{"a": "b", "b": "r", "relation": "SAME"}]
    assert fixed_schedule(altered, B6) == initial
    assert len(initial) == 5 and initial[-1] == {"op": "PAIR_IDENTITY", "a": "b", "b": "r"}
    assert run(m, events())["fixed_schedule_sha256"] == run(m, events(rel="SAME"))["fixed_schedule_sha256"]


def test_b6_uses_observed_same_and_different_without_geometric_truth_shortcut():
    same = run(e=events(rel="SAME"))
    diff = run(e=events(rel="DIFFERENT"))
    assert same["arms"]["B6_FIXED"]["decision"] == "K"
    assert same["arms"]["B6_FIXED"]["count"] == 1
    assert diff["arms"]["B6_FIXED"]["decision"] == "K"
    assert diff["arms"]["B6_FIXED"]["count"] == 2
    assert same["arms"]["B5_NMS"]["count"] == diff["arms"]["B5_NMS"]["count"] == 1
    assert same["arms"]["B5_COMPLETE_LINK"]["count"] == 1
    assert diff["arms"]["B5_COMPLETE_LINK"]["count"] == 2
    assert same["cost_scope"]["observed_slots"] == diff["cost_scope"]["observed_slots"] == 5
    assert same["cost_scope"]["actual_model_calls"] is None


def test_missing_event_is_unknown_no_refill_or_precomputed_leak():
    m = manifest()
    m["baseline"][0]["evidence"] = {"semantic": "YES", "whole_instance": "YES"}
    m["rescue"][0]["evidence"] = {"semantic": "YES", "whole_instance": "YES"}
    m["identity_edges"] = [{"a": "b", "b": "r", "relation": "DIFFERENT"}]
    result = run(m, [])
    assert result["cost_scope"]["scheduled_slots"] == 5
    assert result["cost_scope"]["missing_slots"] == 5
    assert result["arms"]["B6_FIXED"]["decision"] == "ABSTAIN"
    assert result["arms"]["B6_FIXED"]["count"] is None
    assert result["arms"]["B6_FIXED"]["evaluation_fallback_count"] == 1
    assert all(t["evidence_ref"] is None for t in result["action_trace"])


def test_unscheduled_bank_answers_do_not_influence_any_arm_decision():
    cfg = copy.deepcopy(B6)
    cfg["verification_budget_calls"] = 2
    extra = events()
    base = run(e=[], b6=cfg)
    with_extra = run(e=extra, b6=cfg)
    assert base["fixed_schedule_sha256"] == with_extra["fixed_schedule_sha256"]
    assert base["arms"] != with_extra["arms"]
    observed_unary = extra[:2]
    assert with_extra["arms"] == run(e=observed_unary, b6=cfg)["arms"]
    assert with_extra["cost_scope"]["observed_slots"] == 2
    assert with_extra["cost_scope"]["scheduled_slots"] == 2


def test_unknown_is_not_no_and_zero_is_not_scene_certified():
    m = manifest(n=1)
    e = [unary("b", "REGION_SEMANTIC", "NO"), unary("b", "REGION_WHOLE", "NO")]
    result = run(m, e)
    assert result["arms"]["B5_NMS"]["count"] == 0
    assert result["arms"]["B6_FIXED"]["decision"] == "ABSTAIN"
    assert result["arms"]["B6_FIXED"]["strict_result"]["observed_pool_decision"] == "ZERO"
    assert result["arms"]["B6_FIXED"]["strict_result"]["scene_zero_certified"] is False
    e[0]["verdict"] = "UNKNOWN"
    result2 = run(m, e)
    assert result2["arms"]["B5_NMS"]["count"] == 1


def test_empty_pool_does_not_certify_absence():
    m = manifest(n=0)
    m["baseline_count"] = 0
    result = run(m, [])
    assert result["arms"]["B6_FIXED"]["decision"] == "ABSTAIN"
    assert result["cost_scope"]["scheduled_slots"] == 0
    assert result["arms"]["B5_NMS"]["count"] == 0


@pytest.mark.parametrize("invalid", [
    lambda m, e: m.update({"gt_count": 99}),
    lambda m, e: e.append({"op": "REGION_WHOLE", "candidate_id": "x", "verdict": "YES", "evidence_ref": "x"}),
    lambda m, e: e.append({"op": "REGION_SEMANTIC", "candidate_id": "b", "verdict": "NO", "evidence_ref": "duplicate_action"}),
    lambda m, e: e[0].update({"ground_truth": 99}),
    lambda m, e: e[0].update({"verdict": "CERTAIN"}),
    lambda m, e: e[0].update({"evidence_ref": ""}),
])
def test_invalid_or_leaking_evidence_rejected(invalid):
    m, e = manifest(), events()
    invalid(m, e)
    with pytest.raises(ValueError):
        run(m, e)


def test_bank_event_permutation_is_decision_invariant():
    e = events()
    before = run(e=e)
    after = run(e=list(reversed(e)))
    assert before == after


def test_different_manifest_source_names_produce_distinct_hashes():
    m = manifest()
    a = run(m)
    m["image_id"] = "other_source"
    b = run(m)
    assert a["manifest_sha256"] != b["manifest_sha256"]


def test_real_model_cost_not_inferred_from_action_slots():
    result = run()
    assert result["online_verifier_calls_executed"] == 0
    assert result["cost_scope"]["matched_to_b7_measured_cost"] is False
    assert result["cost_scope"]["measured_gpu_seconds"] is None


def test_cli_create_only_and_complete_synthetic_fixture(tmp_path):
    out = tmp_path / "run"
    script = ROOT / "tools/thuy/t12_b5_b6_replay_v1.py"
    args = [sys.executable, str(script), "--inputs", str(INPUTS), "--out-dir", str(out)]
    first = subprocess.run(args, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    lock = json.loads((out / "replay_lock.json").read_text(encoding="utf-8"))
    data = (out / "B5_B6_predictions.jsonl").read_bytes()
    assert lock["roster_pairs"] >= 3
    assert lock["predictions_sha256"] == hashlib.sha256(data).hexdigest()
    for line in data.splitlines():
        result = json.loads(line)
        assert set(result["arms"]) == {"B5_NMS", "B5_COMPLETE_LINK", "B6_FIXED"}
    second = subprocess.run(args, text=True, capture_output=True)
    assert second.returncode != 0 and "create-only" in second.stderr


def test_all_three_arms_report_same_scheduled_observed_evidence():
    result = run()
    b5 = result["arms"]["B5_NMS"]
    b5_link = result["arms"]["B5_COMPLETE_LINK"]
    b6 = result["arms"]["B6_FIXED"]
    assert b5["used_observed_events_sha256"] == b5_link["used_observed_events_sha256"]
    assert b5["used_observed_events_sha256"] == b6["used_observed_events_sha256"]
    assert b6["used_observed_events_sha256"] == result["observed_events_sha256"]
    assert len([t for t in result["action_trace"] if t["observation_status"] == "OBSERVED"]) == 5


def test_schedule_uses_static_risk_and_candidate_id_tiebreak():
    m = manifest()
    m["baseline"][0]["risk_hint"] = 0.1
    m["rescue"][0]["risk_hint"] = 0.95
    schedule = fixed_schedule(m, B6)
    assert schedule[0] == {"op": "REGION_SEMANTIC", "candidate_id": "r"}
    assert schedule[1] == {"op": "REGION_WHOLE", "candidate_id": "r"}
    m["baseline"][0]["risk_hint"] = 0.95
    assert fixed_schedule(m, B6)[0] == {"op": "REGION_SEMANTIC", "candidate_id": "b"}


def test_far_pair_cannot_enter_geometric_prioritized_schedule():
    result = run(m=manifest(overlap=False))
    assert result["cost_scope"]["scheduled_slots"] == 4
    assert all(t["action"]["op"] != "PAIR_IDENTITY" for t in result["action_trace"])
    assert result["arms"]["B6_FIXED"]["decision"] == "ABSTAIN"


def test_budget_zero_cannot_consume_precomputed_bank():
    cfg = copy.deepcopy(B6)
    cfg["verification_budget_calls"] = 0
    result = run(e=events(), b6=cfg)
    assert result["action_trace"] == []
    assert result["cost_scope"]["observed_slots"] == 0
    assert result["arms"]["B6_FIXED"]["decision"] == "ABSTAIN"
    assert result["arms"]["B5_NMS"]["used_observed_events_sha256"] == result["observed_events_sha256"]
