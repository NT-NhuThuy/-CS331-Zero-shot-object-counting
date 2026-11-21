"""T06 acceptance: roster integrity, observed evidence, GT separation, replay hashes."""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from countlab.method.t06_pilot_replay_v1 import (
    replay_batch, replay_row, parse_jsonl, run_pilot_file, validate_config,
    canonical_bytes, jsonl_bytes, sha_obj,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "docs/method/T06_SYNTHETIC_PILOT_INPUTS.jsonl"
CONFIG = ROOT / "configs/method/t06_pilot_replay_config.json"
CLI = ROOT / "tools/thuy/t06_pilot_replay_v1.py"


def rows():
    return parse_jsonl(FIXTURES.read_bytes())


def config():
    return json.loads(CONFIG.read_text())


def by_case(name):
    t09 = ROOT / "docs/method/solver_trace_examples.jsonl"
    r = next(json.loads(x) for x in t09.read_text().splitlines() if json.loads(x)["case"] == name)
    return {"schema":"countlab_t06_pilot_input_row_v1","manifest":r["manifest"],
            "observed_events":r["observed_events"],"feature_join":{"status":"UNAVAILABLE"},
            "provenance":{"mode":"SYNTHETIC_CONTRACT"}}


def test_exactly_nine_fixture_pairs_and_no_duplicates():
    xs = rows()
    assert len(xs) == 9
    assert len({(r["manifest"]["image_id"], r["manifest"]["query_id"]) for r in xs}) == len(xs)


@pytest.mark.parametrize("name,decision,count", [
    ("empty_pool","ABSTAIN",None),
    ("unknown_baseline","ABSTAIN",None),
    ("dual_no_baseline_not_scene_zero","ABSTAIN",None),
    ("singleton_k","K",1),
    ("explicit_different_k2","K",2),
    ("same_object_dedup","K",1),
    ("pair_same_different_conflict","ABSTAIN",None),
    ("hidden_precomputed_answers_ignored","ABSTAIN",None),
    ("rescue_only_positive_conditional","K",1),
])
def test_all_synthetic_cases(name, decision, count):
    pred, trace = replay_row(by_case(name), solver_config=config()["solver_config"])
    assert (pred["decision"],pred["count"]) == (decision,count)
    assert pred["n_candidate_trace_nodes"] == len(trace["nodes"])
    assert pred["graph_trace_sha256"] == sha_obj(trace)
    assert pred["scene_zero_certified"] is False
    assert pred["gt_used"] is False
    assert pred["model_calls_by_T06"] == 0
    assert pred["handoff_status"] == "SYNTHETIC_CPU_CONTRACT_ONLY"
    assert all(x["observed"]["candidate_id"] == x["candidate_id"] for x in trace["nodes"])


def test_empty_absent_is_not_certified_zero():
    out, _ = replay_row(by_case("empty_pool"), solver_config=config()["solver_config"])
    assert out["evaluation_fallback_count"] == 0
    assert out["count"] is None
    assert out["observed_pool_decision"] == "ABSTAIN"


def test_missing_evidence_is_never_no():
    _, trace = replay_row(by_case("unknown_baseline"), solver_config=config()["solver_config"])
    assert trace["nodes"][0]["observed"]["observed_semantic"] == "UNKNOWN"
    assert trace["nodes"][0]["observed"]["observed_whole_instance"] == "UNKNOWN"


def test_same_reduces_count_only_when_observed():
    row = by_case("same_object_dedup")
    pred, trace = replay_row(row, solver_config=config()["solver_config"])
    assert pred["count"] == 1 and pred["n_strict_instances"] == 1
    assert set(trace["physical_instances"][0]["member_candidate_ids"]) == {"b1","r1"}
    changed = copy.deepcopy(row)
    changed["observed_events"] = [e for e in changed["observed_events"] if e["op"] != "PAIR_IDENTITY"]
    new, _ = replay_row(changed, solver_config=config()["solver_config"])
    assert new["decision"] == "ABSTAIN"


def test_no_hidden_positive_event_influence():
    x = by_case("hidden_precomputed_answers_ignored")
    pred,_ = replay_row(x, solver_config=config()["solver_config"])
    for node in x["manifest"]["baseline"] + x["manifest"]["rescue"]:
        node["evidence"] = {"semantic":"UNKNOWN","whole_instance":"UNKNOWN"}
    x["manifest"]["identity_edges"] = []
    cleaned,_ = replay_row(x, solver_config=config()["solver_config"])
    assert pred["decision"] == cleaned["decision"] == "ABSTAIN"
    assert pred["source_visible_sha256"] == cleaned["source_visible_sha256"]


@pytest.mark.parametrize("bad", ["ground_truth_count", "annotation", "gt_count", "labels"])
def test_gt_leak_anywhere_rejected(bad):
    x = by_case("singleton_k")
    x["provenance"][bad] = 999
    with pytest.raises(ValueError, match="GT/annotation"):
        replay_row(x, solver_config=config()["solver_config"])


def test_wrong_event_candidate_rejected():
    x = by_case("singleton_k")
    x["observed_events"][0]["candidate_id"] = "ghost"
    with pytest.raises(ValueError):
        replay_row(x, solver_config=config()["solver_config"])


def test_wrong_event_edge_rejected():
    x = by_case("same_object_dedup")
    e = next(e for e in x["observed_events"] if e["op"]=="PAIR_IDENTITY")
    e["b"]="ghost"
    with pytest.raises(ValueError):
        replay_row(x, solver_config=config()["solver_config"])


def test_repeat_event_ref_rejected():
    x = by_case("singleton_k")
    x["observed_events"][1]["evidence_ref"] = x["observed_events"][0]["evidence_ref"]
    with pytest.raises(ValueError, match="duplicated evidence_ref"):
        replay_row(x, solver_config=config()["solver_config"])


def test_pairwise_duplicate_candidate_id_fails():
    x = by_case("same_object_dedup")
    x["manifest"]["rescue"][0]["candidate_id"]="b1"
    with pytest.raises(ValueError, match="duplicate candidate ID"):
        replay_row(x, solver_config=config()["solver_config"])


def test_noncanonical_feature_join_rejected():
    x = by_case("same_object_dedup")
    x["feature_join"]={"status":"EXACT_ID_JOIN_DECLARED_UNVERIFIED","feature_source_sha256":"a"*64,
                       "candidate_map":{"b1":{"node_graph_id":1,"raw_query_index":1}}}
    with pytest.raises(ValueError, match="exactly all"):
        replay_row(x, solver_config=config()["solver_config"])


def test_feature_join_complete_attested_only_as_declared():
    x = by_case("same_object_dedup")
    x["feature_join"]={"status":"EXACT_ID_JOIN_DECLARED_UNVERIFIED","feature_source_sha256":"a"*64,
                       "candidate_map":{"b1":{"node_graph_id":1,"raw_query_index":1},
                                        "r1":{"node_graph_id":2,"raw_query_index":1}}}
    pred, trace = replay_row(x, solver_config=config()["solver_config"])
    assert trace["feature_join"]["mapped_candidates"] == 2
    assert "UNVERIFIED" in trace["feature_join"]["status"]
    assert pred["count"] == 1


def test_duplicate_node_raw_feature_pair_rejected():
    x=by_case("same_object_dedup")
    x["feature_join"]={"status":"EXACT_ID_JOIN_DECLARED_UNVERIFIED","feature_source_sha256":"a"*64,
                       "candidate_map":{"b1":{"node_graph_id":1,"raw_query_index":1},
                                        "r1":{"node_graph_id":1,"raw_query_index":1}}}
    with pytest.raises(ValueError, match="duplicate node"):
        replay_row(x, solver_config=config()["solver_config"])


def test_external_provenance_never_becomes_signed_off():
    x = by_case("singleton_k")
    x["provenance"]={"mode":"EXTERNAL_HANDOFF_UNVERIFIED",
                    "source_image_sha256":"1"*64,"t04_pool_sha256":"2"*64,
                    "t07_events_sha256":"3"*64}
    pred,_=replay_row(x,solver_config=config()["solver_config"])
    assert pred["handoff_status"] == "EXTERNAL_UNVERIFIED_PENDING_C1_C2_SIGNOFF"


def test_external_provenance_missing_hash_rejected():
    x = by_case("singleton_k")
    x["provenance"]={"mode":"EXTERNAL_HANDOFF_UNVERIFIED",
                     "source_image_sha256":"a"*64,"t04_pool_sha256":"a"*64}
    with pytest.raises(ValueError):
        replay_row(x, solver_config=config()["solver_config"])


def test_config_must_disable_claims():
    cfg = config();cfg["claims_allowed"] = True
    with pytest.raises(ValueError):validate_config(cfg)


def test_replay_rejects_duplicate_query_roster():
    xs=rows();xs.append(copy.deepcopy(xs[0]))
    with pytest.raises(ValueError, match="duplicate image/query"):
        replay_batch(xs,config())


def test_batch_never_drops_queries_and_preserves_order():
    preds, traces, summary = replay_batch(rows(),config())
    ids = [(x["manifest"]["image_id"],x["manifest"]["query_id"]) for x in rows()]
    assert [(p["image_id"],p["query_id"]) for p in preds] == ids
    assert [(t["image_id"],t["query_id"]) for t in traces] == ids
    assert summary["roster_pairs"]==len(ids)==len(preds)==len(traces)
    assert summary["synthetic_pairs"]==len(ids)
    assert summary["real_handoff_accepted"] is False


def test_parse_blank_line_fails_closed():
    with pytest.raises(ValueError,match="blank row"):
        parse_jsonl(b'{}\n\n{}\n')


def test_nan_payload_is_rejected():
    with pytest.raises(ValueError, match="malformed JSON"):
        parse_jsonl(b'{"a": NaN}\n')


def test_duplicate_json_keys_rejected():
    with pytest.raises(ValueError, match="malformed JSON"):
        parse_jsonl(b'{"schema":"bad","schema":"good"}\n')


def test_run_file_deterministic_bytes_create_only_and_locked(tmp_path):
    a=tmp_path/"run_a";b=tmp_path/"run_b"
    first=run_pilot_file(FIXTURES,CONFIG,a)
    second=run_pilot_file(FIXTURES,CONFIG,b)
    assert first==second
    assert sorted(x.name for x in a.iterdir()) == sorted(x.name for x in b.iterdir())
    for name in ("pilot_predictions.jsonl", "graph_trace.jsonl", "pilot_replay_report.md", "pilot_replay_lock.json"):
        assert (a/name).read_bytes() == (b/name).read_bytes()
    assert first["predictions_sha256"]==hashlib.sha256((a/"pilot_predictions.jsonl").read_bytes()).hexdigest()
    assert first["graph_trace_sha256"]==hashlib.sha256((a/"graph_trace.jsonl").read_bytes()).hexdigest()
    assert first["real_integration_signed_off"] is False
    with pytest.raises(FileExistsError):run_pilot_file(FIXTURES,CONFIG,a)


def test_invalid_roster_no_output_created(tmp_path):
    xs=rows();xs[3]["observed_events"][0]["candidate_id"]="ghost"
    p=tmp_path/"bad.jsonl";p.write_bytes(jsonl_bytes(xs))
    out=tmp_path/"output"
    with pytest.raises(ValueError):run_pilot_file(p,CONFIG,out)
    assert not out.exists()


def test_t06_cli(tmp_path):
    out=tmp_path/"cli"
    cmd=[sys.executable,"-B",str(CLI),"--inputs",str(FIXTURES),"--config",str(CONFIG),"--out-dir",str(out)]
    run=subprocess.run(cmd,capture_output=True,text=True,cwd=ROOT)
    assert run.returncode==0, run.stderr
    assert "T06_CPU_CONTRACT_REPLAY_PASS_REAL_PILOT_HOLD" in run.stdout
    assert (out/"pilot_predictions.jsonl").is_file()
    assert len((out/"graph_trace.jsonl").read_text().splitlines())==9


def test_frozen_synthetic_delivery_matches_reference_bytes(tmp_path):
    output = tmp_path / "reference"
    result = run_pilot_file(FIXTURES, CONFIG, output)
    known = {
        "pilot_predictions.jsonl": ROOT / "docs/method/T06_SYNTHETIC_pilot_predictions.jsonl",
        "graph_trace.jsonl": ROOT / "docs/method/T06_SYNTHETIC_graph_trace.jsonl",
        "pilot_replay_lock.json": ROOT / "docs/method/T06_SYNTHETIC_REPLAY_LOCK.json",
    }
    for produced, reference in known.items():
        assert (output / produced).read_bytes() == reference.read_bytes()
    assert result["roster_pairs"] == 9
    assert result["real_integration_signed_off"] is False


def test_event_permutation_does_not_change_strict_decision():
    a = by_case("explicit_different_k2")
    b = copy.deepcopy(a)
    b["observed_events"].reverse()
    p1, _ = replay_row(a, solver_config=config()["solver_config"])
    p2, _ = replay_row(b, solver_config=config()["solver_config"])
    assert (p1["decision"],p1["count"],p1["evaluation_fallback_count"]) == (
        p2["decision"],p2["count"],p2["evaluation_fallback_count"])


def test_declared_external_source_image_hash_mismatch_rejected():
    x = by_case("singleton_k")
    x["manifest"]["metadata"] = {"image_sha256": "4" * 64}
    x["provenance"] = {"mode": "EXTERNAL_HANDOFF_UNVERIFIED",
                       "source_image_sha256": "1" * 64, "t04_pool_sha256": "2" * 64,
                       "t07_events_sha256": "3" * 64}
    with pytest.raises(ValueError, match="source-image SHA mismatch"):
        replay_row(x, solver_config=config()["solver_config"])


def test_declared_t05_feature_hash_mismatch_rejected():
    x = by_case("singleton_k")
    x["manifest"]["metadata"] = {"thu_frozen_features": {"feature_sha256": "a" * 64}}
    x["feature_join"] = {"status": "EXACT_ID_JOIN_DECLARED_UNVERIFIED",
                         "feature_source_sha256": "b" * 64,
                         "candidate_map": {"b1": {"node_graph_id": 1, "raw_query_index": 1}}}
    x["provenance"] = {"mode": "EXTERNAL_HANDOFF_UNVERIFIED",
                       "source_image_sha256": "1" * 64, "t04_pool_sha256": "2" * 64,
                       "t07_events_sha256": "3" * 64}
    with pytest.raises(ValueError, match="T05 feature SHA mismatch"):
        replay_row(x, solver_config=config()["solver_config"])
