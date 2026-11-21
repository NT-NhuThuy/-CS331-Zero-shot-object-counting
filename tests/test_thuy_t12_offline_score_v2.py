"""Synthetic-only unit tests. No image access, verifier, or GT inference."""

import importlib.util
import csv
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/thuy/t12_offline_score_v2.py"
spec = importlib.util.spec_from_file_location("c3_t12_scorer_v2_candidate", SCRIPT)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def prediction(image_id="image:1", query_id="query:1", group="group:1", a0=3, a4=2, *, abstain=True):
    a5 = {"forced_count": a0 if abstain else 2,
          "original_fallback_count": a0,
          "selective_count": None if abstain else 2,
          "status": "ABSTAIN" if abstain else "PROVISIONAL_POSITIVE"}
    return {"image_id": image_id, "query_id": query_id, "source_group_id": group,
            "cohort_role": "dev", "pool_scope": "full_query_pool",
            "split_protocol_id": "split:approved", "frozen_pool_run_id": "pool:approved",
            "t12_protocol_sha256": scorer.PROTOCOL_SHA, "gt_used": False,
            "scene_zero_certified": False, "model_calls_in_this_replay": 0,
            "measured_verifier_cost_attached": False,
            "arms": {"A0": {"forced_count": a0}, "A1": {"forced_count": a0},
                     "A3": {"forced_count": a0}, "A4": {"forced_count": a4, "search_exact": True},
                     "A5": a5}}


def roster(image_id="image:1", query_id="query:1", group="group:1", eligibility="COUNT_ELIGIBLE"):
    return {"dataset": "fsc147", "image_id": image_id, "query_id": query_id,
            "source_image_id": "src:" + image_id, "source_group_id": group,
            "split": "DEV", "eligibility": eligibility, "exposure_status": "UNEXPOSED",
            "lineage_status": "VERIFIED"}


def label(row, *, presence="PRESENT", count=2):
    return {**{k: row[k] for k in ("dataset", "image_id", "query_id", "source_image_id", "source_group_id", "split")},
            "presence_label": presence, "gt_count": count}


def assemble(rows):
    preds = [p for p, _, _ in rows]
    rosters = [r for _, r, _ in rows]
    labels = [y for _, _, y in rows]
    return scorer.build_items(preds, rosters, labels,
                              split_id="split:approved", pool_id="pool:approved")


def test_count_and_presence_denominators_are_distinct():
    r0 = roster(eligibility="COUNT_ELIGIBLE")
    r1 = roster("image:2", "query:2", "group:2", "PRESENCE_ONLY")
    r2 = roster("image:3", "query:3", "group:3", "INELIGIBLE")
    rows = [
        (prediction(a0=3, a4=2), r0, label(r0, presence="PRESENT", count=2)),
        (prediction("image:2", "query:2", "group:2", a0=0, a4=1), r1,
         label(r1, presence="PRESENT", count=None)),
        (prediction("image:3", "query:3", "group:3", a0=50, a4=0), r2,
         label(r2, presence=None, count=None)),
    ]
    items = assemble(rows)
    m0 = scorer.metrics(items, "A0")
    m4 = scorer.metrics(items, "A4")
    assert (m0["n_count_eligible"], m0["n_presence_only"], m0["n_ineligible"]) == (1, 1, 1)
    assert m0["overall_mae"] == 1
    assert m0["present_mae"] == 1
    assert m0["fzr_plus"] == pytest.approx(0.5)
    assert m4["present_mae"] == 0
    assert m4["fzr_plus"] == 0
    assert scorer.paired_help_harm(items, "A4", "A0") == {
        "n": 1, "help": 1, "harm": 0, "tie": 0,
        "interpretation": "DESCRIPTIVE_PAIRED_QUERY_COUNTS"
    }


def test_presence_only_absent_aza_is_counted_but_not_mae():
    r = roster(eligibility="PRESENCE_ONLY")
    items = assemble([(prediction(a0=0, a4=1), r, label(r, presence="ABSENT", count=None))])
    m = scorer.metrics(items, "A0")
    assert m["n_count_eligible"] == 0
    assert m["overall_mae"] is None and m["present_mae"] is None
    assert m["aza"] == 1 and m["fzr_plus"] is None


def test_count_eligible_absent_included_in_overall_not_present():
    r = roster()
    items = assemble([(prediction(a0=3), r, label(r, presence="ABSENT", count=0))])
    m = scorer.metrics(items, "A0")
    assert m["overall_mae"] == 3 and m["present_mae"] is None
    assert m["aza"] == 0


@pytest.mark.parametrize("eligibility,presence,count,reason", [
    ("COUNT_ELIGIBLE", "ABSENT", 2, "COUNT_PRESENCE_CONTRADICTION"),
    ("COUNT_ELIGIBLE", "PRESENT", None, "COUNT_ELIGIBLE_GT_INVALID"),
    ("PRESENCE_ONLY", "PRESENT", 0, "PRESENCE_ONLY_GT_COUNT_FORBIDDEN"),
    ("PRESENCE_ONLY", None, None, "PRESENCE_ONLY_LABEL_INVALID"),
    ("INELIGIBLE", "PRESENT", None, "INELIGIBLE_LABELS_MUST_BE_NULL"),
    ("UNVERIFIED", None, None, "UNVERIFIED_OR_UNKNOWN_ELIGIBILITY"),
])
def test_fail_closed_eligibility(eligibility, presence, count, reason):
    r = roster(eligibility=eligibility)
    with pytest.raises(ValueError, match=reason):
        scorer.score_label(r, label(r, presence=presence, count=count))


def test_join_rejects_unmatched_queries():
    r = roster()
    with pytest.raises(ValueError, match="DEV_JOIN_NOT_EXACT"):
        scorer.build_items([prediction()], [r], [], split_id="split:approved", pool_id="pool:approved")


def test_source_group_mismatch_blocks_scoring():
    r = roster()
    p = prediction(group="wrong-group")
    with pytest.raises(ValueError, match="PREDICTION_SOURCE_GROUP_MISMATCH"):
        assemble([(p, r, label(r))])


def test_a1_a3_parity_and_abstention_fallback_enforced():
    r = roster()
    p = prediction()
    p["arms"]["A3"]["forced_count"] = 5
    with pytest.raises(ValueError, match="A1_A3_PARITY_FAILED"):
        assemble([(p, r, label(r))])
    p = prediction()
    p["arms"]["A5"]["forced_count"] = 0
    with pytest.raises(ValueError, match="A5_ABSTAIN_FALLBACK_INVALID"):
        assemble([(p, r, label(r))])


def test_a5_separate_abstention_and_selective_metrics():
    r1 = roster()
    r2 = roster("image:2", "query:2", "group:2")
    rows = [(prediction(), r1, label(r1)),
            (prediction("image:2", "query:2", "group:2", abstain=False), r2, label(r2))]
    m = scorer.metrics(assemble(rows), "A5")
    assert m["abstention_rate_all_dev"] == pytest.approx(0.5)
    assert m["coverage_count_eligible"] == pytest.approx(0.5)
    assert m["selective_mae_present"] == 0
    assert m["n_selective_present"] == 1


def test_interval_refuses_small_source_cluster_claim():
    r = roster()
    items = assemble([(prediction(), r, label(r))])
    report = scorer.paired_interval(items, "A4", "A0", "present_mae")
    assert report["interval_95"] is None
    assert report["reason"] == "FEWER_THAN_10_SOURCE_CLUSTERS"


def test_prediction_rejects_nonzero_model_calls_and_gt():
    r = roster()
    p = prediction()
    p["model_calls_in_this_replay"] = 1
    with pytest.raises(ValueError, match="REPLAY_MODEL_CALLS_NOT_ZERO"):
        assemble([(p, r, label(r))])
    p = prediction()
    p["gt_used"] = True
    with pytest.raises(ValueError, match="GT_USED_IN_PREDICTION"):
        assemble([(p, r, label(r))])


def test_runtime_requires_c1_module(monkeypatch):
    original_import = __import__
    def reject(name, *args, **kwargs):
        if name == "countlab.thu_t12_contract":
            raise ImportError("C1 not in this feature worktree")
        return original_import(name, *args, **kwargs)
    import builtins
    monkeypatch.setattr(builtins, "__import__", reject)
    with pytest.raises(ValueError, match="C1_CONTRACT_MODULE_NOT_IN_WORKTREE"):
        scorer.c1_gate([], {})


def test_end_to_end_requires_explicit_hash_bound_approval(tmp_path, monkeypatch):
    r = roster()
    p = prediction()
    y = label(r)
    paths = {name: tmp_path / (name + (".csv" if name == "roster" else ".json"))
             for name in ("predictions", "labels", "roster", "contract", "attestation", "pool_manifest", "policy", "out")}
    paths["predictions"].write_text(json.dumps(p) + "\n", encoding="utf-8")
    paths["labels"].write_text(json.dumps(y) + "\n", encoding="utf-8")
    with paths["roster"].open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(r))
        writer.writeheader()
        writer.writerow(r)
    paths["pool_manifest"].write_text("{}\n", encoding="utf-8")
    paths["policy"].write_text("{}\n", encoding="utf-8")
    contract = {"paired_roster_sha256": scorer.digest(paths["roster"]),
                "frozen_pool_manifest_sha256": scorer.digest(paths["pool_manifest"]),
                "policy_sha256": scorer.digest(paths["policy"]),
                "selection_split": "CAL"}
    paths["contract"].write_text(json.dumps(contract), encoding="utf-8")
    attestation = {"status": "approved_for_dev_scoring",
                   "approved_by": "reviewer", "approval_reference": "test-only",
                   "g0_decision": "GO_FOR_DEV", "g0_evidence_reference": "test-only",
                   "source_disjoint_verified": True, "candidate_ids_attested": True,
                   "source_image_hashes_attested": True,
                   "t12_protocol_sha256": scorer.PROTOCOL_SHA,
                   "split_protocol_id": "split:approved", "frozen_pool_run_id": "pool:approved"}
    for field, file in (("prediction_file_sha256", "predictions"),
                        ("label_file_sha256", "labels"),
                        ("roster_file_sha256", "roster"),
                        ("contract_file_sha256", "contract"),
                        ("pool_manifest_sha256", "pool_manifest"),
                        ("policy_file_sha256", "policy")):
        attestation[field] = scorer.digest(paths[file])
    paths["attestation"].write_text(json.dumps(attestation), encoding="utf-8")
    args = types.SimpleNamespace(**paths)
    monkeypatch.setattr(scorer, "c1_gate", lambda roster, contract:
                        ({"metadata_status": "PASS"}, {"status": "PASS"}))
    scorer.run(args)
    report = json.loads(paths["out"].read_text(encoding="utf-8"))
    assert report["arms"]["A4"]["present_mae"] == 0
    assert report["arms"]["A0"]["present_mae"] == 1
    assert report["team_approval_status"] == "CANDIDATE_NOT_TEAM_APPROVED"
    assert "A2" not in report["arms"]
    with pytest.raises(ValueError, match="OUTPUT_ALREADY_EXISTS"):
        scorer.run(args)
    paths["out"].unlink()
    attestation["roster_file_sha256"] = "f" * 64
    paths["attestation"].write_text(json.dumps(attestation), encoding="utf-8")
    with pytest.raises(ValueError, match="ATTESTATION_HASH_MISMATCH_roster_file_sha256"):
        scorer.run(args)
    assert not paths["out"].exists()


def test_prediction_with_embedded_gt_field_is_refused():
    r = roster()
    p = prediction()
    p["arms"]["A4"]["ground_truth"] = 2
    with pytest.raises(ValueError, match="GT_FIELD_IN_PREDICTION_ground_truth"):
        assemble([(p, r, label(r))])
