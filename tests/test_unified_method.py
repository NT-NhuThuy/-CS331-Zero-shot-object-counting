from copy import deepcopy
import pytest
from countlab.method.pipeline import run_method
from countlab.adapters.fqie_legacy import convert


def p(cid, box, semantic="UNKNOWN", whole="UNKNOWN", view="full", score=.5):
    return {"candidate_id": cid, "box_xyxy": box, "view_id": view,
            "confidence": score, "evidence": {"semantic": semantic, "whole_instance": whole}}


def d(base=None, rescue=None, edges=None):
    base, rescue = base or [], rescue or []
    return {"schema": "countlab_candidate_pool_v1", "image_id": "i", "query_id": "q",
            "query_text": "chair", "baseline": base, "baseline_count": len(base),
            "rescue": rescue, "identity_edges": edges or []}


def test_unchanged_without_evidence():
    out = run_method(d([p("b1", [0,0,10,10]), p("b2", [20,0,30,10])]))
    assert out["count"] == 2 and len(out["unknown_regions"]) == 2
    assert out["count_status"] == "partial"


def test_only_dual_no_rejects_baseline():
    base = [p("a", [0,0,10,10], "NO", "NO"), p("b", [20,0,30,10], "NO", "YES")]
    out = run_method(d(base))
    assert out["count"] == 1
    assert [z["candidate_id"] for z in out["rejected_regions"]] == ["a"]


def test_rescue_unmatched_admitted():
    out = run_method(d([p("a", [0,0,10,10])], [p("r", [40,0,50,10], "YES", "YES", "tile")]))
    assert out["count"] == 2


def test_rescue_match_prevents_duplicate_and_contradiction_restores_baseline():
    base = [p("a", [0,0,10,10], "NO", "NO")]
    rescue = [p("r", [1,0,11,10], "YES", "YES", "tile")]
    out = run_method(d(base, rescue))
    assert out["count"] == 1 and len(out["matched_rescue_to_baseline"]) == 1
    assert len(out["rejected_regions"]) == 0 and "contradictory_evidence" in out["evidence_trace"][-1]["conflicts"][0]


def test_safe_identity_complete_clique_one_instance():
    r1 = p("r1", [30,0,45,15], "YES", "YES", "tile1")
    r2 = p("r2", [31,1,46,16], "YES", "YES", "tile2")
    out = run_method(d(rescue=[r1, r2], edges=[{"a":"r1","b":"r2","relation":"SAME"}]))
    assert out["count"] == 1
    assert out["accepted_instances"][0]["members"] == ["r1", "r2"]


def test_unknown_overlap_does_not_double_count():
    r1 = p("r1", [30,0,45,15], "YES", "YES", "tile1")
    r2 = p("r2", [31,1,46,16], "YES", "YES", "tile2")
    out = run_method(d(rescue=[r1, r2]))
    assert out["count"] == 0 and len(out["unknown_regions"]) == 2
    assert out["count_status"] == "partial" and out["count_upper_bound"] == 2


def test_different_overlapping_rescues_count_separately():
    r1 = p("r1", [30,0,45,15], "YES", "YES", "tile1")
    r2 = p("r2", [31,1,46,16], "YES", "YES", "tile2")
    out = run_method(d(rescue=[r1, r2], edges=[{"a":"r1","b":"r2","relation":"DIFFERENT"}]))
    assert out["count"] == 2


def test_verification_budget_zero_falls_back():
    base = [p("a", [0,0,10,10], "NO", "NO")]
    out = run_method(d(base), budget=0)
    assert out["count"] == 1 and out["unknown_regions"][0]["candidate_id"] == "a"


def test_explicit_abstention_with_fair_evaluation_fallback():
    res = [p("r1", [30,0,45,15], "YES", "YES", "tile1"), p("r2", [31,0,46,15], "YES", "YES", "tile2")]
    out = run_method(d(rescue=res), abstain_on_conflict=True)
    assert out["count"] is None and out["query_abstained"]
    assert out["evaluation_fallback_count"] == 0


def test_inference_rejects_ground_truth_fields():
    inp = d()
    inp["metadata"] = {"GT_count": 7}
    with pytest.raises(ValueError, match="GT/label"):
        run_method(inp)


def test_baseline_count_mismatch_guard():
    inp = d([p("a", [0,0,10,10])])
    inp["baseline_count"] = 20
    with pytest.raises(ValueError, match="baseline_count"):
        run_method(inp)


def test_duplicate_ids_guard():
    with pytest.raises(ValueError, match="unique"):
        run_method(d([p("x", [0,0,10,10])], [p("x", [20,0,30,10])]))


def test_import_legacy_tasks():
    idx = {"schema": "fqie_bidirectional_verification_manifest_v1", "uses_gt": False,
           "tasks": [
               {"task_id":"baseline:1:9","query_id":1,"source_image_id":77,"query_caption":"chair",
                "role":"baseline","box_xyxy":[0,0,10,10],"proposal_score":.8,"evidence_refs":["full_p0"]},
               {"task_id":"rescue:r1","query_id":1,"source_image_id":77,"query_caption":"chair",
                "role":"rescue","box_xyxy":[30,0,40,10],"evidence_refs":["tile"]}]}
    records = [{"candidate_id":"baseline:1:9","semantic":"NO","whole":"NO"},
               {"candidate_id":"rescue:r1","semantic":"YES","whole":"YES"}]
    manifest, = convert(idx, records, hashes={"test":"123"})
    out = run_method(manifest)
    assert out["count"] == 1
    assert manifest["image_id"] == "77"


def test_replay_is_deterministic():
    inp = d([p("a", [0,0,10,10])])
    assert run_method(inp) == run_method(deepcopy(inp))


def test_policy_freeze_validation_and_output_hash():
    inp = d([p("a", [0,0,10,10])])
    pol = {"baseline_rescue_match_iou":.5, "identity_min_same_iou":.1,
           "identity_overlap_ambiguity_iou":.3}
    out = run_method(inp, policy=pol)
    assert out["method_policy"] == pol and len(out["method_policy_sha256"]) == 64
    with pytest.raises(ValueError, match="EXACTLY"):
        run_method(inp, policy={"bad":.5})
    with pytest.raises(ValueError, match="ambiguity"):
        run_method(inp, policy={**pol, "identity_min_same_iou":.8})
