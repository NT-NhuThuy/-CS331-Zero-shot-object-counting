import copy
import pytest

from countlab.method.t09_set_nonmatch_contract import audit_set_nonmatch, universe_sha256

IMAGE_HASH = "a" * 64


def setup():
    m = {
        "image_id": "img", "query_id": "q", "query_text": "apple .",
        "baseline": [{"candidate_id": "b1", "box_xyxy": [1, 1, 3, 3]},
                     {"candidate_id": "b2", "box_xyxy": [5, 5, 8, 8]}],
        "rescue": [{"candidate_id": "r1", "box_xyxy": [9, 9, 10, 10]}],
    }
    events = [
        {"op": "REGION_SEMANTIC", "candidate_id": "r1", "verdict": "YES", "evidence_ref": "s"},
        {"op": "REGION_WHOLE", "candidate_id": "r1", "verdict": "YES", "evidence_ref": "w"},
    ]
    h = universe_sha256(m, IMAGE_HASH)
    obs = [{"op": "SET_IDENTITY", "rescue_candidate_id": "r1",
            "baseline_candidate_ids": ["b1", "b2"], "verdict": "NONE_MATCH",
            "evidence_ref": "set1", "universe_sha256": h,
            "source_image_sha256": IMAGE_HASH, "query_id": "q"}]
    return m, events, obs


def run(m, events, obs, gated=frozenset({"s", "w", "set1"})):
    return audit_set_nonmatch(m, events, obs, source_image_sha256=IMAGE_HASH,
                              gated_evidence_refs=gated)


def test_no_operational_count_promotion_and_complete_membership():
    m, events, obs = setup()
    out = run(m, events, obs)
    assert out["forced_count"] is None
    assert not out["operational_count_changed"]
    assert out["rescue_audits"][0]["conditional_set_separation_supported"]


def test_missing_baseline_member_fails_closed():
    m, events, obs = setup()
    obs[0]["baseline_candidate_ids"] = ["b1"]
    out = run(m, events, obs)
    assert not out["rescue_audits"][0]["conditional_set_separation_supported"]
    assert out["rescue_audits"][0]["uncovered_baseline_candidate_ids"] == ["b2"]


def test_ungated_evidence_fails_closed():
    m, events, obs = setup()
    out = run(m, events, obs, gated=frozenset({"s", "w"}))
    assert not out["rescue_audits"][0]["conditional_set_separation_supported"]
    assert out["evidence_provenance_issues"] == ["UNGATED_SET_EVENT:set1"]


def test_universe_tampering_rejected():
    m, events, obs = setup()
    m["baseline"][0]["box_xyxy"][0] += 1
    with pytest.raises(ValueError, match="SET_EVIDENCE_UNIVERSE_MISMATCH"):
        run(m, events, obs)


def test_explicit_same_vetoes_group_nonmatch():
    m, events, obs = setup()
    events.append({"op": "PAIR_IDENTITY", "a": "r1", "b": "b1",
                   "relation": "SAME", "evidence_ref": "same"})
    out = run(m, events, obs, gated=frozenset({"s", "w", "set1", "same"}))
    assert not out["rescue_audits"][0]["conditional_set_separation_supported"]


def test_explicit_some_match_vetoes_none_match():
    m, events, obs = setup()
    extra = copy.deepcopy(obs[0]); extra["evidence_ref"] = "set2"; extra["verdict"] = "SOME_MATCH"
    obs.append(extra)
    out = run(m, events, obs, gated=frozenset({"s", "w", "set1", "set2"}))
    assert not out["rescue_audits"][0]["conditional_set_separation_supported"]


def test_unary_conflicting_observation_fails_closed():
    m, events, obs = setup()
    events.append({"op": "REGION_WHOLE", "candidate_id": "r1", "verdict": "NO", "evidence_ref": "w_no"})
    out = run(m, events, obs, gated=frozenset({"s", "w", "w_no", "set1"}))
    assert not out["rescue_audits"][0]["conditional_set_separation_supported"]


def test_oversized_group_rejected():
    m, events, obs = setup()
    with pytest.raises(ValueError, match="UNBOUND_SET_MEMBERS"):
        audit_set_nonmatch(m, events, obs, source_image_sha256=IMAGE_HASH,
                           gated_evidence_refs={"s", "w", "set1"}, max_group_size=1)
