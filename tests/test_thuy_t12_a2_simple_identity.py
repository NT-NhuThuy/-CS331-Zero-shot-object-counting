import copy
import pytest

from countlab.method.t12_a2_simple_identity import (
    run_a2_simple_identity as run,
)


def manifest(items):
    return {
        "baseline_count": len(items),
        "baseline": items,
        "rescue": [],
    }


def candidate(cid, box, score=0.9):
    return {
        "candidate_id": cid,
        "box_xyxy": box,
        "confidence": score,
    }


def event(a, b, relation, ref):
    return {
        "op": "PAIR_IDENTITY",
        "a": a,
        "b": b,
        "relation": relation,
        "evidence_ref": ref,
    }


def test_nms_does_not_invent_same_evidence():
    data = manifest([
        candidate("a", [0, 0, 10, 10]),
        candidate("b", [0, 0, 10, 10]),
    ])
    result = run(data, [], a1_rejected_ids=[])
    assert result["A2_NMS"]["count"] == 1
    assert result["A2_COMPLETE_LINK"]["count"] == 2
    assert result["gt_used"] is False


def test_nontransitive_same_and_unknown_refinement():
    data = manifest([
        candidate("a", [0, 0, 10, 10]),
        candidate("b", [20, 0, 30, 10]),
        candidate("c", [40, 0, 50, 10]),
    ])
    events = [
        event("a", "b", "SAME", "s1"),
        event("b", "c", "SAME", "s2"),
        event("a", "c", "UNKNOWN", "u1"),
    ]
    original = copy.deepcopy(events)
    result = run(data, events, a1_rejected_ids=[])
    assert result["A2_COMPLETE_LINK"]["groups"] == [
        ["a", "b"], ["c"]
    ]
    assert result["A2_COMPLETE_LINK"]["direct_same_evidence_refs"] == ["s1"]
    assert events == original


def test_a1_rejections_and_identity_conflict():
    data = manifest([
        candidate("a", [0, 0, 10, 10]),
        candidate("b", [0, 0, 10, 10]),
        candidate("z", [30, 0, 40, 10]),
    ])
    events = [
        event("a", "b", "SAME", "s"),
        event("a", "b", "DIFFERENT", "d"),
    ]
    result = run(data, events, a1_rejected_ids=["z"])
    assert result["a1_retained_baseline_count"] == 2
    assert result["A2_COMPLETE_LINK"]["count"] == 2
    assert result["A2_COMPLETE_LINK"]["pair_conflicts"] == [["a", "b"]]
    assert result["A2_NMS"]["count"] == 1


def test_invalid_geometry_and_unknown_rejection_fail_closed():
    data = manifest([
        candidate("a", [0, 0, 0, 10]),
        candidate("b", [0, 0, 10, 10]),
    ])
    result = run(data, [], a1_rejected_ids=[])
    assert result["A2_NMS"]["count"] == 2
    assert result["A2_NMS"]["invalid_geometry_retained"] == ["a"]

    with pytest.raises(ValueError):
        run(data, [], a1_rejected_ids=["missing"])
