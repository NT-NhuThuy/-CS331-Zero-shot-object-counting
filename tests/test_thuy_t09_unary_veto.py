import pytest

from countlab.method.c3_rescue_arbitration import reconcile_rescues as original
from countlab.method.c3_rescue_arbitration_unary_veto import reconcile_rescues as patched


def unary(cid, op, verdict, ref):
    return {
        "op": op,
        "candidate_id": cid,
        "verdict": verdict,
        "evidence_ref": ref,
    }


def pair(a, b, relation, ref):
    return {
        "op": "PAIR_IDENTITY",
        "a": a,
        "b": b,
        "relation": relation,
        "evidence_ref": ref,
    }


def scenario(name):
    positive = [
        unary("r1", "REGION_SEMANTIC", "YES", "s-r1"),
        unary("r1", "REGION_WHOLE", "YES", "w-r1"),
    ]
    different = pair("r1", "b", "DIFFERENT", "d-r1-b")
    rescues = ["r1"]

    if name == "VALID_DIRECT_ADMISSION":
        events = positive + [different]
    elif name == "NEGATIVE_WITHIN_SAME_GROUP":
        rescues.append("r2")
        events = positive + [
            unary("r2", "REGION_SEMANTIC", "NO", "s-r2-no"),
            pair("r1", "r2", "SAME", "same-r1-r2"),
            different,
        ]
    elif name == "NO_EVIDENCE_FALLBACK":
        events = []
    elif name == "SAME_RETAINED_BASELINE":
        events = positive + [
            pair("r1", "b", "SAME", "same-r1-b"),
        ]
    elif name == "CONFLICTING_UNARY_RETRY":
        rescues.append("r2")
        events = positive + [
            unary("r2", "REGION_SEMANTIC", "YES", "s-r2-yes"),
            unary("r2", "REGION_SEMANTIC", "NO", "s-r2-no"),
            pair("r1", "r2", "SAME", "same-r1-r2"),
            different,
        ]
    else:
        raise AssertionError(name)

    manifest = {
        "baseline": [{"candidate_id": "b"}],
        "rescue": [{"candidate_id": cid} for cid in rescues],
    }
    return manifest, events


@pytest.mark.parametrize(
    "name,old_count,new_count",
    [
        ("VALID_DIRECT_ADMISSION", 2, 2),
        ("NEGATIVE_WITHIN_SAME_GROUP", 2, 1),
        ("NO_EVIDENCE_FALLBACK", 1, 1),
        ("SAME_RETAINED_BASELINE", 1, 1),
        ("CONFLICTING_UNARY_RETRY", 2, 1),
    ],
)
def test_unary_veto_scenarios(name, old_count, new_count):
    manifest, events = scenario(name)
    shadow = {"rejected": []}

    before = original(manifest, events, shadow)
    after = patched(manifest, events, shadow)

    assert before["operational_count"] == old_count
    assert after["operational_count"] == new_count

    assert before["search_exact"]
    assert after["search_exact"]

    if name in {
        "NEGATIVE_WITHIN_SAME_GROUP",
        "CONFLICTING_UNARY_RETRY",
    }:
        assert len(before["admitted_rescue_groups"]) == 1
        assert len(after["admitted_rescue_groups"]) == 0

    if name == "VALID_DIRECT_ADMISSION":
        assert len(after["admitted_rescue_groups"]) == 1
