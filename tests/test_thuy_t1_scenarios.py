"""Predeclared synthetic counterexamples; offline truth never enters manifests."""
import pytest

from countlab.method.pipeline import run_method


def p(cid, box, semantic="UNKNOWN", whole="UNKNOWN", view="full"):
    return {"candidate_id": cid, "box_xyxy": box, "view_id": view,
            "evidence": {"semantic": semantic, "whole_instance": whole}}


def m(baseline=(), rescue=(), edges=()):
    return {"schema": "countlab_candidate_pool_v1", "image_id": "synthetic-image",
            "query_id": "synthetic-query", "query_text": "target", "baseline": list(baseline),
            "baseline_count": len(baseline), "rescue": list(rescue), "identity_edges": list(edges)}


def test_s1_spurious_baseline_dual_no_rejects():
    out = run_method(m([p("b", [0, 0, 10, 10], "NO", "NO")]))
    assert out["count"] == 0 and [x["candidate_id"] for x in out["rejected_regions"]] == ["b"]


def test_s2_harmful_unary_rejection_is_visible_offline_not_corrected_inference():
    # Synthetic offline truth is deliberately absent from m(...).
    out = run_method(m([p("true_but_no", [0, 0, 10, 10], "NO", "NO")]))
    assert out["count"] == 0


def test_s3_baseline_rescue_conflict_is_traced_not_blindly_added():
    out = run_method(m([p("b", [0, 0, 10, 10], "NO", "NO")],
                       [p("r", [1, 0, 11, 10], "YES", "YES", "tile")]))
    assert out["count"] == 1 and out["evidence_trace"][-1]["conflicts"]


def test_s4_new_rescue_is_admitted():
    assert run_method(m([p("b", [0, 0, 10, 10])],
                        [p("r", [30, 0, 40, 10], "YES", "YES", "tile")]))["count"] == 2


def test_s5_observed_same_duplicate_is_one_component():
    out = run_method(m([], [p("r1", [0, 0, 10, 10], "YES", "YES", "v1"),
                            p("r2", [1, 0, 11, 10], "YES", "YES", "v2")],
                       [{"a": "r1", "b": "r2", "relation": "SAME"}]))
    assert out["count"] == 1


def test_s6_unsupported_bridge_never_merges_three_candidates():
    rescue = [p("a", [0, 0, 10, 10], "YES", "YES", "v1"),
              p("b", [1, 0, 11, 10], "YES", "YES", "v2"),
              p("c", [2, 0, 12, 10], "YES", "YES", "v3")]
    out = run_method(m([], rescue, [{"a": "a", "b": "b", "relation": "SAME"},
                                    {"a": "b", "b": "c", "relation": "SAME"}]))
    assert out["count"] == 0 and len(out["unknown_regions"]) == 3


def test_s7_adjacent_different_candidates_are_not_iou_merged():
    out = run_method(m([], [p("a", [0, 0, 10, 10], "YES", "YES", "v1"),
                            p("b", [1, 0, 11, 10], "YES", "YES", "v2")],
                       [{"a": "a", "b": "b", "relation": "DIFFERENT"}]))
    assert out["count"] == 2


def test_s8_absent_distractor_exposes_unary_false_promotion_risk():
    # It counts under v1; whether it is wrong is evaluator-only truth.
    assert run_method(m([p("distractor", [0, 0, 10, 10], "YES", "YES")]))["count"] == 1


def test_s9_timeout_is_unknown_not_no():
    out = run_method(m([p("timeout", [0, 0, 10, 10])]))
    assert out["count"] == 1 and [x["candidate_id"] for x in out["unknown_regions"]] == ["timeout"]


def test_s10_cross_pool_edge_is_rejected_by_schema():
    with pytest.raises(ValueError, match="identity edge invalid"):
        run_method(m([p("b", [0, 0, 10, 10], "YES", "YES")], [],
                     [{"a": "b", "b": "different-query", "relation": "SAME"}]))
