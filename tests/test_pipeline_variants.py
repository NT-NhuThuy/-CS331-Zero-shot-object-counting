from countlab.method.pipeline import run_method


def proposal(cid, box, semantic="YES", whole="YES", view="full"):
    return {
        "candidate_id": cid,
        "box_xyxy": box,
        "view_id": view,
        "evidence": {"semantic": semantic, "whole_instance": whole},
    }


def manifest(edge):
    baseline = [proposal("b", [0, 0, 10, 10])]
    rescue = [proposal("r", [1, 0, 11, 10], view="tile")]
    return {
        "schema": "countlab_candidate_pool_v1",
        "image_id": "variant-image",
        "query_id": "variant-query",
        "query_text": "target",
        "baseline": baseline,
        "baseline_count": len(baseline),
        "rescue": rescue,
        "identity_edges": [{"a": "b", "b": "r", "relation": edge}],
    }


def test_joint_variant_is_explicit_and_does_not_change_v1_default():
    data = manifest("DIFFERENT")
    sequential = run_method(data)
    joint = run_method(data, arbitration="joint_t1", pairwise_budget=1)

    assert sequential["method_variant"] == "sequential_v1"
    assert sequential["count"] == 1
    assert joint["method_variant"] == "joint_t1"
    assert joint["count"] == 2
    assert joint["pairwise_allocation_summary"]["n_collision_pairs"] == 1


def test_joint_variant_preserves_same_and_unknown_safety():
    same = run_method(manifest("SAME"), arbitration="joint_t1", pairwise_budget=1)
    unknown = run_method(
        {**manifest("UNKNOWN"), "identity_edges": []},
        arbitration="joint_t1",
        pairwise_budget=1,
    )

    assert same["count"] == 1
    assert unknown["count"] == 1
    assert unknown["unresolved_candidates"] == [
        {"candidate_id": "r", "reason": "collision_relation_unknown_or_conflict"}
    ]


def test_joint_variant_honors_policy_and_recovery_identity_flags():
    data = manifest("DIFFERENT")
    permissive = run_method(
        data, arbitration="joint_t1", pairwise_budget=1,
        policy={"baseline_rescue_match_iou": .99, "identity_min_same_iou": .1,
                "identity_overlap_ambiguity_iou": .3},
    )
    no_recovery = run_method(data, arbitration="joint_t1", pairwise_budget=1,
                             enable_recovery=False)
    no_identity = run_method(data, arbitration="joint_t1", pairwise_budget=1,
                             enable_identity=False)
    assert permissive["joint_policy"]["collision_iou"] == .99
    assert permissive["count"] == 2
    assert no_recovery["count"] == 1
    assert no_identity["count"] == 1
    assert no_recovery["input_sha256"] == run_method(data, arbitration="joint_t1")["input_sha256"]
    assert no_identity["input_sha256"] == run_method(data, arbitration="joint_t1")["input_sha256"]
