"""Create-only CLI for matched T1 frozen-evidence replay and GT-separated audit.

Inference/replay inputs contain only candidate and evidence manifests. Optional
instance annotations are read in a separate offline evaluator stage and are
never passed to any decision arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .baselines import ARM_IDS
from .schema import parse_input
from .t1_comparison import run_matched_t1_comparison


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_instance_annotations(prediction: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    """Evaluate predicted candidate partitions against reviewed instance IDs.

    Annotation schema: ``{"schema":"countlab_t1_instance_truth_v1",
    "image_id":..., "query_id":..., "truth_by_candidate":{candidate_id: instance_id}}``.
    Unannotated candidates are excluded from pair metrics and listed.
    """
    if annotation.get("schema") != "countlab_t1_instance_truth_v1":
        raise ValueError("unsupported offline identity annotation schema")
    if (str(prediction.get("image_id")) != str(annotation.get("image_id")) or
            str(prediction.get("query_id")) != str(annotation.get("query_id"))):
        raise ValueError("prediction and annotation image/query identity mismatch")
    truth = annotation.get("truth_by_candidate")
    if not isinstance(truth, dict):
        raise ValueError("truth_by_candidate must be an object")
    components = prediction.get("predicted_components")
    if not isinstance(components, list):
        raise ValueError("prediction must contain predicted_components")
    flat = [str(candidate_id) for component in components for candidate_id in component]
    if len(flat) != len(set(flat)):
        raise ValueError("predicted candidate appears in multiple components")
    universe_raw = prediction.get("candidate_universe_ids", flat)
    if not isinstance(universe_raw, list) or any(not isinstance(candidate_id, str) for candidate_id in universe_raw):
        raise ValueError("candidate_universe_ids must be a list of strings")
    universe = sorted(set(universe_raw))
    if any(candidate_id not in universe for candidate_id in flat):
        raise ValueError("predicted component contains a candidate outside its universe")
    component_of = {str(candidate_id): index for index, component in enumerate(components)
                    for candidate_id in component}
    scored = sorted(candidate_id for candidate_id in universe if candidate_id in truth)
    accepted_scored = sorted(candidate_id for candidate_id in flat if candidate_id in truth)
    unscored = sorted(candidate_id for candidate_id in universe if candidate_id not in truth)
    omitted = sorted(candidate_id for candidate_id in scored if candidate_id not in component_of)
    true_same = false_merge = false_split = missed_same = true_separate = 0
    for i, left in enumerate(scored):
        for right in scored[i + 1:]:
            actual_same = truth[left] == truth[right]
            left_component = component_of.get(left)
            right_component = component_of.get(right)
            if actual_same:
                if left_component is None or right_component is None:
                    missed_same += 1
                elif left_component == right_component:
                    true_same += 1
                else:
                    false_split += 1
            elif left_component is not None and left_component == right_component:
                false_merge += 1
            else:
                true_separate += 1
    predicted_same = true_same + false_merge
    actual_same = true_same + false_split
    return {
        "schema": "countlab_t1_instance_metrics_v1",
        "n_candidate_universe": len(universe), "n_predicted_candidates": len(flat),
        "n_scored_candidates": len(scored), "n_accepted_scored_candidates": len(accepted_scored),
        "n_unscored_candidates": len(unscored), "unscored_candidate_ids": unscored,
        "omitted_reviewed_candidate_ids": omitted,
        "n_reviewed_instances_observed": len({truth[candidate_id] for candidate_id in scored}),
        "pair_true_same": true_same, "pair_false_merge": false_merge,
        "pair_false_split": false_split, "pair_missed_same": missed_same,
        "pair_true_separate": true_separate,
        "pair_precision": true_same / predicted_same if predicted_same else None,
        "pair_recall": true_same / actual_same if actual_same else None,
        "instance_candidate_coverage": len({truth[candidate_id] for candidate_id in accepted_scored}) /
            len({truth[candidate_id] for candidate_id in scored}) if scored else None,
        "scope": "reviewed candidate universe only; not full-scene recall or count bounds",
    }


def evaluate_count_annotation(prediction: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    """Evaluate point count offline without changing the inference decision.

    The annotation is intentionally separate from ``countlab_candidate_pool_v1``.
    If an arm abstains, both its public count (``None``) and conservative
    ``evaluation_fallback_count`` are reported; callers must not silently turn
    fallback into the claimed count.
    """
    if annotation.get("schema") != "countlab_t1_instance_truth_v1":
        raise ValueError("unsupported offline count annotation schema")
    if (str(prediction.get("image_id")) != str(annotation.get("image_id")) or
            str(prediction.get("query_id")) != str(annotation.get("query_id"))):
        raise ValueError("prediction and annotation image/query identity mismatch")
    truth = annotation.get("ground_truth_count")
    if not isinstance(truth, int) or isinstance(truth, bool) or truth < 0:
        raise ValueError("ground_truth_count must be a non-negative integer")
    # Baseline replay arms predate the unified output field and expose only
    # ``count``; for a non-abstaining arm that is the same fallback value.
    fallback = prediction.get("evaluation_fallback_count", prediction.get("count"))
    if not isinstance(fallback, int) or isinstance(fallback, bool) or fallback < 0:
        raise ValueError("prediction must contain a non-negative integer evaluation_fallback_count")
    public = prediction.get("count")
    if public is not None and (not isinstance(public, int) or isinstance(public, bool) or public < 0):
        raise ValueError("prediction count must be None or a non-negative integer")
    public_abs_error = None if public is None else abs(public - truth)
    fallback_abs_error = abs(fallback - truth)
    return {
        "schema": "countlab_t1_count_metrics_v1",
        "ground_truth_count": truth,
        "predicted_count": public,
        "evaluation_fallback_count": fallback,
        "query_abstained": public is None or bool(prediction.get("query_abstained", False)),
        "public_abs_error": public_abs_error,
        "fallback_abs_error": fallback_abs_error,
        "public_exact": None if public is None else public == truth,
        "fallback_exact": fallback == truth,
        "scope": "offline point-count metric; does not establish candidate recall or identity correctness",
    }


def _prediction_partition(output: dict[str, Any], universe: list[str], *, image_id: Any, query_id: Any) -> dict[str, Any]:
    """Normalize arm output to an explicit partition without inferring identity.

    Arms that only suppress a candidate through NMS retain singleton accepted
    components: suppression is a selection action, not proof of SAME identity.
    """
    raw_components = output.get("predicted_components", output.get("accepted_components"))
    if raw_components is None:
        raw_components = [[candidate_id] for candidate_id in output.get("accepted_candidate_ids", [])]
    return {"image_id": image_id, "query_id": query_id,
            "candidate_universe_ids": universe, "predicted_components": raw_components}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def replay(*, manifest_path: Path, config_path: Path, out_dir: Path,
           annotation_path: Path | None = None) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"create-only output exists: {out_dir}")
    config = _read_json(config_path)
    if config.get("schema") != "countlab_t1_experiment_config_v1":
        raise ValueError("unsupported T1 experiment config")
    if config.get("claims_allowed") is not False:
        raise ValueError("replay config cannot enable scientific claims")
    manifest = _read_json(manifest_path)
    arms = tuple(config.get("arms", ()))
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARM_IDS for arm in arms):
        raise ValueError("config arms must be unique registered B0-B9 identifiers")
    replay_result = run_matched_t1_comparison(
        manifest, budget=config.get("verification_budget"),
        pairwise_budget=config.get("pairwise_verification_budget"),
        policy=config.get("policy"), arms=arms)
    result = {
        "schema": "countlab_t1_replay_artifact_v1",
        "status": "FROZEN_EVIDENCE_REPLAY_NO_CLAIM",
        "uses_gt_for_replay": False,
        "manifest_sha256": sha256(manifest_path), "config_sha256": sha256(config_path),
        "input_sha256": replay_result["input_sha256"],
        "replay": replay_result,
    }
    if annotation_path is not None:
        annotation = _read_json(annotation_path)
        # The annotation is joined only after every arm has been computed.
        # This evaluator never feeds labels back into a method or its policy.
        baseline, rescues, _ = parse_input(manifest)
        universe = sorted(candidate.candidate_id for candidate in baseline + rescues)
        def partition(output: dict[str, Any]) -> dict[str, Any]:
            return _prediction_partition(output, universe, image_id=manifest["image_id"],
                                         query_id=manifest["query_id"])

        def count_prediction(output: dict[str, Any]) -> dict[str, Any]:
            # Older B0--B9 replay outputs do not carry image/query metadata;
            # attach only the immutable manifest identity for offline checking.
            return {**output, "image_id": manifest["image_id"], "query_id": manifest["query_id"]}

        result["offline_evaluation"] = {
            "annotation_sha256": sha256(annotation_path),
            "metrics_by_arm": {
                arm: evaluate_instance_annotations(
                    partition(output), annotation)
                for arm, output in replay_result["arms"].items()
            },
            "joint_metrics": evaluate_instance_annotations(
                partition(replay_result["experimental_joint"]), annotation),
            "geometry_control_metrics": evaluate_instance_annotations(
                partition(replay_result["pairwise_geometry_control"]), annotation),
            "risk_control_metrics": evaluate_instance_annotations(
                partition(replay_result["pairwise_risk_control"]), annotation),
            "count_metrics_by_arm": {
                arm: evaluate_count_annotation(count_prediction(output), annotation)
                for arm, output in replay_result["arms"].items()
            },
            "joint_count_metrics": evaluate_count_annotation(
                count_prediction(replay_result["experimental_joint"]), annotation),
            "geometry_control_count_metrics": evaluate_count_annotation(
                count_prediction(replay_result["pairwise_geometry_control"]), annotation),
            "risk_control_count_metrics": evaluate_count_annotation(
                count_prediction(replay_result["pairwise_risk_control"]), annotation),
            "uses_gt_for_inference_or_selection": False,
        }
        result["status"] = "OFFLINE_EVALUATED_NO_POLICY_SELECTION"
    out_dir.mkdir(parents=True, exist_ok=False)
    target = out_dir / "t1_replay.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    lock = {"schema": "countlab_t1_replay_lock_v1", "status": result["status"],
            "manifest_sha256": result["manifest_sha256"], "config_sha256": result["config_sha256"],
            "output_sha256": sha256(target), "uses_gt_for_replay": False,
            "annotation_sha256": result.get("offline_evaluation", {}).get("annotation_sha256")}
    with (out_dir / "replay_lock.json").open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2, sort_keys=True); stream.write("\n")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="one GT-blind frozen candidate manifest")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, help="optional offline-only candidate instance truth")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(manifest_path=args.manifest, config_path=args.config,
                            out_dir=args.out_dir, annotation_path=args.annotations), indent=2))


if __name__ == "__main__":
    main()
