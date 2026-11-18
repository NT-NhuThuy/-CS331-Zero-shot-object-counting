"""Offline-only evaluator for matched T1 batch replay.

This module accepts reviewed annotations only after each arm has been replayed
on the same GT-blind candidate manifests. It never feeds labels into routing,
pairwise allocation, arbitration, or policy selection.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .baselines import ARM_IDS
from .schema import parse_input
from .t1_comparison import run_matched_t1_comparison
from .t1_replay import evaluate_count_annotation, evaluate_instance_annotations


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _partition(output: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    baseline, rescues, _ = parse_input(manifest)
    raw = output.get("predicted_components")
    if raw is None:
        raw = [[candidate_id] for candidate_id in output.get("accepted_candidate_ids", [])]
    return {
        "image_id": manifest["image_id"], "query_id": manifest["query_id"],
        "candidate_universe_ids": sorted(x.candidate_id for x in baseline + rescues),
        "predicted_components": raw,
    }


def _count_prediction(output: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return {**output, "image_id": manifest["image_id"], "query_id": manifest["query_id"]}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty evaluation rows")
    public = [row for row in rows if row["public_abs_error"] is not None]
    return {
        "n_pairs": len(rows),
        "public_evaluable_pairs": len(public),
        "public_mae": sum(row["public_abs_error"] for row in public) / len(public) if public else None,
        "fallback_mae": sum(row["fallback_abs_error"] for row in rows) / len(rows),
        "public_exact": sum(row["public_exact"] for row in public) / len(public) if public else None,
        "fallback_exact": sum(row["fallback_exact"] for row in rows) / len(rows),
        "abstained_pairs": sum(row["query_abstained"] for row in rows),
        "false_merge": sum(row["identity"]["pair_false_merge"] for row in rows),
        "false_split": sum(row["identity"]["pair_false_split"] for row in rows),
        "missed_same": sum(row["identity"]["pair_missed_same"] for row in rows),
        "reviewed_candidate_pairs": sum(row["identity"]["n_scored_candidates"] for row in rows),
    }


def evaluate_batch(*, index_path: Path, annotations_index_path: Path, config_path: Path) -> dict[str, Any]:
    index = _read(index_path)
    annotations_index = _read(annotations_index_path)
    config = _read(config_path)
    if index.get("schema") != "countlab_t1_manifest_index_v1":
        raise ValueError("unsupported manifest index")
    if annotations_index.get("schema") != "countlab_t1_annotation_index_v1":
        raise ValueError("unsupported annotation index")
    if config.get("schema") != "countlab_t1_experiment_config_v1" or config.get("claims_allowed") is not False:
        raise ValueError("evaluation requires a no-claim T1 config")
    manifest_paths = [index_path.parent / item for item in index.get("manifest_paths", [])]
    if not manifest_paths or len(manifest_paths) != len(set(manifest_paths)):
        raise ValueError("manifest index must contain unique non-empty paths")
    annotation_records = annotations_index.get("annotations", [])
    if not isinstance(annotation_records, list) or not annotation_records:
        raise ValueError("annotation index must contain a non-empty annotations list")
    annotation_by_key = {
        (str(item["image_id"]), str(item["query_id"])): annotations_index_path.parent / item["path"]
        for item in annotation_records
    }
    if len(annotation_by_key) != len(annotation_records):
        raise ValueError("annotation index contains duplicate image/query keys")
    arms = tuple(config.get("arms", ()))
    if not arms or any(arm not in ARM_IDS for arm in arms):
        raise ValueError("config arms must be registered B0-B9")
    metric_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest_path in manifest_paths:
        manifest = _read(manifest_path)
        key = (str(manifest["image_id"]), str(manifest["query_id"]))
        annotation_path = annotation_by_key.get(key)
        if annotation_path is None:
            raise ValueError(f"missing annotation for image/query: {key}")
        annotation = _read(annotation_path)
        comparison = run_matched_t1_comparison(
            manifest, budget=config.get("verification_budget"),
            pairwise_budget=config.get("pairwise_verification_budget"),
            policy=config.get("policy"), arms=arms)
        outputs = {**comparison["arms"],
                   "JOINT_T1": comparison["experimental_joint"],
                   "GEOMETRY_CONTROL": comparison["pairwise_geometry_control"],
                   "RISK_CONTROL": comparison["pairwise_risk_control"],}
        for name, output in outputs.items():
            count = evaluate_count_annotation(_count_prediction(output, manifest), annotation)
            identity = evaluate_instance_annotations(_partition(output, manifest), annotation)
            metric_rows[name].append({"count": count, "identity": identity})
    return {
        "schema": "countlab_t1_batch_offline_evaluation_v1",
        "status": "OFFLINE_EVALUATED_NO_POLICY_SELECTION",
        "uses_gt_for_inference_or_selection": False,
        "metrics_by_arm": {
            name: _aggregate([{"public_abs_error": row["count"]["public_abs_error"],
                               "fallback_abs_error": row["count"]["fallback_abs_error"],
                               "public_exact": row["count"]["public_exact"],
                               "fallback_exact": row["count"]["fallback_exact"],
                               "query_abstained": row["count"]["query_abstained"],
                               "identity": row["identity"]} for row in rows])
            for name, rows in sorted(metric_rows.items())
        },
        "limitations": [
            "Annotations are joined only after GT-blind replay for every arm",
            "No policy, threshold, or arm selection is performed",
            "Metrics cover reviewed candidate universes, not full-scene recall",
        ],
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--annotations-index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_batch(index_path=args.index, annotations_index_path=args.annotations_index,
                            config_path=args.config)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
