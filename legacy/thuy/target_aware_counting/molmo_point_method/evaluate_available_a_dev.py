"""Exploratory paired evaluation of completed A rows available at invocation.

This is a development snapshot, not the frozen full-cohort evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from target_aware_counting.reference_correspondence.evaluate import load_labels, truth_for

from .evaluate import SIDECARS, _metrics, read_rows
from .export_a_predictions import METHODS, prediction_for_method
from .native_greedy import COHORTS, load_manifest, sha256


def evaluate_available(cohort: str, source: Path) -> dict:
    manifest, items = load_manifest(cohort)
    expected = {item["item_id"]: item for item in items}
    source_hash = sha256(source)
    rows = read_rows(source)
    if not rows or len(rows) != len({row["item_id"] for row in rows}):
        raise ValueError("empty source or duplicate item")
    contract = rows[0]["contract"]
    if contract.get("method") != "molmopoint_a_cached_beam_v1" or \
            contract.get("cohort") != cohort or contract.get("manifest_sha256") != sha256(manifest):
        raise ValueError("source contract mismatch")
    for row in rows:
        item = expected.get(row["item_id"])
        if item is None or row["contract"] != contract or \
                row["source_image_id"] != item["source_image_id"] or \
                row["query"] != item["query"]:
            raise ValueError("source identity or contract mismatch")
    sidecar = SIDECARS / f"candidates_{cohort}.jsonl"
    controls = {row["item_id"]: row for row in read_rows(sidecar)}
    if len(controls) != len(items) or set(controls) != set(expected):
        raise ValueError("M0 sidecar denominator mismatch")
    # Open labels only after snapshot identity and hash checks.
    labels = load_labels(cohort)
    totals = {name: [] for name in ("m0", *METHODS)}
    statuses = Counter()
    per_item = []
    for row in rows:
        item = expected[row["item_id"]]
        status, truth = truth_for(cohort, item, labels)
        statuses[status] += 1
        control = controls[row["item_id"]]
        if control["query"] != item["query"] or \
                control["source_image_id"] != item["source_image_id"]:
            raise ValueError("M0 identity mismatch")
        by_id = {candidate["candidate_id"]: candidate
                 for candidate in control["candidates"]}
        ids = control["m0_final_ids"]
        if len(by_id) != len(control["candidates"]) or len(ids) != len(set(ids)) or \
                any(cid not in by_id or not by_id[cid]["is_m0"] for cid in ids):
            raise ValueError("invalid M0 candidates")
        boxes = [by_id[cid]["box_xyxy_original"] for cid in ids]
        m0_points = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                     for box in boxes]
        metrics = {"m0": _metrics(len(ids), m0_points, truth, cohort, 16.0, status)}
        for method in METHODS:
            pred = prediction_for_method(row, method)
            metrics[method] = _metrics(pred["count"], pred["points_xy"],
                                       truth, cohort, 16.0, status)
        metrics = {name: {key: int(value) if hasattr(value, "item") else value
                          for key, value in measure.items()}
                   for name, measure in metrics.items()}
        for name, value in metrics.items():
            totals[name].append(value)
        per_item.append({"item_id": row["item_id"], "source_image_id": row["source_image_id"],
                         "status": status, "gt_count": len(truth), "methods": metrics})
    if sha256(source) != source_hash:
        raise ValueError("source changed during evaluation; retry snapshot")
    summary = {}
    for name, values in totals.items():
        summary[name] = {
            "mae": sum(v["abs_error"] for v in values) / len(values),
            "rmse": math.sqrt(sum(v["squared_error"] for v in values) / len(values)),
            **{key: sum(v[key] for v in values)
               for key in ("exact", "tp", "fp", "fn", "false_zero", "absent_positive")},
        }
    return {"provisional": True, "cohort": cohort, "available_n": len(rows),
            "cohort_n": len(items), "missing_item_ids": sorted(set(expected) -
                                                             {row["item_id"] for row in rows}),
            "source": str(source.resolve()), "source_sha256": source_hash,
            "m0_sidecar_sha256": sha256(sidecar), "status_counts": dict(statuses),
            "methods": summary, "per_item": per_item}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_available(args.cohort, args.source)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_item"}, indent=2))


if __name__ == "__main__":
    main()
