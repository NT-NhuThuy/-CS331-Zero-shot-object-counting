"""Paired VAL-development count and point evaluation after prediction freeze."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from scipy.optimize import linear_sum_assignment

from target_aware_counting.reference_correspondence.evaluate import load_labels, truth_for
from .native_greedy import COHORTS, ROOT, load_manifest, sha256

SIDECARS = ROOT / "target_aware_counting/output/reference_correspondence_rc01_20261029"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def matched_points(points: list, truth: list, cohort: str, radius: float) -> int:
    if not points or not truth:
        return 0
    if cohort == "lvis_dev300":
        edges = [[(x >= box[0] and x <= box[2] and y >= box[1] and y <= box[3])
                  for box in truth] for x, y in points]
    else:
        edges = [[math.dist(point, target) <= radius for target in truth] for point in points]
    # Max-cardinality assignment: feasible edges cost 0; unmatched cost 1.
    n, m = len(points), len(truth)
    cost = [[0 if edges[i][j] else 2 for j in range(m)] + [1]*n
            for i in range(n)]
    rows, cols = linear_sum_assignment(cost)
    return sum(cols[k] < m and edges[rows[k]][cols[k]] for k in range(len(rows)))


def _metrics(count: int, points: list, truth: list, cohort: str, radius: float,
             status: str) -> dict:
    tp = matched_points(points, truth, cohort, radius) if status == "present" else 0
    return {"count": count, "abs_error": abs(count-len(truth)),
            "squared_error": (count-len(truth))**2, "exact": int(count == len(truth)),
            "tp": tp, "fp": count-tp, "fn": len(truth)-tp,
            "false_zero": int(status == "present" and count == 0),
            "absent_positive": int(status == "verified_absent" and count > 0)}


def _ci(rows: list[dict], method: str = "native", seed: int = 42,
        samples: int = 2000) -> list[float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["source_image_id"])].append(row)
    groups = list(grouped.values())
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        draw = [r for _ in groups for r in groups[rng.randrange(len(groups))]]
        deltas.append(sum(r[method]["abs_error"] - r["m0"]["abs_error"]
                          for r in draw)/len(draw))
    deltas.sort()
    return [deltas[int(.025*samples)], deltas[int(.975*samples)-1]]


def evaluate(cohort: str, predictions: Path, freeze: Path, output: Path,
             radius: float, method_label: str = "native") -> dict:
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError("positive finite point radius required")
    if method_label not in ("native", "capacity_b", "capacity_b_retry", "a_native", "proper_map",
                            "proper_mbr", "guided_map", "guided_mbr"):
        raise ValueError("unsupported method label")
    manifest, items = load_manifest(cohort)
    frozen = json.loads(freeze.read_text())
    if not frozen.get("complete") or frozen.get("labels_opened_before_freeze") is not False or \
            frozen["cohort"] != cohort or frozen["manifest_sha256"] != sha256(manifest) or \
            frozen["predictions_sha256"] != sha256(predictions) or \
            frozen["predictions"] != str(predictions.resolve()):
        raise ValueError("prediction freeze failed or files changed")
    expected_method = ("molmopoint_capacity_b_v2_retry_short_prompt"
                       if method_label == "capacity_b_retry"
                       else "molmopoint_capacity_b_v1" if method_label == "capacity_b"
                       else "molmopoint_native_greedy" if method_label == "native"
                       else "molmopoint_a_native" if method_label == "a_native"
                       else f"molmopoint_a_{method_label}")
    if frozen["contract"].get("method") != expected_method:
        raise ValueError("frozen prediction method differs from requested evaluation label")
    native = {row["item_id"]: row for row in read_rows(predictions)}
    if len(native) != len(items):
        raise ValueError("duplicate or missing native predictions")
    sidecar_path = SIDECARS / f"candidates_{cohort}.jsonl"
    controls = {row["item_id"]: row for row in read_rows(sidecar_path)}
    expected_ids = {item["item_id"] for item in items}
    if len(controls) != len(items) or set(controls) != expected_ids or set(native) != expected_ids:
        raise ValueError("duplicate or missing M0 sidecar items")
    # First label access happens after frozen prediction and denominator checks.
    labels = load_labels(cohort)
    scored = []
    for item in items:
        item_id = item["item_id"]
        pred, control = native[item_id], controls[item_id]
        if pred["status"] != "complete" or pred["count"] != len(pred["points_xy"]) or \
                pred["query"] != item["query"] or pred["source_image_id"] != item["source_image_id"] or \
                control["query"] != item["query"] or control["source_image_id"] != item["source_image_id"]:
            raise ValueError(f"prediction/control identity mismatch: {item_id}")
        by_id = {c["candidate_id"]: c for c in control["candidates"]}
        m0_ids = control["m0_final_ids"]
        if len(by_id) != len(control["candidates"]) or len(set(m0_ids)) != len(m0_ids) or \
                any(cid not in by_id or not by_id[cid]["is_m0"] for cid in m0_ids):
            raise ValueError(f"M0 sidecar invalid: {item_id}")
        boxes = [by_id[cid]["box_xyxy_original"] for cid in m0_ids]
        m0_points = [[(box[0]+box[2])/2, (box[1]+box[3])/2] for box in boxes]
        status, truth = truth_for(cohort, item, labels)
        scored.append({"item_id": item_id, "source_image_id": item["source_image_id"],
                       "status": status, "gt_count": len(truth),
                       method_label: _metrics(pred["count"], pred["points_xy"], truth,
                                              cohort, radius, status),
                       "m0": _metrics(len(m0_ids), m0_points, truth, cohort, radius, status)})
    summary = {"cohort": cohort, "n": len(scored),
               "source_images": len({r["source_image_id"] for r in scored}),
               "status_counts": dict(Counter(r["status"] for r in scored)),
               "localization_metric": "point_inside_gt_box_one_to_one" if cohort == "lvis_dev300"
                                      else f"point_within_{radius:g}px_one_to_one",
               "point_limit": "M0 box centers versus MolmoPoint points; LVIS box IoU50 is separate",
               "m0_sidecar_sha256": sha256(sidecar_path),
               "prediction_sha256": sha256(predictions), "methods": {}}
    for method in ("m0", method_label):
        measures = [r[method] for r in scored]
        summary["methods"][method] = {
            "mae": sum(m["abs_error"] for m in measures)/len(measures),
            "rmse": math.sqrt(sum(m["squared_error"] for m in measures)/len(measures)),
            **{key: sum(m[key] for m in measures)
               for key in ("exact", "tp", "fp", "fn", "false_zero", "absent_positive")},
            "absent_count_mass": sum(r[method]["count"] for r in scored
                                     if r["status"] == "verified_absent"),
        }
    summary[f"{method_label}_delta_mae_vs_m0"] = (
        summary["methods"][method_label]["mae"] - summary["methods"]["m0"]["mae"])
    summary[f"{method_label}_delta_mae_ci95_source_bootstrap"] = _ci(scored, method_label)
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"summary": summary, "per_item": scored}, stream, indent=2)
        stream.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fsc-point-radius-px", type=float, default=16.0)
    parser.add_argument("--method-label", choices=("native", "capacity_b", "capacity_b_retry", "a_native",
                                                   "proper_map", "proper_mbr",
                                                   "guided_map", "guided_mbr"),
                        default="native")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.cohort, args.predictions, args.freeze,
                              args.output, args.fsc_point_radius_px, args.method_label), indent=2))


if __name__ == "__main__":
    main()
