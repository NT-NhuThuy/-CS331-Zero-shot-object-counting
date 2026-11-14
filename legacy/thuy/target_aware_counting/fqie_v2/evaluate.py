"""Post-freeze FSC development evaluator; never imported by inference stages."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .gpu import load_manifest
from .run import sha, write_once

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "data/fscd147/countgd_val_fscd147.json"
GT_SHA = "6f9bc8d200d74789aa515e0131fd93275713392c64c3124828bdf258858ec807"


def match_count(boxes: list[list[float]], points: list[tuple[float, float]]) -> int:
    edges = [[j for j, (x, y) in enumerate(points)
              if b[0] <= x <= b[2] and b[1] <= y <= b[3]] for b in boxes]
    owner = {}

    def augment(i: int, seen: set[int]) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    return sum(augment(i, set()) for i in range(len(boxes)))


def summarize(rows: list[dict], mode: str) -> dict:
    present = [x for x in rows if x["cohort"] == "primary"]
    absent = [x for x in rows if x["cohort"] == "absent"]
    if not present or not absent:
        raise RuntimeError("missing cohort")
    errors = [x[f"{mode}_count"] - x["gt_count"] for x in present]
    tp = sum(x[f"{mode}_tp"] for x in present)
    return {"primary_mae": sum(abs(e) for e in errors) / len(errors),
            "primary_rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
            "primary_tp": tp,
            "primary_fp": sum(x[f"{mode}_count"] for x in present) - tp,
            "primary_fn": sum(x["gt_count"] for x in present) - tp,
            "primary_false_zero": sum(x[f"{mode}_count"] == 0 for x in present),
            "absent_mass": sum(x[f"{mode}_count"] for x in absent),
            "absent_active_queries": sum(x[f"{mode}_count"] > 0 for x in absent)}


def bootstrap(rows: list[dict], seed: int = 20261104, reps: int = 2000) -> dict:
    sources = defaultdict(list)
    for row in rows:
        sources[row["source_image_id"]].append(row)
    keys = sorted(sources)
    rng = random.Random(seed)
    names = ("primary_mae", "primary_rmse", "primary_tp", "primary_false_zero",
             "absent_mass", "absent_active_queries")
    deltas = {name: [] for name in names}
    for _ in range(reps):
        sample = [row for key in rng.choices(keys, k=len(keys)) for row in sources[key]]
        if not any(x["cohort"] == "primary" for x in sample) or not any(x["cohort"] == "absent" for x in sample):
            continue
        base, final = summarize(sample, "a4"), summarize(sample, "final")
        for name in names:
            deltas[name].append(final[name] - base[name])
    result = {}
    for name, values in deltas.items():
        values.sort()
        if not values:
            raise RuntimeError("bootstrap produced no valid resamples")
        result[name] = [values[int(.025 * (len(values) - 1))], values[int(.975 * (len(values) - 1))]]
    return result


def evaluate(manifest_path: Path, freeze_path: Path, evidence_path: Path,
             prediction_path: Path, destination: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    candidates = load_manifest(manifest_path)
    frozen = json.loads(freeze_path.read_text())
    if frozen.get("status") != "FROZEN_BEFORE_LABEL_JOIN" or frozen["manifest_sha256"] != sha(manifest_path):
        raise RuntimeError("inference evidence was not frozen before evaluation")
    pred = json.loads(prediction_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    if evidence.get("gpu_freeze_sha256") != sha(freeze_path) or evidence.get("manifest_sha256") != sha(manifest_path):
        raise RuntimeError("assembled evidence/freeze binding mismatch")
    if (pred["status"] != "FROZEN_REMOVE_ONLY_DECISION" or
            pred["input_sha256"]["candidates"] != sha(manifest_path) or
            pred["input_sha256"]["evidence"] != sha(evidence_path)):
        raise RuntimeError("prediction/manifest binding mismatch")
    if sha(GT) != GT_SHA:
        raise RuntimeError("FSC VAL annotation hash mismatch")
    all_ids = {x["candidate_id"] for x in candidates}
    kept = set(pred["candidate_ids"])
    if not kept <= all_ids or pred["count"] != len(kept) or pred["base_count"] != len(all_ids):
        raise RuntimeError("final count or subset invariant violated")
    by_item = defaultdict(list)
    for c in candidates:
        by_item[c["item_id"]].append(c)
    gt = json.loads(GT.read_text())
    points = defaultdict(list)
    for ann in gt["annotations"]:
        x, y, w, h = ann["bbox"]
        points[str(ann["image_id"])].append((x + w / 2, y + h / 2))
    rows = []
    for item in manifest["query_items"]:
        cid = item["item_id"]
        subset = by_item[cid]
        primary = item["base_mode"] == "A4_FROZEN"
        if primary and len(subset) != item["a4_count"]:
            raise RuntimeError("A4 count parity mismatch")
        if not primary and len(subset) != item["m0_count"]:
            raise RuntimeError("M0 count parity mismatch")
        gt_points = points[item["source_image_id"]] if primary else []
        if primary and not gt_points:
            raise RuntimeError(f"missing FSC primary points: {cid}")
        row = {"item_id": cid, "source_image_id": item["source_image_id"],
               "cohort": "primary" if primary else "absent", "gt_count": len(gt_points)}
        for mode in ("m0", "a4", "final"):
            boxes = [c["box_xyxy"] for c in subset
                     if (mode != "m0" or c["branch"] == "M0") and (mode != "final" or c["candidate_id"] in kept)]
            row[f"{mode}_count"] = len(boxes)
            row[f"{mode}_tp"] = match_count(boxes, gt_points) if primary else 0
        rows.append(row)
    if len(rows) != 211:
        raise RuntimeError("query denominator mismatch")
    result = {"status": "FSC_VAL_DEVELOPMENT_POST_FREEZE", "denominators": manifest["counts"],
              "metrics": {mode: summarize(rows, mode) for mode in ("m0", "a4", "final")},
              "final_minus_a4_source_bootstrap_95ci": bootstrap(rows),
              "input_sha256": {"manifest": sha(manifest_path), "freeze": sha(freeze_path),
                               "evidence": sha(evidence_path),
                               "prediction": sha(prediction_path), "fsc_val_gt": GT_SHA},
              "limits": "FSC point containment proxy; absent111 and primary100 are development, not independent confirmation."}
    write_once(destination / "evaluation.json", result)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    csv_path = destination / "per_query.csv"
    data = buffer.getvalue().encode()
    if csv_path.exists() and csv_path.read_bytes() != data:
        raise RuntimeError("locked CSV differs")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(data)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ("manifest", "freeze", "evidence", "prediction", "destination"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    print(json.dumps(evaluate(a.manifest, a.freeze, a.evidence, a.prediction, a.destination)["metrics"], indent=2))


if __name__ == "__main__":
    main()
