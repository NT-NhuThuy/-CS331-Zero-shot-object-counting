"""Resumable GT-free CPU capacity audit on the three locked development cohorts.

No model weights or evaluation labels are loaded. Native predictions are optional
diagnostic support and never silently enter the F1-only warp construction.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .capacity_warp import assignment_capacity, build_warp
from .native_greedy import COHORTS, load_manifest, prepare_one, sha256

ROOT = Path(__file__).resolve().parents[2]
SIDECAR_DIR = ROOT / "target_aware_counting/output/reference_correspondence_rc01_20261029"
PROCESSOR_FILES = ("config.json", "processor_config.json", "preprocessor_config.json",
                   "image_processing_molmo2.py", "processing_molmo2.py", "tokenizer.json")


def locked_processor(snapshot: Path, revision: str) -> dict:
    lock_path = snapshot / "molmo_point_snapshot_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock["repo"] != "allenai/MolmoPoint-8B" or lock["revision"] != revision:
        raise ValueError("snapshot identity differs from requested revision")
    for name in PROCESSOR_FILES:
        if sha256(snapshot / name) != lock["files"][name]["sha256"]:
            raise ValueError(f"locked processor file changed: {name}")
    return {"revision": revision, "lock_sha256": sha256(lock_path),
            "processor_file_sha256": {name: lock["files"][name]["sha256"] for name in PROCESSOR_FILES}}


def sidecar_rows(cohort: str, manifest_rows: list[dict]) -> tuple[Path, dict]:
    name = ("candidates_fsc_primary_dev100_full.jsonl"
            if cohort == "fsc_primary_dev100" else f"candidates_{cohort}.jsonl")
    path = SIDECAR_DIR / name
    expected = {r["item_id"]: r for r in manifest_rows}
    found = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            item_id = row["item_id"]
            if item_id not in expected or item_id in found:
                raise ValueError(f"foreign or duplicate F1 sidecar item: {item_id}")
            item = expected[item_id]
            if row["source_image_id"] != item["source_image_id"] or row["query"] != item["query"]:
                raise ValueError(f"F1 sidecar identity mismatch: {item_id}")
            found[item_id] = row
    if found.keys() != expected.keys():
        raise ValueError(f"F1 sidecar denominator mismatch for {cohort}")
    return path, found


def native_rows(specs: list[str], manifests: dict) -> tuple[dict, dict]:
    indexed, hashes = {}, {}
    for spec in specs:
        cohort, sep, name = spec.partition("=")
        if not sep or cohort not in manifests or cohort in indexed:
            raise ValueError("--native-predictions expects unique COHORT=PATH")
        path = Path(name).resolve()
        expected = {r["item_id"]: r for r in manifests[cohort][1]}
        rows = {}
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                item_id = row["item_id"]
                if item_id not in expected or item_id in rows or \
                        row["source_image_id"] != expected[item_id]["source_image_id"] or \
                        row["query"] != expected[item_id]["query"]:
                    raise ValueError(f"native prediction identity mismatch: {item_id}")
                if row.get("status") != "complete" or row.get("count") != len(row.get("points_xy", [])):
                    raise ValueError(f"native prediction is not complete: {item_id}")
                rows[item_id] = row["points_xy"]
        indexed[cohort] = rows
        hashes[cohort] = {"path": str(path), "sha256": sha256(path), "rows": len(rows)}
    return indexed, hashes


def proposal_centers(candidates: list[dict], size: tuple[int, int]) -> tuple[np.ndarray, int]:
    width, height = size
    centers, invalid = [], 0
    for c in candidates:
        box = c.get("box_xyxy_original")
        if box is None or len(box) != 4:
            invalid += 1
            continue
        x0, y0, x1, y1 = map(float, box)
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or \
                not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            invalid += 1
            continue
        centers.append(((x0+x1)/2, (y0+y1)/2))
    return np.asarray(centers, dtype=float).reshape(-1, 2), invalid


def image_budget(inputs: dict) -> dict:
    return {key: list(inputs[key].shape) for key in
            ("input_ids", "pixel_values", "image_token_pooling", "image_num_crops")}


def audit_item(processor, item: dict, sidecar: dict, native_points, args) -> dict:
    with Image.open(item["image_path"]) as source:
        image = source.convert("RGB")
    source_inputs, source_meta, size = prepare_one(processor, item, image)
    raw = [c for c in sidecar["candidates"] if c["forward_id"] == "F1_raw"]
    centers, invalid = proposal_centers(raw, size)
    raw_available = bool(sidecar["raw_pool_complete"] and raw)
    if not raw_available:
        centers = np.empty((0, 2), dtype=float)
    warp, geometry = build_warp(*size, centers, source_meta, bins=min(args.bins, *size),
                                floor=args.floor, min_slope=args.min_slope,
                                max_slope=args.max_slope)
    warped = warp.render(image)
    warped_inputs, warped_meta, _ = prepare_one(processor, item, warped)
    before_budget, after_budget = image_budget(source_inputs), image_budget(warped_inputs)
    if before_budget != after_budget:
        raise ValueError("processor input budget changed after warp")
    before = geometry["assignment_capacity"]
    after = assignment_capacity(warp.forward(centers), warped_meta)
    native = {"available": False}
    if native_points is not None:
        points = np.asarray(native_points, dtype=float).reshape(-1, 2)
        if not np.isfinite(points).all() or np.any(points < 0) or \
                np.any(points[:, 0] > size[0]) or np.any(points[:, 1] > size[1]):
            raise ValueError("native support points outside source image")
        native = {"available": True, "point_count": len(points),
                  "native_before": assignment_capacity(points, source_meta),
                  "native_after": assignment_capacity(warp.forward(points), warped_meta),
                  "combined_before": assignment_capacity(np.concatenate((centers, points)), source_meta),
                  "combined_after": assignment_capacity(
                      warp.forward(np.concatenate((centers, points))), warped_meta)}
    identity = bool(np.allclose(warp.x_edges, np.linspace(0, size[0], len(warp.x_edges))) and
                    np.allclose(warp.y_edges, np.linspace(0, size[1], len(warp.y_edges))))
    return {"status": "ok" if raw_available else "no_complete_f1_raw_pool",
            "raw_f1_count": len(raw), "invalid_f1_boxes": invalid,
            "source_size_wh": list(size), "input_budget": before_budget,
            "identity_warp": identity, "geometry": geometry,
            "capacity_before": before, "capacity_after": after,
            "capacity_gain": after["matched_count"] - before["matched_count"],
            "native_support": native,
            "warp_x_edges": warp.x_edges.tolist(), "warp_y_edges": warp.y_edges.tolist()}


def run(args):
    from transformers import AutoProcessor

    snapshot = args.snapshot.resolve()
    processor_lock = locked_processor(snapshot, args.revision)
    cohorts = list(COHORTS) if args.cohort == "all" else [args.cohort]
    manifests = {cohort: load_manifest(cohort) for cohort in cohorts}
    sidecars = {cohort: sidecar_rows(cohort, rows)
                for cohort, (_, rows) in manifests.items()}
    natives, native_hashes = native_rows(args.native_predictions, manifests)
    contract = {"method": "capacity_b_cpu_audit_v1", "processor": processor_lock,
                "manifest_sha256": {c: sha256(p) for c, (p, _) in manifests.items()},
                "sidecar_sha256": {c: sha256(p) for c, (p, _) in sidecars.items()},
                "native_sources": native_hashes,
                "geometry": {"bins": args.bins, "floor": args.floor,
                             "min_slope": args.min_slope, "max_slope": args.max_slope}}
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError("output exists; use a new path or --resume")
    expected = {(c, r["item_id"]) for c, (_, rows) in manifests.items() for r in rows}
    existing = {}
    if output.exists():
        with output.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = (row["cohort"], row["item_id"])
                if row["contract"] != contract or key not in expected or key in existing:
                    raise ValueError("mixed, foreign or duplicate row in resume file")
                existing[key] = row
    processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True,
                                              local_files_only=True, padding_side="left", use_fast=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        for cohort in cohorts:
            for item in manifests[cohort][1]:
                key = (cohort, item["item_id"])
                if key in existing:
                    continue
                result = audit_item(processor, item, sidecars[cohort][1][item["item_id"]],
                                    natives.get(cohort, {}).get(item["item_id"]), args)
                row = {"cohort": cohort, "item_id": item["item_id"],
                       "source_image_id": item["source_image_id"],
                       "query": item["query"], "contract": contract, **result}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                existing[key] = row
                print(f"{len(existing)}/{len(expected)} {cohort} {item['item_id']} "
                      f"{result['status']} gain={result['capacity_gain']}", flush=True)
    if existing.keys() != expected:
        raise ValueError("audit did not cover every locked item")
    summary = {"contract": contract, "output": str(output), "denominator": len(expected),
               "cohorts": {}}
    for cohort in cohorts:
        rows = [existing[(cohort, item["item_id"])] for item in manifests[cohort][1]]
        gains = [row["capacity_gain"] for row in rows]
        summary["cohorts"][cohort] = {
            "denominator": len(rows), "raw_pool_available": sum(r["status"] == "ok" for r in rows),
            "capacity_gain_positive": sum(g > 0 for g in gains),
            "capacity_gain_zero": sum(g == 0 for g in gains),
            "capacity_gain_negative": sum(g < 0 for g in gains),
            "total_matched_before": sum(r["capacity_before"]["matched_count"] for r in rows),
            "total_matched_after": sum(r["capacity_after"]["matched_count"] for r in rows),
            "native_support_available": sum(r["native_support"]["available"] for r in rows),
            "identity_warp": sum(r["identity_warp"] for r in rows)}
    summary_path = args.summary.resolve()
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"summary": str(summary_path), "cohorts": summary["cohorts"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=(*COHORTS, "all"), default="all")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--native-predictions", action="append", default=[], metavar="COHORT=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--floor", type=float, default=0.5)
    parser.add_argument("--min-slope", type=float, default=0.5)
    parser.add_argument("--max-slope", type=float, default=2.0)
    args = parser.parse_args()
    if args.bins < 2 or not 0 < args.floor <= 1 or \
            not 0 < args.min_slope <= 1 <= args.max_slope:
        parser.error("invalid geometry bounds")
    run(args)


if __name__ == "__main__":
    main()
