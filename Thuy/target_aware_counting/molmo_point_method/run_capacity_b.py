"""One-item capacity-allocation diagnostic; no GT or TEST access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .capacity_warp import assignment_capacity, build_warp, collision_groups
from .f1_sidecar import load_f1_sidecar
from .native_greedy import COHORTS, infer_one, load_manifest, prepare_one, sha256
from .snapshot_lock import verify


def run(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    manifest, rows = load_manifest(args.cohort)
    item = next((r for r in rows if r["item_id"] == args.item_id), None)
    if item is None:
        raise ValueError("item not in locked development cohort")
    sidecar, candidates, available = load_f1_sidecar(args.cohort, item)
    if not available:
        raise ValueError("complete F1 raw proposal pool required for capacity diagnostic")
    snapshot = Path(args.snapshot).resolve()
    lock = json.loads((snapshot / "molmo_point_snapshot_lock.json").read_text())
    if lock["revision"] != args.revision or lock["repo"] != "allenai/MolmoPoint-8B":
        raise ValueError("processor snapshot revision mismatch")
    for name in ("config.json", "processor_config.json", "preprocessor_config.json",
                 "image_processing_molmo2.py", "processing_molmo2.py", "tokenizer.json"):
        if sha256(snapshot / name) != lock["files"][name]["sha256"]:
            raise ValueError(f"processor snapshot changed: {name}")
    verification = "processor_files_only"
    processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True,
                                              local_files_only=True, padding_side="left")
    with Image.open(item["image_path"]) as source:
        image = source.convert("RGB")
    source_inputs, source_metadata, size = prepare_one(processor, item)
    centers = np.asarray([[(c["box_xyxy_original"][0]+c["box_xyxy_original"][2])/2,
                           (c["box_xyxy_original"][1]+c["box_xyxy_original"][3])/2]
                          for c in candidates], dtype=float)
    warp, diagnostics = build_warp(*size, centers, source_metadata, bins=args.bins,
                                   floor=args.floor, min_slope=args.min_slope,
                                   max_slope=args.max_slope)
    warped = warp.render(image)
    warped_inputs, warped_metadata, _ = prepare_one(processor, item, warped)
    after = assignment_capacity(warp.forward(centers), warped_metadata)
    before_budget = {key: tuple(source_inputs[key].shape) for key in
                     ("input_ids", "pixel_values", "image_token_pooling", "image_num_crops")}
    after_budget = {key: tuple(warped_inputs[key].shape) for key in before_budget}
    if before_budget != after_budget:
        raise ValueError("processor changed input budget; comparison is not matched")
    before = diagnostics["assignment_capacity"]
    gain = after["matched_count"] - before["matched_count"]
    should_decode = not args.diagnostic_only and (gain > 0 or args.allow_no_gain)
    result = None
    if should_decode:
        verify(snapshot, args.revision)
        verification = "full_snapshot"
        dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
        model = AutoModelForImageTextToText.from_pretrained(
            snapshot, trust_remote_code=True, local_files_only=True, dtype=dtype,
            device_map=args.device_map, offload_folder=args.offload_folder,
            offload_state_dict=True).eval()
        result = infer_one(model, processor, item, args.max_new_tokens, warped)
        result["points_xy_warped"] = result["points_xy"]
        result["points_xy"] = warp.inverse(result["points_xy_warped"]).tolist()
        result["raw_points_warped"] = result.pop("raw_points")
    return {"item_id": item["item_id"], "source_image_id": item["source_image_id"],
            "query": item["query"], "cohort": args.cohort,
            "manifest_sha256": sha256(manifest), "f1_sidecar_sha256": sha256(sidecar),
            "revision": lock["revision"], "snapshot_verification": verification,
            "method": "native_capacity_warp_b",
            "config": {"bins": args.bins, "floor": args.floor,
                       "min_slope": args.min_slope, "max_slope": args.max_slope,
                       "max_new_tokens": args.max_new_tokens, "dtype": args.dtype},
            "processor_input_shapes_before_after": [before_budget, after_budget],
            "assignment_capacity_after": after, "assignment_capacity_gain": gain,
            "shared_address_pairs_after": len(collision_groups(warp.forward(centers), warped_metadata)),
            "warp_x_edges": warp.x_edges.tolist(), "warp_y_edges": warp.y_edges.tolist(),
            "diagnostics": diagnostics, "prediction": result,
            "status": (result["status"] if result is not None else
                       ("diagnostic_only" if args.diagnostic_only else "capacity_no_gain"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=COHORTS)
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", required=True, type=int)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--floor", type=float, default=0.5)
    parser.add_argument("--min-slope", type=float, default=0.5)
    parser.add_argument("--max-slope", type=float, default=2.0)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--offload-folder")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--allow-no-gain", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "matched_before": result["diagnostics"]["assignment_capacity"]["matched_count"],
                      "matched_after": result["assignment_capacity_after"]["matched_count"],
                      "capacity_gain": result["assignment_capacity_gain"]}))


if __name__ == "__main__":
    main()
