"""Run frozen MolmoPoint B policy from a complete, GT-free CPU capacity audit.

One image and one point set are decoded per item. The policy uses the warped
image only when the precomputed assignment capacity gain is positive; otherwise
it uses the source image. This does not read evaluation labels.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .capacity_warp import Warp
from .native_greedy import COHORTS, infer_one, load_existing, load_manifest, sha256
from .snapshot_lock import verify


def load_audit(path: Path, cohort: str, items: list[dict], snapshot: Path,
               revision: str, manifest_sha256: str) -> tuple[dict, dict]:
    rows = {}
    contract = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if contract is None:
                contract = row["contract"]
            if row["contract"] != contract:
                raise ValueError("mixed CPU audit contracts")
            if row["cohort"] != cohort:
                continue
            item_id = row["item_id"]
            if item_id in rows:
                raise ValueError(f"duplicate CPU audit item: {item_id}")
            rows[item_id] = row
    if contract is None or contract["method"] != "capacity_b_cpu_audit_v1" or \
            contract["processor"]["revision"] != revision or \
            contract["manifest_sha256"].get(cohort) != manifest_sha256 or \
            contract["processor"]["lock_sha256"] != sha256(snapshot / "molmo_point_snapshot_lock.json"):
        raise ValueError("CPU audit does not match locked processor or manifest")
    expected = {item["item_id"]: item for item in items}
    if rows.keys() != expected.keys():
        raise ValueError("CPU audit does not cover full cohort")
    for item_id, row in rows.items():
        item = expected[item_id]
        if row["source_image_id"] != item["source_image_id"] or row["query"] != item["query"]:
            raise ValueError(f"CPU audit identity mismatch: {item_id}")
        gain = row["capacity_after"]["matched_count"] - row["capacity_before"]["matched_count"]
        if row["capacity_gain"] != gain or row["status"] not in ("ok", "no_complete_f1_raw_pool"):
            raise ValueError(f"CPU audit capacity mismatch: {item_id}")
    return contract, rows


def infer_item(model, processor, item: dict, audit: dict, max_new_tokens: int, *,
               prompt_style: str = "every_instance",
               decoding_style: str = "native") -> dict:
    use_warp = audit["status"] == "ok" and audit["capacity_gain"] > 0
    if use_warp:
        width, height = audit["source_size_wh"]
        warp = Warp(np.asarray(audit["warp_x_edges"], dtype=float),
                    np.asarray(audit["warp_y_edges"], dtype=float), width, height)
        with Image.open(item["image_path"]) as source:
            image = warp.render(source.convert("RGB"))
        result = infer_one(model, processor, item, max_new_tokens, image,
                           prompt_style=prompt_style, decoding_style=decoding_style)
        result["points_xy_warped"] = result["points_xy"]
        result["points_xy"] = warp.inverse(result["points_xy_warped"]).tolist()
        result["raw_points_warped"] = result.pop("raw_points")
    else:
        result = infer_one(model, processor, item, max_new_tokens,
                           prompt_style=prompt_style, decoding_style=decoding_style)
    return {"selection": "capacity_warp" if use_warp else "native_identity",
            "capacity_gain": audit["capacity_gain"], **result}


def run(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    manifest, items = load_manifest(args.cohort)
    snapshot = args.snapshot.resolve()
    verify(snapshot, args.revision)
    audit_contract, audit = load_audit(args.audit.resolve(), args.cohort, items,
                                       snapshot, args.revision, sha256(manifest))
    selected = items
    if args.item_id is not None:
        selected = [item for item in items if item["item_id"] == args.item_id]
        if not selected:
            raise ValueError("item_id not in locked cohort")
    method = ("molmopoint_capacity_b_strict_points_v1"
              if args.decoding_style == "strict_points_v1" else "molmopoint_capacity_b_v1")
    contract = {"method": method, "cohort": args.cohort,
                "manifest_sha256": sha256(manifest), "audit_sha256": sha256(args.audit.resolve()),
                "audit_geometry": audit_contract["geometry"],
                "snapshot": str(snapshot), "revision": args.revision,
                "snapshot_lock_sha256": sha256(snapshot / "molmo_point_snapshot_lock.json"),
                "max_new_tokens": args.max_new_tokens, "load_dtype": args.dtype,
                "decoding_style": args.decoding_style,
                "selection_rule": "warp iff complete F1 raw pool and assignment capacity gain > 0",
                "item_id_filter": args.item_id}
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError("output exists; pass --resume with the same contract")
    existing = load_existing(output, contract, {item["item_id"] for item in selected})
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True, dtype=dtype,
        device_map=args.device_map, offload_folder=args.offload_folder,
        offload_state_dict=True).eval()
    processor = AutoProcessor.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True,
        padding_side="left", use_fast=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        for item in selected:
            if item["item_id"] in existing:
                continue
            try:
                result = infer_item(model, processor, item, audit[item["item_id"]],
                                    args.max_new_tokens,
                                    decoding_style=args.decoding_style)
            except Exception as exc:
                if args.fail_fast:
                    raise
                result = {"selection": "capacity_warp" if audit[item["item_id"]]["capacity_gain"] > 0
                          and audit[item["item_id"]]["status"] == "ok" else "native_identity",
                          "capacity_gain": audit[item["item_id"]]["capacity_gain"],
                          "status": "error", "count": None, "points_xy": [],
                          "error_type": type(exc).__name__, "error": str(exc)}
            record = {"item_id": item["item_id"], "source_image_id": item["source_image_id"],
                      "query": item["query"], "contract": contract, **result}
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            existing[item["item_id"]] = record
            print(f"{len(existing)}/{len(selected)} {item['item_id']} "
                  f"{result['selection']} {result['status']}", flush=True)
    print(json.dumps({"rows": len(existing), "denominator": len(selected),
                      "complete": sum(r["status"] == "complete" for r in existing.values()),
                      "errors": sum(r["status"] == "error" for r in existing.values()),
                      "truncated": sum(r["status"] == "truncated" for r in existing.values()),
                      "warped": sum(r["selection"] == "capacity_warp" for r in existing.values()),
                      "output": str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--item-id", help="one-item smoke; output cannot be frozen as a full cohort")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--decoding-style", choices=("native", "strict_points_v1"),
                        default="native")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--offload-folder")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    run(args)


if __name__ == "__main__":
    main()
