"""Build pooled/structured hypotheses from frozen GT-free atoms and gallery."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .candidate_adapter import Candidate
from .config import file_sha256, load_locked_config, stable_hash
from .hypothesis_builder import build_item
from .predict import keyed, rows


def build(manifest: Path, candidates: Path, atom_index: Path, gallery: Path,
          config_path: Path, output: Path, max_new_items: int | None = None,
          max_runtime_seconds: float | None = None) -> dict:
    if max_new_items is not None and max_new_items < 1:
        raise ValueError("max_new_items must be positive")
    if max_runtime_seconds is not None and max_runtime_seconds <= 0:
        raise ValueError("max_runtime_seconds must be positive")
    started = time.monotonic()
    new_items = 0
    config,_ = load_locked_config(config_path)
    if file_sha256(gallery) != config["gallery_sha256"]:
        raise ValueError("gallery snapshot changed")
    items,pools,index = (keyed(path) for path in (manifest,candidates,atom_index))
    if set(items) != set(pools) or not set(items) <= set(index):
        raise ValueError("manifest/candidate item set differs or atom index is missing items")
    gallery_rows = rows(gallery)
    existing = keyed(output) if output.exists() else {}
    if not set(existing) <= set(items):
        raise ValueError("resume hypothesis output has unknown items")
    for item_id,item in items.items():
        if item_id in existing:
            row = existing[item_id]
            if row["feature_config_sha256"] != stable_hash(config["feature"]) or \
                    row["numerical_config_sha256"] != stable_hash(config["numerical"]) or \
                    row["reference_snapshot_sha256"] != config["gallery_sha256"]:
                raise ValueError("resume hypothesis config/gallery mismatch")
            continue
        if (max_new_items is not None and new_items >= max_new_items) or \
                (max_runtime_seconds is not None and new_items and
                 time.monotonic()-started >= max_runtime_seconds):
            break
        pool = pools[item_id]
        atoms = index[item_id]
        if any(str(x["source_image_id"]) != str(item["source_image_id"]) or
               x["query"] != item["query"] for x in (pool,atoms)):
            raise ValueError("item/pool/atom index identity mismatch")
        if pool["image_sha256"] != atoms["image_sha256"] or \
                pool["image_sha256"] != file_sha256(item["image_path"]):
            raise ValueError("source image changed since candidate/feature extraction")
        if not pool["raw_pool_complete"]:
            row = {"item_id":item_id,"query":item["query"],
                   "source_image_id":item["source_image_id"],
                   "feature_config_sha256":stable_hash(config["feature"]),
                   "numerical_config_sha256":stable_hash(config["numerical"]),
                   "reference_snapshot_sha256":config["gallery_sha256"],
                   "global_hypotheses":[],"match_hypotheses":[],"gamma_zero_hypotheses":[],
                   "unresolved_candidate_ids":[],"status":"raw_pool_incomplete"}
        else:
            refs = [r for r in gallery_rows if r["canonical_query"] == item["query"]]
            valid = [r for r in refs if r["status"] == "valid"]
            available = (sum(r["semantic_role"] == "target" for r in valid) >=
                         config["reference_policy"]["min_valid_target"] and
                         sum(r["semantic_role"] == "competitor" for r in valid) >=
                         config["reference_policy"]["min_valid_competitor"])
            if available:
                candidate_objects = [Candidate(**c) for c in pool["candidates"]]
                row = build_item(item,candidate_objects,refs,atoms["atom_cache_path"],
                                 config["feature"],config["numerical"],config["gallery_sha256"])
            else:
                row = {"item_id":item_id,"query":item["query"],
                       "source_image_id":item["source_image_id"],
                       "feature_config_sha256":stable_hash(config["feature"]),
                       "numerical_config_sha256":stable_hash(config["numerical"]),
                       "reference_snapshot_sha256":config["gallery_sha256"],
                       "global_hypotheses":[],"match_hypotheses":[],"gamma_zero_hypotheses":[],
                       "unresolved_candidate_ids":[],"status":"reference_unavailable"}
        with output.open("a") as stream:
            stream.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"item_id":item_id,"status":row["status"],
                          "hypotheses":len(row["match_hypotheses"])}),flush=True)
        new_items += 1
    completed = len(existing)+new_items
    return {"complete":completed==len(items),"items":completed,
            "planned":len(items),"new_items":new_items,
            "sha256":file_sha256(output) if output.exists() else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest","candidates","atom-index","gallery","config","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--max-new-items",type=int)
    parser.add_argument("--max-runtime-seconds",type=float)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest,args.candidates,args.atom_index,args.gallery,
                           args.config,args.out,args.max_new_items,
                           args.max_runtime_seconds),sort_keys=True))


if __name__ == "__main__":
    main()
