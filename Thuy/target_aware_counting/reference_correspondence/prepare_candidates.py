"""Freeze the locked development candidate pool without exposing labels to inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .candidate_adapter import action_pool, m0_prediction_pool
from .config import file_sha256


FORBIDDEN = {"gt_count", "gt_boxes", "query_status", "status", "annotations", "points_xy"}


def read_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(FORBIDDEN & row.keys() for row in rows):
        raise ValueError(f"label/status field in GT-free input {path}")
    return rows


def build(manifest_path: Path, output_path: Path, *, actions_path: Path | None = None,
          m0_predictions_path: Path | None = None) -> dict:
    if output_path.exists():
        raise FileExistsError(output_path)
    if (actions_path is None) == (m0_predictions_path is None):
        raise ValueError("choose exactly one frozen candidate source")
    manifest = read_rows(manifest_path)
    if len({r["item_id"] for r in manifest}) != len(manifest):
        raise ValueError("duplicate manifest item ID")
    if actions_path is not None:
        actions = {str(r["query"]["query_id"]): r for r in read_rows(actions_path)}
        predictions = None
    else:
        predictions = {r["item_id"]: r for r in read_rows(m0_predictions_path)}
        actions = None
    output = []
    for item in manifest:
        with Image.open(item["image_path"]) as opened:
            image_hw = (opened.height, opened.width)
        if actions is not None:
            source = actions[str(item["query_id"])]
            q = source["query"]
            if str(q["source_image_id"]) != str(item["source_image_id"]) or \
                    q["query"] != item["query"] or \
                    (q["image_height"], q["image_width"]) != image_hw:
                raise ValueError(f"action/manifest image-query mismatch: {item['item_id']}")
            candidates = action_pool(source,image_hw)
            m0_ids = sorted(q["m0_ids"])
            raw_complete = True
        else:
            pred = predictions[item["item_id"]]
            if str(pred["source_image_id"]) != str(item["source_image_id"]) or \
                    str(pred["query_id"]) != str(item["query_id"]) or \
                    pred["query"] != item["query"]:
                raise ValueError(f"M0/manifest image-query mismatch: {item['item_id']}")
            candidates = m0_prediction_pool(pred,image_hw)
            m0_ids = sorted(c.candidate_id for c in candidates)
            raw_complete = False
        output.append({"item_id":item["item_id"],"query":item["query"],
                       "source_image_id":item["source_image_id"],
                       "image_sha256":file_sha256(item["image_path"]),
                       "image_hw":list(image_hw),
                       "raw_pool_complete":raw_complete,
                       "m0_final_ids":m0_ids,
                       "candidates":[c.serializable() for c in candidates],
                       "candidate_source_sha256":file_sha256(actions_path or m0_predictions_path)})
    with output_path.open("x") as stream:
        for row in output:
            stream.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
    return {"items":len(output),"candidates":sum(len(r["candidates"]) for r in output),
            "m0_boxes":sum(len(r["m0_final_ids"]) for r in output),
            "raw_complete_items":sum(r["raw_pool_complete"] for r in output),
            "sha256":file_sha256(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True,type=Path)
    parser.add_argument("--actions",type=Path)
    parser.add_argument("--m0-predictions",type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest,args.out,actions_path=args.actions,
                           m0_predictions_path=args.m0_predictions),sort_keys=True))


if __name__ == "__main__":
    main()
