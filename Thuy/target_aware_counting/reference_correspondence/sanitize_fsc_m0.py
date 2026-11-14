"""Extract only frozen prediction boxes from the legacy FSC COCO-shaped file.

The legacy file also contains posthoc labels on image records. This converter
uses only identity and prediction fields, emits a GT-free sidecar, and checks
the native post-text-filter count sidecar for every requested item.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .config import file_sha256


def _rows(path: Path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sanitize(manifest: Path, legacy_predictions: Path, count_sidecar: Path,
             output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    items = _rows(manifest)
    legacy = json.loads(legacy_predictions.read_text())
    images = {str(r["id"]):r for r in legacy["images"]}
    annotations = defaultdict(list)
    for row in legacy["annotations"]:
        annotations[str(row["image_id"])].append(row)
    counts = {str(r["query_image_id"]):int(r["pred_count_post_text_mask"])
              for r in _rows(count_sidecar)}
    if len({r["item_id"] for r in items}) != len(items):
        raise ValueError("duplicate FSC manifest item")
    records = []
    for item in items:
        query_id = str(item["query_id"])
        image = images[query_id]
        if str(image["source_image_id"]) != str(item["source_image_id"]) or \
                image["query_caption"] != item["query"] or \
                Path(image["file_name"]).name != Path(item["image_path"]).name:
            raise ValueError(f"legacy FSC prediction identity mismatch: {item['item_id']}")
        boxes = annotations[query_id]
        if len(boxes) != counts[query_id]:
            raise ValueError(f"native post-text-filter count mismatch: {item['item_id']}")
        final = []
        for slot,row in enumerate(boxes):
            x,y,w,h = (float(v) for v in row["bbox"])
            final.append({"source_id":f"{item['item_id']}:M0:f2:{slot}",
                          "slot":slot,"bbox_xyxy_float":[x,y,x+w,y+h],
                          "bbox_xywh_native_int":row["bbox"]})
        records.append({"item_id":item["item_id"],"query_id":item["query_id"],
                        "source_image_id":item["source_image_id"],"query":item["query"],
                        "variant":"M0","count":len(final),"final_boxes":final})
    with output.open("x") as stream:
        for record in records:
            stream.write(json.dumps(record,sort_keys=True)+"\n")
    return {"items":len(records),"boxes":sum(r["count"] for r in records),
            "source_sha256":file_sha256(legacy_predictions),
            "count_sidecar_sha256":file_sha256(count_sidecar),
            "sanitized_sha256":file_sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest","legacy-predictions","count-sidecar","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    args = parser.parse_args()
    print(json.dumps(sanitize(args.manifest,args.legacy_predictions,args.count_sidecar,args.out),
                     sort_keys=True))


if __name__ == "__main__":
    main()
