"""Reconstruct FSC primary top-10/view raw F1 candidates from frozen traces."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

from .config import file_sha256
from .prepare_candidates import read_rows


def top10(trace: dict, item_id: str, view: str, width: int, height: int,
          offset: tuple[int,int] = (0,0)) -> list[dict]:
    scores = trace["native_scores"].reshape(-1)
    boxes = trace["pred_boxes_norm"]
    if scores.numel() != 900 or boxes.shape != (900,4):
        raise ValueError("F1 trace is not the locked 900-slot forward")
    indices = sorted(range(900),key=lambda i:(-float(scores[i]),i))[:10]
    result = []
    for slot in indices:
        cx,cy,bw,bh = (float(v) for v in boxes[slot])
        x0,y0 = max(0.,(cx-bw/2)*width),max(0.,(cy-bh/2)*height)
        x1,y1 = min(float(width),(cx+bw/2)*width),min(float(height),(cy+bh/2)*height)
        if x0 >= x1 or y0 >= y1:
            raise ValueError("invalid F1 trace box")
        result.append({"candidate_id":f"{item_id}:{view}:{slot}",
                       "box_xyxy":[x0+offset[0],y0+offset[1],x1+offset[0],y1+offset[1]],
                       "forward":"F1_raw","view":view,"slot_index":slot,
                       "native_score":float(scores[slot])})
    return result


def build(manifest: Path, m0_predictions: Path, full_traces: Path, tile_traces: Path,
          tile_map: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    items = read_rows(manifest)
    m0 = {r["item_id"]:r for r in read_rows(m0_predictions)}
    mapping = {}
    with tile_map.open(newline="") as stream:
        for row in csv.DictReader(stream):
            mapping.setdefault(str(row["query_image_id"]),[]).append(row)
    records = []
    for item in items:
        qid = str(item["query_id"])
        stem = Path(item["image_path"]).stem
        path = full_traces/f"{qid}_{stem}.pt"
        full = torch.load(path,map_location="cpu",weights_only=True)
        if str(full["image_id"]) != qid or full["file_name"] != Path(item["image_path"]).name or \
                full["query"] != item["query"]:
            raise ValueError(f"full F1 identity mismatch: {item['item_id']}")
        with Image.open(item["image_path"]) as opened:
            width,height = opened.size
        candidates = top10(full,item["item_id"],"full",width,height)
        tiles = sorted(mapping[qid],key=lambda r:int(r["tile_index"]))
        if len(tiles) != 4 or [int(r["tile_index"]) for r in tiles] != list(range(4)):
            raise ValueError("incomplete tile trace geometry")
        for row in tiles:
            index = int(row["tile_index"])
            rect = [int(row[k]) for k in ("x0","y0","x1","y1")]
            tw,th = rect[2]-rect[0],rect[3]-rect[1]
            filename = f"{row['tile_image_id']}_{Path(row['tile_file_name']).stem}.pt"
            tile = torch.load(tile_traces/filename,map_location="cpu",weights_only=True)
            if str(tile["image_id"]) != str(row["tile_image_id"]) or \
                    tile["file_name"] != row["tile_file_name"] or tile["query"] != item["query"] or \
                    str(row["source_image_id"]) != str(item["source_image_id"]) or \
                    (int(row["source_width"]),int(row["source_height"])) != (width,height):
                raise ValueError("tile F1 identity/geometry mismatch")
            candidates += top10(tile,item["item_id"],f"tile{index}",tw,th,(rect[0],rect[1]))
        native = m0[item["item_id"]]
        if native["count"] != len(native["final_boxes"]):
            raise ValueError("M0 count/boxes mismatch")
        m0_ids = []
        for box in native["final_boxes"]:
            cid = str(box["source_id"])
            m0_ids.append(cid)
            candidates.append({"candidate_id":cid,"box_xyxy":box["bbox_xyxy_float"],
                               "forward":"F2_M0","view":"full","slot_index":box["slot"],
                               "native_score":None})
        records.append({"query":{"query_id":item["query_id"],"source_image_id":item["source_image_id"],
                                  "query":item["query"],"image_width":width,"image_height":height,
                                  "candidates":candidates,"m0_ids":m0_ids,"m0_count":len(m0_ids)}})
    with output.open("x") as stream:
        for row in records:
            stream.write(json.dumps(row,sort_keys=True)+"\n")
    return {"items":len(records),"raw_f1_boxes":sum(sum(c["forward"] == "F1_raw"
            for c in r["query"]["candidates"]) for r in records),"sha256":file_sha256(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest","m0-predictions","full-traces","tile-traces","tile-map","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest,args.m0_predictions,args.full_traces,
                           args.tile_traces,args.tile_map,args.out),sort_keys=True))


if __name__ == "__main__":
    main()
