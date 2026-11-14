"""Fixed small H2 same-image query switch diagnostic; no semantic decision."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "target_aware_counting"))
from fqie_alpha_smoke import (alpha_mask, context_crop, cosine, load_model,
                              normalized_text, sha, synchronized_transform, write_once)

SOURCE = ROOT / "target_aware_counting/output/lvis_query_switch_val_20261010_205455/runs/no_pseudo/predictions.json"
DEST = ROOT / "target_aware_counting/output/fqie_h2_alpha_query_switch_20261104_v1"
IMAGE_DIR = ROOT / "data/lvis_v1_val_source/selected_images"
PAIRS = ((227035, "bottle cap", "peeler (tool for fruit and vegetables)"),
         (350988, "bench", "kitchen table"))


def prepare():
    data = json.loads(SOURCE.read_text())
    items = []
    for sid, present, absent in PAIRS:
        images = {x["query_name"]: x for x in data["images"] if x["source_image_id"] == sid}
        assert images[present]["query_status"] == "present"
        assert images[absent]["query_status"] == "verified_absent"
        assert images[present]["gt_count"] > 0 and images[absent]["gt_count"] == 0
        assert images[present]["file_name"] == images[absent]["file_name"]
        path = IMAGE_DIR / images[present]["file_name"]
        for origin in (present, absent):
            im = images[origin]
            proposals = sorted((a for a in data["annotations"] if a["image_id"] == im["id"]),
                               key=lambda a: (-a["score"], a["id"]))[:2]
            assert len(proposals) == 2
            for a in proposals:
                x,y,w,h = a["bbox"]
                items.append({"case_id": f"{sid}:{im['id']}:{a['id']}",
                              "source_image_id": sid, "image_path": str(path),
                              "image_sha256": sha(path), "origin_query": origin,
                              "origin_query_status": im["query_status"],
                              "native_score": a["score"],
                              "box_xyxy": [x,y,x+w,y+h],
                              "comparison_queries": [present, absent]})
    assert len(items) == 8
    write_once(DEST / "manifest.json", {"status": "FIXED_BEFORE_ALPHA_INFERENCE",
        "source_predictions_sha256": sha(SOURCE), "selection": "top_two_native_score_each_query",
        "view_policy": "full_and_1.5x_context_letterbox_RGB_alpha",
        "model": "official_AlphaCLIP_L14_same_lock_as_smoke_v2",
        "limitations": "selected two known query-switch source images; diagnostic, not population estimate",
        "items": items})
    print(json.dumps({"items": len(items), "manifest_sha256": sha(DEST / "manifest.json")}))


def run():
    import torch
    from PIL import Image
    manifest = json.loads((DEST / "manifest.json").read_text())
    assert manifest["source_predictions_sha256"] == sha(SOURCE) and len(manifest["items"]) == 8
    torch.manual_seed(0)
    model, ac = load_model()
    resolution = int(model.visual.input_resolution)
    started = time.monotonic()
    with torch.inference_mode():
        text_cache = {}
        for item in manifest["items"]:
            out = DEST / "scores" / (item["case_id"].replace(":", "_") + ".json")
            if out.exists():
                assert json.loads(out.read_text())["case_id"] == item["case_id"]
                continue
            image = Image.open(item["image_path"]).convert("RGB")
            assert sha(Path(item["image_path"])) == item["image_sha256"]
            mask = alpha_mask(image.size, item["box_xyxy"])
            queries = item["comparison_queries"]
            key = tuple(queries)
            if key not in text_cache:
                text_cache[key] = model.encode_text(ac.tokenize([normalized_text(q) for q in queries], truncate=True).cuda())
            result = {}
            for view in ("full", "context"):
                if view == "full":
                    rgb, alpha = image, mask
                else:
                    rgb, bounds = context_crop(image, item["box_xyxy"])
                    alpha = mask.crop(bounds)
                rt, at, pixels = synchronized_transform(rgb, alpha, resolution)
                if pixels == 0: raise RuntimeError(f"alpha vanished: {item['case_id']} {view}")
                visual = model.visual(rt.unsqueeze(0).to(model.dtype).cuda(),
                                      at.unsqueeze(0).to(model.dtype).cuda())
                result[view] = {"alpha_pixels": pixels,
                    "cosine": {q: cosine(visual, text_cache[key][i:i+1]) for i,q in enumerate(queries)}}
            write_once(out, {"case_id": item["case_id"], "source_image_id": item["source_image_id"],
                             "origin_query": item["origin_query"], "origin_query_status": item["origin_query_status"],
                             "native_score": item["native_score"], "box_xyxy": item["box_xyxy"],
                             "views": result, "semantic_decision": "NOT_MADE"})
            print(json.dumps({"scored": item["case_id"]}), flush=True)
    print(json.dumps({"wall_seconds": time.monotonic()-started,
                      "peak_vram_gib": torch.cuda.max_memory_allocated()/1024**3}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=("prepare","run"))
    args=parser.parse_args()
    prepare() if args.action == "prepare" else run()
