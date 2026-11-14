"""Query-calibrated Alpha-CLIP region controls on the frozen LVIS manifest."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "target_aware_counting"))
from fqie_alpha_smoke import (alpha_mask, cosine, load_model, normalized_text,
                              sha, synchronized_transform, write_once)

BASE = ROOT / "target_aware_counting/output/fqie_h2_alpha_lvis_full_20261104_v1"
DEST = ROOT / "target_aware_counting/output/fqie_h2_alpha_region_controls_20261104_v1"
BASE_MANIFEST_SHA = "523637bfa91db228c19513644fe9b9fba94ba5855321533f2e58d9b9fcd774c9"


def box_iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    aa = max(0, a[2]-a[0])*max(0, a[3]-a[1]); bb = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb>inter else 0.0


def far_box(size, box):
    w, h = size
    bw = min(w, max(1, math.ceil(box[2])-math.floor(box[0])))
    bh = min(h, max(1, math.ceil(box[3])-math.floor(box[1])))
    candidates = ([0,0,bw,bh], [w-bw,0,w,bh], [0,h-bh,bw,h], [w-bw,h-bh,w,h])
    return min(candidates, key=lambda x: (box_iou(x, box), -((x[0]+x[2]-box[0]-box[2])**2 +
                                                              (x[1]+x[3]-box[1]-box[3])**2)))


def prepare():
    if sha(BASE / "inference_manifest.json") != BASE_MANIFEST_SHA:
        raise RuntimeError("base manifest changed")
    base = json.loads((BASE / "inference_manifest.json").read_text())
    write_once(DEST / "manifest.json", {"status": "LOCKED_BEFORE_CONTROL_INFERENCE",
        "base_manifest_sha256": BASE_MANIFEST_SHA, "items": base["items"],
        "controls": ["all_content", "complement_rectangle", "far_same_size_rectangle"],
        "view": "full_letterbox_identical_to_correct_region",
        "purpose": "same-query same-image margins; inference does not read eval labels"})
    print(json.dumps({"items": len(base["items"]), "manifest_sha256": sha(DEST / "manifest.json")}))


def run():
    import torch
    from PIL import Image, ImageChops
    manifest = json.loads((DEST / "manifest.json").read_text())
    if manifest["base_manifest_sha256"] != BASE_MANIFEST_SHA or len(manifest["items"]) != 306:
        raise RuntimeError("control manifest mismatch")
    torch.manual_seed(0)
    model, ac = load_model(); resolution = int(model.visual.input_resolution)
    text_cache = {}; started = time.monotonic()
    with torch.inference_mode():
        for item in manifest["items"]:
            out = DEST / "scores" / (item["candidate_id"].replace(":", "_") + ".json")
            if out.exists():
                if json.loads(out.read_text())["candidate_id"] != item["candidate_id"]: raise RuntimeError("resume mismatch")
                continue
            image = Image.open(item["image_path"]).convert("RGB")
            if sha(Path(item["image_path"])) != item["image_sha256"]: raise RuntimeError("image changed")
            query = item["query"]
            if query not in text_cache:
                text_cache[query] = model.encode_text(ac.tokenize([normalized_text(query)], truncate=True).cuda())
            correct = alpha_mask(image.size, item["box_xyxy"])
            all_content = Image.new("L", image.size, 255)
            complement = ImageChops.invert(correct)
            shifted_box = far_box(image.size, item["box_xyxy"])
            shifted = alpha_mask(image.size, shifted_box)
            values = {}
            for name, mask in (("all_content", all_content), ("complement_rectangle", complement),
                               ("far_same_size_rectangle", shifted)):
                rt, at, pixels = synchronized_transform(image, mask, resolution)
                visual = model.visual(rt.unsqueeze(0).to(model.dtype).cuda(), at.unsqueeze(0).to(model.dtype).cuda())
                values[name] = {"alpha_pixels": pixels, "target_cosine": cosine(visual, text_cache[query])}
            write_once(out, {"candidate_id": item["candidate_id"], "source_image_id": item["source_image_id"],
                "query_image_id": item["query_image_id"], "controls": values, "shifted_box_xyxy": shifted_box,
                "shifted_vs_correct_iou": box_iou(shifted_box, item["box_xyxy"]), "decision": "NOT_MADE"})
        print(json.dumps({"candidates": len(manifest["items"]), "visual_forwards": 3*len(manifest["items"]),
                          "wall_seconds": time.monotonic()-started,
                          "peak_vram_gib": torch.cuda.max_memory_allocated()/1024**3}))


def freeze():
    manifest = json.loads((DEST / "manifest.json").read_text()); hashes={}
    for item in manifest["items"]:
        p=DEST/"scores"/(item["candidate_id"].replace(":", "_")+".json"); d=json.loads(p.read_text())
        if d["candidate_id"] != item["candidate_id"]: raise RuntimeError("binding mismatch")
        if not all(math.isfinite(x["target_cosine"]) and x["alpha_pixels"]>0 for x in d["controls"].values()):
            raise RuntimeError("invalid control")
        hashes[str(p.relative_to(DEST))]=sha(p)
    write_once(DEST/"freeze_scores.json", {"status":"COMPLETE_RAW_CONTROLS_NO_DECISION",
        "manifest_sha256":sha(DEST/"manifest.json"),"score_file_hashes":hashes})
    print(json.dumps({"frozen":len(hashes)}))


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("prepare","run","freeze"));a=p.parse_args().action
    {"prepare":prepare,"run":run,"freeze":freeze}[a]()
