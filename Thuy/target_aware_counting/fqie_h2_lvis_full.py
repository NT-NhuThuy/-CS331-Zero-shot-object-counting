"""Full frozen LVIS query-switch H2 diagnostic with labels outside inference.

The inference stage reads only inference_manifest.json.  Ground-truth status and
box matches live in eval_labels.json and are read only after raw scores freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "target_aware_counting"))
from fqie_alpha_smoke import (alpha_mask, context_crop, cosine, load_model,
                              normalized_text, sha, synchronized_transform,
                              write_once)

SRC = ROOT / "target_aware_counting/output/lvis_query_switch_val_20261010_205455"
GT_PATH = SRC / "input_274/query_pairs_coco.json"
PRED_PATH = SRC / "runs/no_pseudo/predictions.json"
IMAGE_DIR = ROOT / "data/lvis_v1_val_source/selected_images"
DEST = ROOT / "target_aware_counting/output/fqie_h2_alpha_lvis_full_20261104_v1"
GT_SHA = "c884b0c75fba8545087b2758c7a218425f4eea80b68ea12b255b229e80d0adf7"
PRED_SHA = "3fe7fd1775c7cbbd2dce72b7c2fcccd59d2ce26a96604fdf83a8039835735e72"


def xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb > inter else 0.0


def prepare():
    if sha(GT_PATH) != GT_SHA or sha(PRED_PATH) != PRED_SHA:
        raise RuntimeError("frozen LVIS artifact hash mismatch")
    gt, pred = json.loads(GT_PATH.read_text()), json.loads(PRED_PATH.read_text())
    images = {x["id"]: x for x in gt["images"]}
    pred_images = {x["id"]: x for x in pred["images"]}
    if images != pred_images:
        raise RuntimeError("prediction image metadata differs from GT contract")
    gt_by_image = defaultdict(list)
    for a in gt["annotations"]:
        gt_by_image[a["image_id"]].append(xyxy(a["bbox"]))
    inference, labels = [], []
    for a in sorted(pred["annotations"], key=lambda x: x["id"]):
        im = images[a["image_id"]]
        path = IMAGE_DIR / im["file_name"]
        if not path.is_file():
            raise RuntimeError(f"missing image: {path}")
        box = xyxy(a["bbox"])
        overlaps = [iou(box, b) for b in gt_by_image[a["image_id"]]]
        max_iou = max(overlaps, default=0.0)
        candidate_id = f"lvis_qs:{a['image_id']}:{a['id']}"
        inference.append({"candidate_id": candidate_id, "source_image_id": im["source_image_id"],
                          "query_image_id": a["image_id"], "image_path": str(path),
                          "image_sha256": sha(path), "query": im["query_name"],
                          "box_xyxy": box, "native_score": a["score"]})
        # Unmatched proposals from a present pair are UNKNOWN, never auto-negative.
        label = ("TARGET_IOU50" if im["query_status"] == "present" and max_iou >= 0.5 else
                 "VERIFIED_ABSENT" if im["query_status"] == "verified_absent" else "UNKNOWN")
        labels.append({"candidate_id": candidate_id, "source_image_id": im["source_image_id"],
                       "query_image_id": a["image_id"], "query_status": im["query_status"],
                       "gt_count": im["gt_count"], "max_iou": max_iou, "eval_label": label})
    if len(inference) != 306 or len(labels) != 306:
        raise RuntimeError("unexpected candidate count")
    write_once(DEST / "inference_manifest.json", {
        "status": "LOCKED_BEFORE_ALPHA_INFERENCE", "split": "LVIS_V1_VAL_DEVELOPMENT",
        "prediction_sha256": PRED_SHA, "items": inference,
        "primary_view": "context_1.5x", "secondary_view": "full_letterbox",
        "inference_reads_eval_labels": False, "max_candidates": 306})
    write_once(DEST / "eval_labels.json", {
        "status": "LOCKED_BEFORE_ALPHA_INFERENCE", "gt_sha256": GT_SHA,
        "positive_rule": "present query and proposal IoU >= 0.5 with official query-category GT box",
        "negative_rule": "official LVIS neg_category_ids via frozen verified_absent query status",
        "unknown_rule": "present-query proposal below IoU 0.5; excluded from semantic classification metrics",
        "items": labels})
    counts = {k: sum(x["eval_label"] == k for x in labels)
              for k in ("TARGET_IOU50", "VERIFIED_ABSENT", "UNKNOWN")}
    print(json.dumps({"candidates": len(inference), "labels": counts,
                      "inference_sha256": sha(DEST / "inference_manifest.json"),
                      "labels_sha256": sha(DEST / "eval_labels.json")}))


def run():
    import torch
    from PIL import Image
    manifest = json.loads((DEST / "inference_manifest.json").read_text())
    if manifest["prediction_sha256"] != PRED_SHA or len(manifest["items"]) != 306:
        raise RuntimeError("inference lock mismatch")
    torch.manual_seed(0)
    model, ac = load_model()
    resolution = int(model.visual.input_resolution)
    text_cache = {}
    started = time.monotonic()
    with torch.inference_mode():
        for item in manifest["items"]:
            out = DEST / "scores" / (item["candidate_id"].replace(":", "_") + ".json")
            if out.exists():
                saved = json.loads(out.read_text())
                if saved["candidate_id"] != item["candidate_id"]:
                    raise RuntimeError(f"resume mismatch: {out}")
                continue
            image = Image.open(item["image_path"]).convert("RGB")
            if sha(Path(item["image_path"])) != item["image_sha256"]:
                raise RuntimeError(f"image changed: {item['candidate_id']}")
            query = item["query"]
            if query not in text_cache:
                text_cache[query] = model.encode_text(ac.tokenize([normalized_text(query)], truncate=True).cuda())
            mask = alpha_mask(image.size, item["box_xyxy"])
            views = {}
            for view in ("full", "context"):
                if view == "full":
                    rgb, alpha = image, mask
                else:
                    rgb, bounds = context_crop(image, item["box_xyxy"])
                    alpha = mask.crop(bounds)
                rt, at, pixels = synchronized_transform(rgb, alpha, resolution)
                if pixels == 0:
                    raise RuntimeError(f"alpha vanished: {item['candidate_id']} {view}")
                visual = model.visual(rt.unsqueeze(0).to(model.dtype).cuda(),
                                      at.unsqueeze(0).to(model.dtype).cuda())
                views[view] = {"alpha_pixels": pixels,
                               "target_cosine": cosine(visual, text_cache[query])}
            write_once(out, {"candidate_id": item["candidate_id"], "source_image_id": item["source_image_id"],
                             "query_image_id": item["query_image_id"], "query": query,
                             "native_score": item["native_score"], "views": views,
                             "semantic_decision": "NOT_MADE"})
        print(json.dumps({"candidates": len(manifest["items"]), "wall_seconds": time.monotonic()-started,
                          "peak_vram_gib": torch.cuda.max_memory_allocated()/1024**3}))


def freeze():
    manifest = json.loads((DEST / "inference_manifest.json").read_text())
    hashes = {}
    for item in manifest["items"]:
        p = DEST / "scores" / (item["candidate_id"].replace(":", "_") + ".json")
        d = json.loads(p.read_text())
        if d["candidate_id"] != item["candidate_id"]:
            raise RuntimeError("score binding mismatch")
        for view in d["views"].values():
            if view["alpha_pixels"] <= 0 or not math.isfinite(view["target_cosine"]):
                raise RuntimeError("invalid raw score")
        hashes[str(p.relative_to(DEST))] = sha(p)
    write_once(DEST / "freeze_scores.json", {"status": "COMPLETE_RAW_SCORES_NO_DECISION",
        "inference_manifest_sha256": sha(DEST / "inference_manifest.json"),
        "score_file_hashes": hashes})
    print(json.dumps({"frozen": len(hashes), "status": "COMPLETE_RAW_SCORES_NO_DECISION"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "freeze"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "freeze": freeze}[args.action]()
