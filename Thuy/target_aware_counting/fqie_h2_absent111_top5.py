"""Bounded Alpha-CLIP transfer diagnostic on top-5 M0 boxes per absent query."""
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

MANIFEST = ROOT / "target_aware_counting/output/sink_target_selection_20261029_v1/fsc_verified_absent111.jsonl"
PRED = ROOT / "target_aware_counting/output/spatial_reseeding_fsc111_20261102_v1/predictions_absent111_m0.jsonl"
DEST = ROOT / "target_aware_counting/output/fqie_h2_alpha_absent111_top5_20261104_v1"
MANIFEST_SHA = "161ada4aedaa81b028eccb7faa96329f746e39f1e081255b8cc1af9e8c61955e"
PRED_SHA = "8ace7bae36346750e211303d983f554d66cd7f6c711eeb8540d3d23146c35573"


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def prepare():
    if sha(MANIFEST) != MANIFEST_SHA or sha(PRED) != PRED_SHA:
        raise RuntimeError("absent111 frozen source changed")
    meta = {x["item_id"]: x for x in rows(MANIFEST)}
    pred = rows(PRED)
    items, total = [], 0
    for row in pred:
        m = meta[row["item_id"]]
        if row["count"] != len(row["final_boxes"]) or row["count"] != len(row["final_scores"]):
            raise RuntimeError("box/score binding mismatch")
        total += row["count"]
        ranked = sorted(zip(row["final_scores"], row["final_boxes"]),
                        key=lambda z: (-z[0], z[1]["source_id"]))[:5]
        if not ranked:
            raise RuntimeError("unexpected zero-candidate absent query")
        path = Path(m["image_path"])
        for rank, (score, box) in enumerate(ranked, 1):
            items.append({"candidate_id": f"{row['query_id']}:top{rank}", "item_id": row["item_id"],
                          "source_image_id": row["source_image_id"], "image_path": str(path),
                          "image_sha256": sha(path), "query": row["query"],
                          "native_score": score, "box_xyxy": box["bbox_xyxy_float"],
                          "native_rank": rank})
    # One locked query has only two M0 boxes, hence 552 rather than 555.
    if len(pred) != 111 or len(items) != 552 or total != 6942:
        raise RuntimeError("unexpected absent111 coverage")
    write_once(DEST / "inference_manifest.json", {
        "status": "LOCKED_BEFORE_ALPHA_INFERENCE", "cohort": "FSC_VERIFIED_ABSENT111_DEVELOPMENT",
        "manifest_sha256": MANIFEST_SHA, "predictions_sha256": PRED_SHA,
        "selection": "top_5_by_frozen_native_score_per_query; tie_break_source_id",
        "view": "full_letterbox_only_selected_after_LVIS_development_diagnostic",
        "total_frozen_candidates": total, "selected_candidates": len(items),
        "unscored_candidates": total-len(items), "items": items})
    print(json.dumps({"queries": len(pred), "selected": len(items), "total": total,
                      "manifest_sha256": sha(DEST / "inference_manifest.json")}))


def run():
    import torch
    from PIL import Image
    manifest = json.loads((DEST / "inference_manifest.json").read_text())
    if len(manifest["items"]) != 552 or manifest["predictions_sha256"] != PRED_SHA:
        raise RuntimeError("inference manifest mismatch")
    torch.manual_seed(0)
    model, ac = load_model()
    resolution = int(model.visual.input_resolution)
    text_cache = {}
    started = time.monotonic()
    with torch.inference_mode():
        for item in manifest["items"]:
            out = DEST / "scores" / (item["candidate_id"].replace(":", "_") + ".json")
            if out.exists():
                if json.loads(out.read_text())["candidate_id"] != item["candidate_id"]:
                    raise RuntimeError("resume mismatch")
                continue
            image = Image.open(item["image_path"]).convert("RGB")
            if sha(Path(item["image_path"])) != item["image_sha256"]:
                raise RuntimeError("image hash mismatch")
            query = item["query"]
            if query not in text_cache:
                text_cache[query] = model.encode_text(ac.tokenize([normalized_text(query)], truncate=True).cuda())
            rt, at, pixels = synchronized_transform(image, alpha_mask(image.size, item["box_xyxy"]), resolution)
            if pixels == 0:
                raise RuntimeError("alpha vanished")
            visual = model.visual(rt.unsqueeze(0).to(model.dtype).cuda(), at.unsqueeze(0).to(model.dtype).cuda())
            write_once(out, {"candidate_id": item["candidate_id"], "item_id": item["item_id"],
                             "source_image_id": item["source_image_id"], "native_rank": item["native_rank"],
                             "native_score": item["native_score"], "alpha_pixels": pixels,
                             "alpha_full": cosine(visual, text_cache[query]), "semantic_decision": "NOT_MADE"})
        print(json.dumps({"candidates": len(manifest["items"]), "wall_seconds": time.monotonic()-started,
                          "peak_vram_gib": torch.cuda.max_memory_allocated()/1024**3}))


def freeze():
    manifest = json.loads((DEST / "inference_manifest.json").read_text())
    hashes = {}
    for item in manifest["items"]:
        p = DEST / "scores" / (item["candidate_id"].replace(":", "_") + ".json")
        d = json.loads(p.read_text())
        if d["candidate_id"] != item["candidate_id"] or d["alpha_pixels"] <= 0 or not math.isfinite(d["alpha_full"]):
            raise RuntimeError("invalid score")
        hashes[str(p.relative_to(DEST))] = sha(p)
    write_once(DEST / "freeze_scores.json", {"status": "COMPLETE_RAW_SCORES_NO_DECISION",
        "inference_manifest_sha256": sha(DEST / "inference_manifest.json"), "score_file_hashes": hashes})
    print(json.dumps({"frozen": len(hashes)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("prepare", "run", "freeze"))
    {"prepare": prepare, "run": run, "freeze": freeze}[parser.parse_args().action]()
