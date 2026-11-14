"""Resume-safe frozen CycleGRPO/Alpha-CLIP extraction for the FSC manifest.

Each stage processes every candidate. This module never reads evaluation labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

from .parse_descriptor import parse
from .run import sha, write_once

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "target_aware_counting/output/cyclegrpo_capability_probe_20261027_v1"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(ROOT / "target_aware_counting"))

PROMPT = ("Describe the visible content of the indicated region. Name the main visible object "
          "if identifiable. If the region contains several objects or only a part, state that explicitly.")


def preflight(stage: str) -> None:
    if stage in {"encode", "caption", "freeze"}:
        from runtime_gate import inspect_lock
        report = inspect_lock(OLD / "runtime_lock.json")
        if report["blockers"]:
            raise RuntimeError(f"CycleGRPO runtime lock failed: {report['blockers']}")
    if stage in {"compare", "freeze"}:
        from fqie_alpha_smoke import BASE, VISION
        expected = ((BASE, "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"),
                    (VISION, "42c621bb5bac89a511ab625a878f026c11de4de8cf7abf52db5dfd11862bdb8e"))
        for path, digest in expected:
            if sha(path) != digest:
                raise RuntimeError(f"Alpha-CLIP checkpoint hash mismatch: {path}")


def file_for(output: Path, stage: str, candidate_id: str) -> Path:
    return output / stage / (hashlib.sha256(candidate_id.encode()).hexdigest() + ".json")


def clip_box(box: list[float], width: int, height: int) -> list[float]:
    if len(box) != 4 or width <= 0 or height <= 0:
        raise ValueError("invalid box or image size")
    clipped = [max(0.0, min(float(width), float(box[0]))),
               max(0.0, min(float(height), float(box[1]))),
               max(0.0, min(float(width), float(box[2]))),
               max(0.0, min(float(height), float(box[3])))]
    if not all(math.isfinite(v) for v in clipped) or clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("empty/nonfinite box after clipping")
    return clipped


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if data.get("status") != "FROZEN_GT_FREE_FSC_DEVELOPMENT":
        raise RuntimeError("unlocked manifest")
    items = data["items"]
    if len(items) != data["counts"]["candidates"] or len({x["candidate_id"] for x in items}) != len(items):
        raise RuntimeError("manifest completeness/binding mismatch")
    images = {}
    for item in items:
        image = item["image_path"]
        if image in images and images[image] != item["image_sha256"]:
            raise RuntimeError("same image has conflicting hashes")
        images[image] = item["image_sha256"]
    for image, digest in images.items():
        if sha(Path(image)) != digest:
            raise RuntimeError(f"manifest image changed: {image}")
    return items


def checked_existing(path: Path, item: dict, manifest_hash: str) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    if data.get("candidate_id") != item["candidate_id"] or data.get("manifest_sha256") != manifest_hash:
        raise RuntimeError(f"resume binding mismatch: {path}")
    return True


def encode(items: list[dict], manifest_hash: str, output: Path) -> None:
    import torch
    from PIL import Image
    from run_native_smoke import load_vq
    from native_bridge import encode_rectangle_region_native
    lock = json.loads((OLD / "runtime_lock.json").read_text())
    model = load_vq(lock)
    validated_images = set()
    try:
        with torch.inference_mode():
            for i, item in enumerate(items):
                path = file_for(output, "encodings", item["candidate_id"])
                if checked_existing(path, item, manifest_hash):
                    continue
                if item["image_path"] not in validated_images:
                    if sha(Path(item["image_path"])) != item["image_sha256"]:
                        raise RuntimeError(f"image changed: {item['candidate_id']}")
                    validated_images.add(item["image_path"])
                width, height = Image.open(item["image_path"]).size
                clipped = clip_box(item["box_xyxy"], width, height)
                result = encode_rectangle_region_native(item["image_path"], clipped, model)
                write_once(path, {"candidate_id": item["candidate_id"], "manifest_sha256": manifest_hash,
                                  "runtime_lock_sha256": sha(OLD / "runtime_lock.json"),
                                  "original_box_xyxy": item["box_xyxy"],
                                  "box_clipped_to_image": clipped != item["box_xyxy"], **result})
                if i % 100 == 0:
                    print(json.dumps({"stage": "encode", "index": i + 1, "total": len(items)}), flush=True)
    finally:
        del model


def caption(items: list[dict], manifest_hash: str, output: Path) -> None:
    import torch
    from run_native_smoke import load_qwen
    from native_bridge import generate_native_message
    from capability_parser import clean_jpeg_data_url, native_caption_message
    lock = json.loads((OLD / "runtime_lock.json").read_text())
    model, processor = load_qwen(lock)
    try:
        with torch.inference_mode():
            for i, item in enumerate(items):
                path = file_for(output, "captions", item["candidate_id"])
                if checked_existing(path, item, manifest_hash):
                    continue
                ep = file_for(output, "encodings", item["candidate_id"])
                if not checked_existing(ep, item, manifest_hash):
                    raise RuntimeError(f"missing encoding: {item['candidate_id']}")
                enc = json.loads(ep.read_text())
                message = native_caption_message(enc["mask_tokens"], clean_jpeg_data_url(item["image_path"]))
                message[0]["content"][1]["text"] = PROMPT + " " + enc["mask_tokens"]
                raw = generate_native_message(model, processor, message, 96)
                response = raw["raw_response"].split("</think>")[-1].strip()
                descriptor, status = parse(response, truncated=raw["output_tokens"] >= 96)
                write_once(path, {"candidate_id": item["candidate_id"], "manifest_sha256": manifest_hash,
                                  "encoding_sha256": sha(ep), "prompt_text": message[0]["content"][1]["text"],
                                  "raw_response": raw["raw_response"], "output_tokens": raw["output_tokens"],
                                  "generation_seconds": raw["generation_seconds"],
                                  "descriptor": descriptor, "caption_status": status})
                if i % 100 == 0:
                    print(json.dumps({"stage": "caption", "index": i + 1, "total": len(items)}), flush=True)
    finally:
        del model, processor


def compare(items: list[dict], manifest_hash: str, output: Path) -> None:
    import torch
    from PIL import Image
    from fqie_alpha_smoke import alpha_mask, context_crop, cosine, load_model, normalized_text, synchronized_transform
    model, ac = load_model()
    resolution = int(model.visual.input_resolution)
    text_cache = {}
    validated_images = set()
    try:
        with torch.inference_mode():
            for i, item in enumerate(items):
                path = file_for(output, "scores", item["candidate_id"])
                if checked_existing(path, item, manifest_hash):
                    continue
                cp = file_for(output, "captions", item["candidate_id"])
                if not checked_existing(cp, item, manifest_hash):
                    raise RuntimeError(f"missing caption: {item['candidate_id']}")
                cap = json.loads(cp.read_text())
                if cap["caption_status"] != "VALID":
                    write_once(path, {"candidate_id": item["candidate_id"], "manifest_sha256": manifest_hash,
                                      "caption_sha256": sha(cp), "views": {}, "score_status": "UNKNOWN_CAPTION"})
                    continue
                image = Image.open(item["image_path"]).convert("RGB")
                if item["image_path"] not in validated_images:
                    if sha(Path(item["image_path"])) != item["image_sha256"]:
                        raise RuntimeError(f"image changed: {item['candidate_id']}")
                    validated_images.add(item["image_path"])
                query, descriptor = item["query"], cap["descriptor"]
                for value in (query, descriptor):
                    if value not in text_cache:
                        text_cache[value] = model.encode_text(ac.tokenize([normalized_text(value)], truncate=True).cuda())
                mask = alpha_mask(image.size, item["box_xyxy"])
                views = {}
                for name in ("full", "context"):
                    if name == "full":
                        rgb, alpha = image, mask
                    else:
                        rgb, bounds = context_crop(image, item["box_xyxy"])
                        alpha = mask.crop(bounds)
                    rt, at, pixels = synchronized_transform(rgb, alpha, resolution)
                    if pixels <= 0:
                        raise RuntimeError(f"empty transformed region: {item['candidate_id']}")
                    visual = model.visual(rt.unsqueeze(0).to(model.dtype).cuda(),
                                          at.unsqueeze(0).to(model.dtype).cuda())
                    views[name] = {"query_cosine": cosine(visual, text_cache[query]),
                                   "descriptor_cosine": cosine(visual, text_cache[descriptor]),
                                   "alpha_pixels": pixels}
                write_once(path, {"candidate_id": item["candidate_id"], "manifest_sha256": manifest_hash,
                                  "caption_sha256": sha(cp), "views": views, "score_status": "VALID"})
                if i % 100 == 0:
                    print(json.dumps({"stage": "compare", "index": i + 1, "total": len(items)}), flush=True)
    finally:
        del model


def freeze(items: list[dict], manifest_hash: str, output: Path) -> None:
    stages = ("encodings", "captions", "scores")
    hashes = {}
    for stage in stages:
        hashes[stage] = {}
        for item in items:
            path = file_for(output, stage, item["candidate_id"])
            if not checked_existing(path, item, manifest_hash):
                raise RuntimeError(f"incomplete {stage}: {item['candidate_id']}")
            hashes[stage][item["candidate_id"]] = sha(path)
    for item in items:
        cid = item["candidate_id"]
        cap = json.loads(file_for(output, "captions", cid).read_text())
        score = json.loads(file_for(output, "scores", cid).read_text())
        if cap.get("encoding_sha256") != hashes["encodings"][cid]:
            raise RuntimeError(f"caption/encoding hash mismatch: {cid}")
        if score.get("caption_sha256") != hashes["captions"][cid]:
            raise RuntimeError(f"score/caption hash mismatch: {cid}")
        if cap.get("prompt_text") != PROMPT + " " + json.loads(file_for(output, "encodings", cid).read_text())["mask_tokens"]:
            raise RuntimeError(f"caption prompt mismatch: {cid}")
    write_once(output / "freeze.json", {"status": "FROZEN_BEFORE_LABEL_JOIN", "manifest_sha256": manifest_hash,
                                        "runtime_lock_sha256": sha(OLD / "runtime_lock.json"), "hashes": hashes,
                                        "candidate_count": len(items), "query_free_caption": True})
    print(json.dumps({"frozen": len(items)}))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("encode", "caption", "compare", "freeze"))
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--max-items", type=int, default=None,
                   help="bounded technical smoke; never used by freeze")
    a = p.parse_args()
    items, manifest_hash = load_manifest(a.manifest), sha(a.manifest)
    if a.max_items is not None:
        if a.stage == "freeze" or not 0 < a.max_items <= len(items):
            raise ValueError("invalid smoke limit")
        items = items[:a.max_items]
    preflight(a.stage)
    started = time.monotonic()
    {"encode": encode, "caption": caption, "compare": compare, "freeze": freeze}[a.stage](items, manifest_hash, a.output)
    print(json.dumps({"stage": a.stage, "wall_seconds": time.monotonic() - started}))


if __name__ == "__main__":
    main()
