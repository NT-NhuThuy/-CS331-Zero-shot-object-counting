"""Frozen Alpha-CLIP region/text smoke on the eight locked H2 regions.

Scores are raw cosine evidence. This script never makes a REMOVE decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "target_aware_counting/output"
SOURCE = ROOT / "external/AlphaCLIP"
CHECKPOINTS = ROOT / "checkpoints/alphaclip_official_20261104"
BASE = CHECKPOINTS / "ViT-L-14.pt"
VISION = CHECKPOINTS / "clip_l14_grit20m_fultune_2xe.pth"
INPUT = OUT / "fqie_h2_caption_smoke_20261104_v1"
DEST = OUT / "fqie_h2_alpha_smoke_20261104_v2"
sys.path.insert(0, str(SOURCE))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"locked output differs: {path}")
    path.write_bytes(content)


def preflight():
    import subprocess
    if sha(BASE) != "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836":
        raise RuntimeError("base OpenAI CLIP checksum mismatch")
    vision_sha = sha(VISION)
    if vision_sha != "42c621bb5bac89a511ab625a878f026c11de4de8cf7abf52db5dfd11862bdb8e":
        raise RuntimeError("Alpha-CLIP vision checkpoint changed")
    commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if commit != "ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff":
        raise RuntimeError("Alpha-CLIP source revision changed")
    if subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain",
                                "--untracked-files=no"], text=True).strip():
        raise RuntimeError("Alpha-CLIP source modified")
    manifest = json.loads((INPUT / "manifest.json").read_text())
    caption_freeze = json.loads((INPUT / "freeze_raw.json").read_text())
    if sha(INPUT / "manifest.json") != caption_freeze["manifest_sha256"]:
        raise RuntimeError("caption manifest changed")
    write_once(DEST / "lock.json", {"status": "TECHNICAL_SMOKE_LOCKED",
        "base_clip_sha256": sha(BASE), "alpha_vision_sha256": vision_sha,
        "alpha_source_commit": commit, "caption_freeze_sha256": sha(INPUT / "freeze_raw.json"),
        "input_manifest_sha256": sha(INPUT / "manifest.json"), "items": 8,
        "view_policy": "full_and_1.5x_context_clipped",
        "alpha_policy": "rectangle_raster_same_fit_letterbox_as_rgb",
        "text_template": "a photo of {text}.",
        "descriptor_policy": "strip_think_tag; skip_if_caption_token_ceiling",
        "inference_reads_gt": False, "max_image_text_forwards": 16})
    print(json.dumps({"preflight": "PASS", "items": len(manifest["items"]),
                      "base_sha256": sha(BASE), "alpha_sha256": vision_sha}))


def strip_caption(raw):
    value = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return value if value else None


def normalized_text(value):
    value = value.strip().replace("_", " ").rstrip(" .")
    return "a photo of " + value + "."


def context_crop(image, box, factor=1.5):
    from PIL import Image
    w, h = image.size
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1)/2, (y0 + y1)/2
    cw, ch = max(1, (x1-x0)*factor), max(1, (y1-y0)*factor)
    left = max(0, math.floor(cx-cw/2)); top = max(0, math.floor(cy-ch/2))
    right = min(w, math.ceil(cx+cw/2)); bottom = min(h, math.ceil(cy+ch/2))
    if right <= left or bottom <= top:
        raise ValueError("empty context crop")
    return image.crop((left, top, right, bottom)), (left, top, right, bottom)


def alpha_mask(size, box):
    from PIL import Image, ImageDraw
    w, h = size
    left = max(0, min(w, math.floor(box[0])))
    top = max(0, min(h, math.floor(box[1])))
    right = max(0, min(w, math.ceil(box[2])))
    bottom = max(0, min(h, math.ceil(box[3])))
    if right <= left or bottom <= top:
        raise ValueError("empty region mask")
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle((left, top, right-1, bottom-1), fill=255)
    return mask


def synchronized_transform(rgb, alpha, resolution):
    import torchvision.transforms.functional as F
    from torchvision.transforms import InterpolationMode
    from PIL import Image
    # Fit the complete view inside the square. Center crop of a wide full
    # image can entirely remove an edge proposal, as the v1 facet probe found.
    # RGB/alpha use identical size and offset; only interpolation differs.
    width, height = rgb.size
    scale = resolution / max(width, height)
    new_w = max(1, min(resolution, round(width * scale)))
    new_h = max(1, min(resolution, round(height * scale)))
    rgb_scaled = F.resize(rgb, [new_h, new_w], InterpolationMode.BICUBIC)
    alpha_scaled = F.resize(alpha, [new_h, new_w], InterpolationMode.NEAREST)
    offset = ((resolution-new_w)//2, (resolution-new_h)//2)
    rgb = Image.new("RGB", (resolution, resolution), (123, 117, 104))
    alpha = Image.new("L", (resolution, resolution), 0)
    rgb.paste(rgb_scaled, offset)
    alpha.paste(alpha_scaled, offset)
    rgb_tensor = F.normalize(F.to_tensor(rgb),
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711))
    raw_alpha = F.to_tensor(alpha)
    if raw_alpha.shape != (1, resolution, resolution):
        raise RuntimeError("alpha transform shape mismatch")
    alpha_tensor = F.normalize(raw_alpha, (0.5,), (0.26,))
    return rgb_tensor, alpha_tensor, int((raw_alpha > .5).sum().item())


def synthetic_transform_check():
    from PIL import Image
    rgb = Image.new("RGB", (320, 180), "white")
    mask = alpha_mask(rgb.size, [120, 50, 200, 130])
    r, a, positive = synchronized_transform(rgb, mask, 224)
    assert r.shape == (3, 224, 224) and a.shape == (1, 224, 224)
    assert 0 < positive < 224*224
    crop, bounds = context_crop(rgb, [120, 50, 200, 130])
    cropped_alpha = mask.crop(bounds)
    _, _, cropped_positive = synchronized_transform(crop, cropped_alpha, 224)
    assert cropped_positive > positive
    return {"full_positive_alpha_pixels": positive,
            "context_positive_alpha_pixels": cropped_positive}


def load_model():
    import torch
    import alpha_clip
    from alpha_clip.model import build_model
    # The base is the official SHA-verified OpenAI JIT. The alpha checkpoint
    # is loaded with weights_only and strict visual-key matching.
    base = torch.jit.load(str(BASE), map_location="cpu").eval()
    model = build_model(base.state_dict()).eval()
    alpha_state = torch.load(VISION, map_location="cpu", weights_only=True)
    model.visual.load_state_dict(alpha_state, strict=True)
    del base, alpha_state
    model = model.cuda().eval()
    return model, alpha_clip


def cosine(a, b):
    import torch
    a = a.float() / a.float().norm(dim=-1, keepdim=True)
    b = b.float() / b.float().norm(dim=-1, keepdim=True)
    value = float((a*b).sum(dim=-1).item())
    if not math.isfinite(value) or abs(value) > 1.001:
        raise RuntimeError("invalid cosine")
    return value


def run(ids):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    lock = json.loads((DEST / "lock.json").read_text())
    if lock["base_clip_sha256"] != "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836" or lock["items"] != 8:
        raise RuntimeError("preflight lock missing")
    manifest = json.loads((INPUT / "manifest.json").read_text())["items"]
    selected = [x for x in manifest if x["probe_id"] in ids]
    if len(selected) != len(ids):
        raise RuntimeError("unknown or duplicated probe ID")
    torch.manual_seed(0)
    check = synthetic_transform_check()
    started = time.monotonic()
    model, alpha_clip = load_model()
    resolution = int(model.visual.input_resolution)
    with torch.inference_mode():
        for x in selected:
            out = DEST / "scores" / f"{x['probe_id']}.json"
            if out.exists():
                saved = json.loads(out.read_text())
                if saved["candidate_id"] != x["candidate_id"]:
                    raise RuntimeError("resume candidate mismatch")
                continue
            caption = json.loads((INPUT / "captions" / f"{x['probe_id']}.json").read_text())
            descriptor = strip_caption(caption["raw_response"])
            descriptor_status = ("TRUNCATED" if caption["output_tokens"] == 96 else
                                 "AVAILABLE" if descriptor else "UNKNOWN")
            from PIL import Image
            image = Image.open(x["image_path"]).convert("RGB")
            mask = alpha_mask(image.size, x["box_xyxy"])
            view_data = {}
            texts = [normalized_text(x["query_for_later_audit_only"])]
            if descriptor_status == "AVAILABLE":
                texts.append(normalized_text(descriptor))
            tokens = alpha_clip.tokenize(texts, truncate=True).cuda()
            text_features = model.encode_text(tokens)
            for view in ("full", "context"):
                if view == "full":
                    rgb, alpha, bounds = image, mask, (0, 0, *image.size)
                else:
                    rgb, bounds = context_crop(image, x["box_xyxy"])
                    alpha = mask.crop(bounds)
                rt, at, positive = synchronized_transform(rgb, alpha, resolution)
                if positive == 0:
                    raise RuntimeError(f"alpha vanished after transform: {x['probe_id']} {view}")
                visual = model.visual(rt.unsqueeze(0).to(model.dtype).cuda(),
                                      at.unsqueeze(0).to(model.dtype).cuda())
                target = cosine(visual, text_features[0:1])
                desc = cosine(visual, text_features[1:2]) if len(texts) == 2 else None
                view_data[view] = {"crop_xyxy": bounds,
                    "alpha_positive_pixels_after_transform": positive,
                    "target_cosine": target, "descriptor_cosine": desc,
                    "descriptor_minus_target": desc-target if desc is not None else None}
            margins = [v["descriptor_minus_target"] for v in view_data.values()]
            write_once(out, {"probe_id": x["probe_id"], "candidate_id": x["candidate_id"],
                "image_sha256": x["image_sha256"], "box_xyxy": x["box_xyxy"],
                "query_text": x["query_for_later_audit_only"],
                "descriptor_status": descriptor_status, "descriptor_raw_cleaned": descriptor,
                "texts_encoded": texts, "views": view_data,
                "min_descriptor_margin": min(margins) if all(v is not None for v in margins) else None,
                "semantic_decision": "NOT_MADE", "alpha_kind": "rectangle_region_prompt",
                "transform_check": check})
            print(json.dumps({"scored": x["probe_id"], "descriptor_status": descriptor_status}), flush=True)
    print(json.dumps({"items_requested": len(selected), "wall_seconds": time.monotonic()-started,
                      "peak_vram_gib": torch.cuda.max_memory_allocated()/1024**3}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--ids", nargs="+", default=[f"P{i:02d}" for i in range(8)])
    args = parser.parse_args()
    preflight() if args.action == "preflight" else run(args.ids)
