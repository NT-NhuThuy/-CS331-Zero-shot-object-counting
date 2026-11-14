"""Same-image query facet stress for Alpha-CLIP, diagnostic only."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "target_aware_counting"))
from fqie_alpha_smoke import (ROOT as ZOC, cosine, context_crop, alpha_mask,
                              synchronized_transform, load_model, normalized_text,
                              sha, write_once)

OUT = ZOC / "target_aware_counting/output"
OLD = OUT / "cyclegrpo_capability_probe_20261027_v1"
DEST = OUT / "fqie_h2_alpha_facet_probe_20261104_v2"
QUERIES = [
    "pickup truck",
    "green pickup truck",
    "white pickup truck",
    "green pickup truck in front of a white pickup truck",
    "white pickup truck behind a green pickup truck",
    "red pickup truck",
]


def main():
    import torch
    from PIL import Image
    case = next(x for x in (json.loads(s) for s in (OLD / "cases_input.jsonl").read_text().splitlines())
                if x["case_id"] == "case_04")
    if sha(Path(case["image_path"])) != case["image_sha256"] or len(case["regions"]) != 3:
        raise RuntimeError("locked pickup case changed")
    locked = {"status": "FIXED_FACET_DIAGNOSTIC", "case_sha256": sha(OLD / "cases_input.jsonl"),
              "case_id": "case_04", "queries": QUERIES,
              "note": "visual labels from prior assistant review; no independent relation/absent GT"}
    write_once(DEST / "manifest.json", locked)
    torch.manual_seed(0)
    model, alpha_clip = load_model()
    resolution = int(model.visual.input_resolution)
    text = [normalized_text(q) for q in QUERIES]
    with torch.inference_mode():
        embeddings = model.encode_text(alpha_clip.tokenize(text, truncate=True).cuda())
        image = Image.open(case["image_path"]).convert("RGB")
        results = []
        for r in case["regions"]:
            mask = alpha_mask(image.size, r["bbox_xyxy"])
            scores = {}
            for view in ("full", "context"):
                if view == "full":
                    rgb, alpha, bounds = image, mask, (0,0,*image.size)
                else:
                    rgb, bounds = context_crop(image, r["bbox_xyxy"])
                    alpha = mask.crop(bounds)
                rt, at, n = synchronized_transform(rgb, alpha, resolution)
                if n == 0:
                    raise RuntimeError("alpha vanished")
                visual = model.visual(rt.unsqueeze(0).to(model.dtype).cuda(),
                                      at.unsqueeze(0).to(model.dtype).cuda())
                scores[view] = {q: cosine(visual, embeddings[i:i+1]) for i,q in enumerate(QUERIES)}
            results.append({"proposal_id": r["proposal_id"], "box_xyxy": r["bbox_xyxy"],
                            "scores": scores})
    write_once(DEST / "scores.json", {"case_id": "case_04", "image_sha256": case["image_sha256"],
                                      "queries": QUERIES, "regions": results,
                                      "semantic_decision": "NOT_MADE"})
    print(json.dumps({"regions": len(results), "queries": len(QUERIES),
                      "peak_vram_gib": torch.cuda.max_memory_allocated()/1024**3}))


if __name__ == "__main__":
    main()
