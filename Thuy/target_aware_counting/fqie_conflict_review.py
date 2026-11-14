"""GT-free A4/B0 cross-source conflict sampling and visual review cards."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "target_aware_counting/output"
A8 = OUT / "a8_weak_singleton_val_20261102_v1/predictions.jsonl"
B0 = OUT / "b0_a8_paired_val_sm86_20261103_v2/predictions_b0.jsonl"
MANIFEST = OUT / "b0_a8_paired_val_sm86_20261103_v2/input_queries.jsonl"
DEST = OUT / "fqie_gate0_20261104_v1/audit/conflict_review"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def metrics(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa, ab = area(a), area(b)
    iou = inter / (aa + ab - inter) if aa + ab > inter else 0
    iom = inter / min(aa, ab) if min(aa, ab) > 0 else 0
    cax, cay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    cbx, cby = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    dist = math.hypot(cax - cbx, cay - cby) / max(1, math.sqrt(max(aa, ab)))
    return iou, iom, dist


def write_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"locked output differs: {path}")
    else:
        path.write_bytes(content)


def main():
    manifest, old, new = rows(MANIFEST), rows(A8), rows(B0)
    assert len(manifest) == len(old) == len(new) == 100
    pairs = []
    for m, a, b in zip(manifest, old, new):
        assert m["item_id"] == a["item_id"] == b["item_id"]
        for ac in a["a4_add"]:
            for bc in b["additions"]:
                x, y = ac["box_xyxy"], bc["box_xyxy"]
                iou, iom, distance = metrics(x, y)
                if iou > 0 or distance < 1.5:
                    pairs.append({"item_id": m["item_id"], "source_image_id": m["source_image_id"],
                                  "image_path": m["image_path"], "query": m["canonical"],
                                  "a_id": ac["candidate_id"], "b_id": bc["candidate_id"],
                                  "a_box": x, "b_box": y, "iou": iou, "iom": iom,
                                  "center_distance_norm": distance})
    # Deterministic GT-free selections. One pair per item and bin prevents a
    # single dense source from filling the visual review set.
    bins = {
        "high_iou": lambda p: p["iou"] >= .5,
        "part_like": lambda p: p["iom"] >= .8 and p["iou"] < .5,
        "touching": lambda p: p["iou"] < .1 and p["center_distance_norm"] < 1,
    }
    chosen = []
    for name, predicate in bins.items():
        pool = [p for p in pairs if predicate(p)]
        pool.sort(key=lambda p: hashlib.sha256((name + p["item_id"] + p["a_id"] + p["b_id"]).encode()).hexdigest())
        seen = set()
        for p in pool:
            if p["item_id"] in seen:
                continue
            chosen.append((name, p)); seen.add(p["item_id"])
            if len(seen) == 6:
                break
    cards, key = [], []
    for idx, (kind, p) in enumerate(chosen):
        review_id = f"C{idx:03d}"
        image = Image.open(p["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle(p["a_box"], outline="#ff3b30", width=3)
        draw.rectangle(p["b_box"], outline="#007aff", width=3)
        card = DEST / "cards" / f"{review_id}.png"
        card.parent.mkdir(parents=True, exist_ok=True)
        if card.exists():
            raise RuntimeError(f"card already exists: {card}")
        image.save(card)
        cards.append({"review_id": review_id, "kind": kind, "query": p["query"],
                      "card": str(card), "red_box": p["a_box"], "blue_box": p["b_box"],
                      "iou": round(p["iou"], 4), "iom": round(p["iom"], 4),
                      "center_distance_norm": round(p["center_distance_norm"], 4),
                      "review_label": "UNREVIEWED"})
        key.append({"review_id": review_id, "item_id": p["item_id"],
                    "red_source": "A4", "blue_source": "B0",
                    "red_candidate_id": p["a_id"], "blue_candidate_id": p["b_id"]})
    write_once(DEST / "review_manifest.jsonl",
               ("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in cards)).encode())
    write_once(DEST / "source_key.jsonl",
               ("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in key)).encode())
    summary = {"pairs_screened": len(pairs), "cards": len(cards),
               "high_iou_pairs": sum(p["iou"] >= .5 for p in pairs),
               "part_like_pairs": sum(p["iom"] >= .8 and p["iou"] < .5 for p in pairs),
               "touching_pairs": sum(p["iou"] < .1 and p["center_distance_norm"] < 1 for p in pairs),
               "label_status": "UNREVIEWED; geometry is not same-instance truth"}
    write_once(DEST / "summary.json", (json.dumps(summary, indent=2) + "\n").encode())
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
