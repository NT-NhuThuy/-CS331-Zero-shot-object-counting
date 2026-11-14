"""GT-free M0/B0 conflict cards across categories for H3 audit."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "target_aware_counting"))
from fqie_conflict_review import metrics, rows, write_once  # noqa: E402

OUT = ROOT / "target_aware_counting/output"
DEST = OUT / "fqie_gate0_20261104_v1/audit/m0_b0_conflict_review"


def main():
    manifest = rows(OUT / "b0_a8_paired_val_sm86_20261103_v2/input_queries.jsonl")
    a8 = rows(OUT / "a8_weak_singleton_val_20261102_v1/predictions.jsonl")
    b0 = rows(OUT / "b0_a8_paired_val_sm86_20261103_v2/predictions_b0.jsonl")
    pairs = []
    for m, a, b in zip(manifest, a8, b0):
        assert m["item_id"] == a["item_id"] == b["item_id"]
        for mc in a["m0"]:
            for bc in b["additions"]:
                x, y = mc["box_xyxy"], bc["box_xyxy"]
                iou, iom, distance = metrics(x, y)
                if iou > 0 or distance < 1.5:
                    pairs.append({"item_id": m["item_id"], "query": m["canonical"],
                                  "image_path": m["image_path"], "m_id": mc["candidate_id"],
                                  "b_id": bc["candidate_id"], "m_box": x, "b_box": y,
                                  "iou": iou, "iom": iom, "distance": distance})
    bins = {
        "high_iou": lambda p: p["iou"] >= .5,
        "part_like": lambda p: p["iom"] >= .8 and p["iou"] < .5,
        "touching": lambda p: p["iou"] < .1 and p["distance"] < 1,
    }
    selected = []
    for name, predicate in bins.items():
        pool = [p for p in pairs if predicate(p)]
        pool.sort(key=lambda p: hashlib.sha256((name+p["item_id"]+p["m_id"]+p["b_id"]).encode()).hexdigest())
        by_query = Counter()
        seen_items = set()
        for p in pool:
            if p["item_id"] in seen_items or by_query[p["query"]] >= 2:
                continue
            selected.append((name, p)); seen_items.add(p["item_id"]); by_query[p["query"]] += 1
            if len(seen_items) >= 8:
                break
    cards, keys = [], []
    for i, (name, p) in enumerate(selected):
        rid = f"M{i:03d}"
        image = Image.open(p["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle(p["m_box"], outline="#ff3b30", width=3)
        draw.rectangle(p["b_box"], outline="#007aff", width=3)
        card = DEST / "cards" / f"{rid}.png"
        card.parent.mkdir(parents=True, exist_ok=True)
        if card.exists():
            raise RuntimeError(f"card exists: {card}")
        image.save(card)
        cards.append({"review_id": rid, "kind": name, "query": p["query"],
                      "card": str(card), "red_box": p["m_box"], "blue_box": p["b_box"],
                      "iou": round(p["iou"], 4), "iom": round(p["iom"], 4),
                      "center_distance_norm": round(p["distance"], 4),
                      "review_label": "UNREVIEWED"})
        keys.append({"review_id": rid, "item_id": p["item_id"], "red_source": "M0",
                     "blue_source": "B0", "red_id": p["m_id"], "blue_id": p["b_id"]})
    write_once(DEST / "review_manifest.jsonl",
               ("".join(json.dumps(x, separators=(",", ":"))+"\n" for x in cards)).encode())
    write_once(DEST / "source_key.jsonl",
               ("".join(json.dumps(x, separators=(",", ":"))+"\n" for x in keys)).encode())
    print(json.dumps({"pairs_screened": len(pairs), "cards": len(cards),
                      "queries": dict(Counter(x["query"] for x in cards)),
                      "status": "UNREVIEWED"}))


if __name__ == "__main__":
    main()
