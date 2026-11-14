"""GT-free visual cards for every B0-added image in the valid paired cohort."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "target_aware_counting/output"
A8 = OUT / "a8_weak_singleton_val_20261102_v1/predictions.jsonl"
B0 = OUT / "b0_a8_paired_val_sm86_20261103_v2/predictions_b0.jsonl"
MANIFEST = OUT / "b0_a8_paired_val_sm86_20261103_v2/input_queries.jsonl"
DEST = OUT / "fqie_gate0_20261104_v1/audit/addition_review"


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    manifest, old, new = rows(MANIFEST), rows(A8), rows(B0)
    cards, key = [], []
    for m, a, b in zip(manifest, old, new):
        if not b["additions"] and not a["a4_add"]:
            continue
        assert m["item_id"] == a["item_id"] == b["item_id"]
        image = Image.open(m["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        candidates = []
        # Alternating colors deliberately have no stable source meaning.
        for idx, (source, item) in enumerate([("A4", x) for x in a["a4_add"]] +
                                             [("B0", x) for x in b["additions"]]):
            color = "#ff3b30" if idx % 2 == 0 else "#007aff"
            box = item["box_xyxy"]
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0], box[1]), str(idx), fill="#ffffff", stroke_fill="#000000", stroke_width=2)
            candidates.append({"review_index": idx, "box_xyxy": box})
            key.append({"item_id": m["item_id"], "review_index": idx,
                        "source": source, "candidate_id": item["candidate_id"]})
        card = DEST / "cards" / f"{m['source_image_id']}.png"
        card.parent.mkdir(parents=True, exist_ok=True)
        if card.exists():
            raise RuntimeError(f"card exists: {card}")
        image.save(card)
        cards.append({"item_id": m["item_id"], "query": m["canonical"],
                      "card": str(card), "candidates": candidates,
                      "labels_status": "UNREVIEWED"})
    for name, values in (("review_manifest.jsonl", cards), ("source_key.jsonl", key)):
        path = DEST / name
        content = "".join(json.dumps(x, separators=(",", ":")) + "\n" for x in values)
        if path.exists() and path.read_text() != content:
            raise RuntimeError(f"locked output differs: {path}")
        path.write_text(content)
    print(json.dumps({"image_cards": len(cards), "candidate_boxes": len(key),
                      "scope": "A4/B0 additions only; no GT, no M0 boxes"}))


if __name__ == "__main__":
    main()
