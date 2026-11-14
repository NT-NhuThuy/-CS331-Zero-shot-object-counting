"""Build a GT-free complete FSC development manifest from frozen predictions."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .run import sha, write_once

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "target_aware_counting/output"
PRIMARY = OUT / "a8_weak_singleton_val_20261102_v1/predictions.jsonl"
PRIMARY_META = OUT / "b0_a8_paired_val_sm86_20261103_v2/input_queries.jsonl"
ABSENT = OUT / "spatial_reseeding_fsc111_20261102_v1/predictions_absent111_m0.jsonl"
ABSENT_META = OUT / "sink_target_selection_20261029_v1/fsc_verified_absent111.jsonl"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(destination: Path) -> dict:
    primary, primary_meta = rows(PRIMARY), {x["item_id"]: x for x in rows(PRIMARY_META)}
    absent, absent_meta = rows(ABSENT), {x["item_id"]: x for x in rows(ABSENT_META)}
    if len(primary) != 100 or len(primary_meta) != 100 or len(absent) != 111 or len(absent_meta) != 111:
        raise RuntimeError("frozen cohort denominator changed")
    items, candidates = [], []
    for row in primary:
        meta = primary_meta[row["item_id"]]
        if row["source_image_id"] != meta["source_image_id"] or row["query"] != meta["canonical"]:
            raise RuntimeError("primary item binding mismatch")
        m0, add = row["m0"], row["a4_add"]
        if row["m0_count"] != len(m0) or row["a4_count"] != len(m0) + len(add):
            raise RuntimeError("primary count parity mismatch")
        if m0 and add:
            raise RuntimeError("A4 reactivated on nonzero M0")
        item = {"item_id": row["item_id"], "source_image_id": str(row["source_image_id"]),
                "query": row["query"], "image_path": meta["image_path"],
                "image_sha256": meta["image_sha256"], "base_mode": "A4_FROZEN",
                "m0_count": len(m0), "a4_count": len(m0) + len(add)}
        items.append(item)
        for branch, entries in (("M0", m0), ("A4", add)):
            for c in entries:
                candidates.append({**item, "candidate_id": c["candidate_id"], "branch": branch,
                                   "box_xyxy": c["box_xyxy"], "native_score": c.get("score")})
    for row in absent:
        meta = absent_meta[row["item_id"]]
        if str(row["source_image_id"]) != str(meta["source_image_id"]) or row["query"] != meta["query"]:
            raise RuntimeError("absent item binding mismatch")
        if row["count"] != len(row["final_boxes"]) or row["count"] != len(row["final_scores"]):
            raise RuntimeError("absent count parity mismatch")
        item = {"item_id": row["item_id"], "source_image_id": str(row["source_image_id"]),
                "query": row["query"], "image_path": meta["image_path"],
                "image_sha256": sha(Path(meta["image_path"])), "base_mode": "M0_FROZEN",
                "m0_count": row["count"], "a4_count": row["count"]}
        items.append(item)
        for box, score in zip(row["final_boxes"], row["final_scores"]):
            candidates.append({**item, "candidate_id": box["source_id"], "branch": "M0",
                               "box_xyxy": box["bbox_xyxy_float"], "native_score": score})
    ids = [x["candidate_id"] for x in candidates]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate candidate ID")
    for item in items:
        if sha(Path(item["image_path"])) != item["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {item['item_id']}")
    by_source = Counter(x["source_image_id"] for x in items)
    payload = {"status": "FROZEN_GT_FREE_FSC_DEVELOPMENT", "items": candidates,
               "query_items": items, "source_overlap": sorted(k for k, n in by_source.items() if n > 1),
               "source_hashes": {name: sha(path) for name, path in
                                 (("primary_predictions", PRIMARY), ("primary_meta", PRIMARY_META),
                                  ("absent_predictions", ABSENT), ("absent_meta", ABSENT_META))},
               "counts": {"query_items": len(items), "candidates": len(candidates),
                          "primary_candidates": sum(x["base_mode"] == "A4_FROZEN" for x in candidates),
                          "absent_candidates": sum(x["base_mode"] == "M0_FROZEN" for x in candidates)}}
    write_once(destination, payload)
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    result = build(p.parse_args().output)
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
