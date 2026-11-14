"""Read-only Gate 0 audit of the frozen A8/B0 FSC VAL cohort.

Run prepare before evaluate. The review manifest contains no GT-derived fields.
Point coverage is geometric FSC proxy, never semantic or instance identity truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "target_aware_counting/output"
A8 = OUT / "a8_weak_singleton_val_20261102_v1"
B0 = OUT / "b0_a8_paired_val_sm86_20261103_v2"
DATA = ROOT / "data/fscd147/countgd_val_fscd147.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"locked output differs: {path}")
    else:
        path.write_bytes(content)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def validate():
    lock = json.loads((B0 / "input_lock.json").read_text())
    freeze = json.loads((B0 / "freeze.json").read_text())
    afreeze = json.loads((A8 / "freeze.json").read_text())
    assert lock["items"] == 100 and freeze["status"] == "FROZEN_BEFORE_GT"
    assert afreeze["status"] == "FROZEN_BEFORE_GT"
    for path, digest in lock["source_hashes"].items():
        if sha(path) != digest:
            raise RuntimeError(f"input hash mismatch: {path}")
    for path, digest in freeze["input_hashes"].items():
        if sha(path) != digest:
            raise RuntimeError(f"frozen candidate hash mismatch: {path}")
    for field, path in (("input_lock_sha256", B0 / "input_lock.json"),
                        ("predictions_sha256", B0 / "predictions_b0.jsonl"),
                        ("addition_trace_sha256", B0 / "addition_trace.jsonl")):
        if sha(path) != freeze[field]:
            raise RuntimeError(f"freeze hash mismatch: {path}")
    if sha(A8 / "predictions.jsonl") != afreeze["predictions_sha256"]:
        raise RuntimeError("A8 prediction freeze mismatch")
    if sha(DATA) != "6f9bc8d200d74789aa515e0131fd93275713392c64c3124828bdf258858ec807":
        raise RuntimeError("FSC VAL GT hash mismatch")
    manifest = rows(B0 / "input_queries.jsonl")
    old = rows(A8 / "predictions.jsonl")
    new = rows(B0 / "predictions_b0.jsonl")
    if not (len(manifest) == len(old) == len(new) == 100):
        raise RuntimeError("missing items")
    ids = [x["item_id"] for x in manifest]
    if len(set(ids)) != 100 or len({x["source_image_id"] for x in manifest}) != 100:
        raise RuntimeError("duplicate item/source")
    if ids != [x["item_id"] for x in old] or ids != [x["item_id"] for x in new]:
        raise RuntimeError("item order/binding differs")
    for m, a, b in zip(manifest, old, new):
        if (a["m0_count"] != len(a["m0"]) or
            a["a4_count"] != len(a["m0"]) + len(a["a4_add"]) or
            b["b0_count"] != len(b["b0_boxes_xyxy"]) or
            b["m0_count"] != a["m0_count"] or b["a4_count"] != a["a4_count"] or
            b["b0_count"] != a["m0_count"] + len(b["additions"])):
            raise RuntimeError(f"count/binding mismatch: {m['item_id']}")
        if sha(m["image_path"]) != m["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {m['item_id']}")
        if len(a["a8_add"]) != len(a["a4_add"]):
            raise RuntimeError(f"A8 gate changed A4: {m['item_id']}")
    return manifest, old, new


def box_of(candidate):
    return candidate["box_xyxy"]


def contains(box, point):
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def match_count(boxes, points):
    graph = [[j for j, p in enumerate(points) if contains(b, p)] for b in boxes]
    owners = {}

    def augment(i, seen):
        for j in graph[i]:
            if j not in seen:
                seen.add(j)
                if j not in owners or augment(owners[j], seen):
                    owners[j] = i
                    return True
        return False

    for i in range(len(graph)):
        augment(i, set())
    return len(owners)


def prepare(destination):
    manifest, old, new = validate()
    items = []
    for m, a, b in zip(manifest, old, new):
        # GT blind: the reviewer sees query, image, and shuffled candidate provenance keys.
        candidates = []
        for source, entries in (("M0", a["m0"]), ("A4", a["a4_add"]),
                                ("B0", b["additions"])):
            for c in entries:
                candidates.append({"candidate_id": c["candidate_id"],
                                   "box_xyxy": box_of(c), "source_key": source})
        items.append({"item_id": m["item_id"], "source_image_id": m["source_image_id"],
                      "image_path": m["image_path"], "query": m["canonical"],
                      "m0_zero": not a["m0"], "candidates": candidates})
    write_once(destination / "manifest" / "review_gt_blind.jsonl",
               ("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in items)).encode())
    blind = []
    key = []
    for item in items:
        entry = {k: v for k, v in item.items() if k != "candidates"}
        entry["candidates"] = []
        for c in item["candidates"]:
            alias = hashlib.sha256(c["candidate_id"].encode()).hexdigest()[:16]
            entry["candidates"].append({"review_id": alias, "box_xyxy": c["box_xyxy"]})
            key.append({"item_id": item["item_id"], "review_id": alias,
                        "candidate_id": c["candidate_id"], "source": c["source_key"]})
        blind.append(entry)
    write_once(destination / "manifest" / "review_source_blind.jsonl",
               ("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in blind)).encode())
    write_once(destination / "manifest" / "review_source_key.jsonl",
               ("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in key)).encode())
    write_once(destination / "manifest" / "provenance.json", json_bytes({
        "cohort": "paired A8-v2 FSC VAL development", "n_items": 100,
        "input_lock_sha256": sha(B0 / "input_lock.json"),
        "freeze_sha256": sha(B0 / "freeze.json"),
        "a8_predictions_sha256": sha(A8 / "predictions.jsonl"),
        "b0_predictions_sha256": sha(B0 / "predictions_b0.jsonl"),
        "gt_status": "present primary FSC category; review manifest contains no GT",
        "v1_status": "INVALID_EXECUTION; excluded",
    }))
    print(json.dumps({"prepared": len(items), "destination": str(destination)}))


def evaluate(destination):
    manifest, old, new = validate()
    review = destination / "manifest" / "review_gt_blind.jsonl"
    if not review.exists():
        raise RuntimeError("run prepare and lock review manifest before GT evaluation")
    annotations = json.loads(DATA.read_text())["annotations"]
    points = {}
    for r in annotations:
        x, y, w, h = r["bbox"]
        points.setdefault(r["image_id"], []).append((x + w / 2, y + h / 2))
    records = []
    totals = {k: 0 for k in ("gt", "M0", "A4", "B0", "union", "A4_only_points", "B0_only_points", "both_new_points")}
    for m, a, b in zip(manifest, old, new):
        gt = points[m["source_image_id"]]
        m0 = [box_of(c) for c in a["m0"]]
        aa = [box_of(c) for c in a["a4_add"]]
        bb = [box_of(c) for c in b["additions"]]
        bags = {"M0": m0, "A4": m0 + aa, "B0": m0 + bb, "union": m0 + aa + bb}
        matched = {name: match_count(boxes, gt) for name, boxes in bags.items()}
        for name, value in matched.items():
            totals[name] += value
        totals["gt"] += len(gt)
        cover = {name: {i for i, p in enumerate(gt) if any(contains(box, p) for box in boxes)}
                 for name, boxes in bags.items()}
        only_a = cover["A4"] - cover["B0"]
        only_b = cover["B0"] - cover["A4"]
        both_new = (cover["A4"] & cover["B0"]) - cover["M0"]
        totals["A4_only_points"] += len(only_a)
        totals["B0_only_points"] += len(only_b)
        totals["both_new_points"] += len(both_new)
        records.append({"item_id": m["item_id"], "source_image_id": m["source_image_id"],
                        "query": m["canonical"], "gt_points": len(gt), "m0_zero": not m0,
                        **{f"{name}_boxes": len(boxes) for name, boxes in bags.items()},
                        **{f"{name}_point_tp": matched[name] for name in bags},
                        "A4_only_point_coverage": len(only_a),
                        "B0_only_point_coverage": len(only_b),
                        "both_new_point_coverage": len(both_new)})
    header = list(records[0])
    import io
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=header)
    writer.writeheader(); writer.writerows(records)
    write_once(destination / "audit" / "point_proxy_per_item.csv", stream.getvalue().encode())
    write_once(destination / "audit" / "point_proxy_summary.json", json_bytes({
        "metric": "FSC GT point in box, maximum one-to-one matching; NOT semantic/extent/identity",
        "items": len(records), "totals": totals,
        "notes": "Only-point coverage uses set containment and is not an assignable TP increment. Union TP uses bipartite matching.",
    }))
    print(json.dumps(totals, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": evaluate}[args.action](args.out)
