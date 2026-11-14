"""Post-freeze GT-point diagnostics for A4/B0 additions, separate from review."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "target_aware_counting/output"
DEST = OUT / "fqie_gate0_20261104_v1/audit"


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def covers(box, point):
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def write_once(path, content):
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"locked output differs: {path}")
    path.write_bytes(content)


def main():
    a = rows(OUT / "a8_weak_singleton_val_20261102_v1/predictions.jsonl")
    b = rows(OUT / "b0_a8_paired_val_sm86_20261103_v2/predictions_b0.jsonl")
    m = rows(OUT / "b0_a8_paired_val_sm86_20261103_v2/input_queries.jsonl")
    data = json.loads((ROOT / "data/fscd147/countgd_val_fscd147.json").read_text())
    gt = {}
    for x in data["annotations"]:
        px, py, w, h = x["bbox"]
        gt.setdefault(x["image_id"], []).append((x["id"], (px + w/2, py + h/2)))
    ledger = []
    for item, old, new in zip(m, a, b):
        assert item["item_id"] == old["item_id"] == new["item_id"]
        points = gt[item["source_image_id"]]
        base = old["m0"]
        branches = {"A4": old["a4_add"], "B0": new["additions"]}
        covered = {}
        for name, entries in {"M0": base, **branches}.items():
            covered[name] = {pid for pid, p in points
                             if any(covers(x["box_xyxy"], p) for x in entries)}
        for name, entries in branches.items():
            other = "B0" if name == "A4" else "A4"
            for c in entries:
                ids = {pid for pid, p in points if covers(c["box_xyxy"], p)}
                ledger.append({"item_id": item["item_id"], "query": item["canonical"],
                               "source": name, "candidate_id": c["candidate_id"],
                               "point_containment": len(ids),
                               "new_vs_m0_points": len(ids - covered["M0"]),
                               "only_this_source_points": len(ids - covered["M0"] - covered[other]),
                               "multi_point_box": int(len(ids) > 1)})
    import io
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(ledger[0]))
    writer.writeheader(); writer.writerows(ledger)
    write_once(DEST / "candidate_point_ledger.csv", stream.getvalue().encode())
    summary = {name: {"candidates": len(v),
                      "boxes_with_no_point": sum(x["point_containment"] == 0 for x in v),
                      "boxes_with_multiple_points": sum(x["multi_point_box"] for x in v),
                      "boxes_with_point_new_vs_m0": sum(x["new_vs_m0_points"] > 0 for x in v),
                      "boxes_with_point_only_this_source": sum(x["only_this_source_points"] > 0 for x in v)}
               for name in ("A4", "B0") for v in [[x for x in ledger if x["source"] == name]]}
    summary["warning"] = "Containment is geometry only; candidate rows share GT points and cannot be summed as distinct TP."
    write_once(DEST / "candidate_point_summary.json", (json.dumps(summary, indent=2) + "\n").encode())
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
