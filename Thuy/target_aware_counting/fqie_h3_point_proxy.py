"""H3 geometry diagnostic with conservative FSC GT-point pair labels."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "target_aware_counting"))
from fqie_conflict_review import metrics, rows, write_once

OUT = ROOT / "target_aware_counting/output"
DEST = OUT / "fqie_gate0_20261104_v1/audit/h3_point_proxy"


def contains(box, point):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def main():
    manifest = rows(OUT / "b0_a8_paired_val_sm86_20261103_v2/input_queries.jsonl")
    a8 = rows(OUT / "a8_weak_singleton_val_20261102_v1/predictions.jsonl")
    b0 = rows(OUT / "b0_a8_paired_val_sm86_20261103_v2/predictions_b0.jsonl")
    gt_data = json.loads((ROOT / "data/fscd147/countgd_val_fscd147.json").read_text())
    gt = defaultdict(list)
    for ann in gt_data["annotations"]:
        x, y, w, h = ann["bbox"]
        gt[ann["image_id"]].append((ann["id"], (x+w/2, y+h/2)))
    pairs, all_pairs = [], 0
    for m, a, b in zip(manifest, a8, b0):
        if not (m["item_id"] == a["item_id"] == b["item_id"]):
            raise RuntimeError("paired artifact binding mismatch")
        candidates = []
        for source, values in (("M0", a["m0"]), ("A4", a["a4_add"]), ("B0", b["additions"])):
            for c in values:
                ids = tuple(sorted(pid for pid, p in gt[m["source_image_id"]] if contains(c["box_xyxy"], p)))
                candidates.append({"source": source, "id": c["candidate_id"], "box": c["box_xyxy"], "points": ids})
        for i, left in enumerate(candidates):
            for right in candidates[i+1:]:
                all_pairs += 1
                iou, iom, distance = metrics(left["box"], right["box"])
                if not (iou > 0 or distance < 1.5):
                    continue
                if len(left["points"]) == len(right["points"]) == 1:
                    label = "SAME_COUNT_INSTANCE_PROXY" if left["points"] == right["points"] else "DISTINCT_INSTANCE_PROXY"
                else:
                    label = "UNKNOWN"
                pairs.append({"item_id": m["item_id"], "source_pair": "-".join(sorted((left["source"], right["source"]))),
                              "left_id": left["id"], "right_id": right["id"],
                              "left_point_count": len(left["points"]), "right_point_count": len(right["points"]),
                              "label": label, "iou": iou, "iom": iom, "center_distance_norm": distance})
    labeled = [x for x in pairs if x["label"] != "UNKNOWN"]
    same = [x for x in labeled if x["label"] == "SAME_COUNT_INSTANCE_PROXY"]
    distinct = [x for x in labeled if x["label"] == "DISTINCT_INSTANCE_PROXY"]
    curves = {}
    for field, thresholds, direction in (
        ("iou", (0.1, 0.3, 0.5, 0.7, 0.9), "ge"),
        ("iom", (0.5, 0.7, 0.8, 0.9, 0.95), "ge"),
        ("center_distance_norm", (0.25, 0.5, 0.75, 1.0), "le")):
        values = []
        for t in thresholds:
            pred = (lambda x: x[field] >= t) if direction == "ge" else (lambda x: x[field] <= t)
            values.append({"threshold": t,
                           "same_proxy_captured": sum(pred(x) for x in same), "same_proxy_total": len(same),
                           "same_proxy_recall": sum(pred(x) for x in same)/len(same) if same else None,
                           "distinct_proxy_merged": sum(pred(x) for x in distinct), "distinct_proxy_total": len(distinct),
                           "distinct_proxy_false_merge_rate": sum(pred(x) for x in distinct)/len(distinct) if distinct else None})
        curves[field] = values
    summary = {"status": "GT_POINT_IDENTITY_PROXY_DIAGNOSTIC", "items": len(manifest),
               "all_union_pairs": all_pairs, "screened_conflict_pairs": len(pairs),
               "labels": dict(Counter(x["label"] for x in pairs)),
               "source_pairs_labeled": dict(Counter(x["source_pair"] for x in labeled)),
               "curves": curves,
               "limitations": [
                   "Same singleton GT point means same count-instance proxy; it cannot distinguish duplicate from part.",
                   "Different singleton GT points are a distinct-instance proxy under FSC point annotations.",
                   "Pairs with zero or multiple contained points are UNKNOWN and excluded from rates.",
                   "Geometry thresholds are diagnostics only; none is selected or promoted as a merge rule."]}
    write_once(DEST / "summary.json", (json.dumps(summary, indent=2) + "\n").encode())
    write_once(DEST / "pairs.jsonl", ("".join(json.dumps(x, separators=(",", ":"))+"\n" for x in pairs)).encode())
    print(json.dumps({k: summary[k] for k in ("items", "all_union_pairs", "screened_conflict_pairs", "labels")}))
    print(json.dumps(curves))


if __name__ == "__main__":
    main()
