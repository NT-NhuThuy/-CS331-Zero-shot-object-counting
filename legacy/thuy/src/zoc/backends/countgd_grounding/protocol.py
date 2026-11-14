from __future__ import annotations

import random
from .io import read_jsonl, write_jsonl, norm_query, digest


def make_manifest(ds, split, path, negatives=4, all_queries=False, limit=None, overrides=None):
    """Other classes are UNKNOWN unless a supplied reviewed annotation confirms them."""
    if negatives < 0:
        raise ValueError("negatives must be nonnegative")
    names = ds.splits[split][:limit] if limit else ds.splits[split]
    vocab = sorted(ds.class_sets[split])
    annotations = {}
    if overrides:
        for r in read_jsonl(overrides):
            name, query = r["image"], norm_query(r["query"])
            if ds.image_split.get(name) != split or query not in ds.class_sets[split]:
                raise ValueError(f"Override must use images/classes of {split}: {name}/{query}")
            key = name, query
            if key in annotations:
                raise ValueError(f"Duplicate override: {key}")
            if r["status"] not in ("present", "absent") or not str(r.get("evidence", "")).strip():
                raise ValueError("Overrides require present/absent status and evidence/reviewer")
            annotations[key] = r
    rng = random.Random(ds.cfg["seed"])
    rows = []
    for name in names:
        w, h = ds.validate_image(name)
        target = ds.classes[name]
        others = [q for q in vocab if q != target]
        queries = [target] + (others if all_queries else rng.sample(others, min(negatives, len(others))))
        # Ensure reviewed supplemental queries are actually included.
        queries += sorted(q for n, q in annotations if n == name and q not in queries)
        for query in queries:
            status = "present" if query == target else "unknown"
            points = ds.points(name).tolist() if status == "present" else None
            evidence = "FSC147 primary class annotation" if status == "present" else None
            if (name, query) in annotations:
                r = annotations[name, query]
                status, evidence = r["status"], r["evidence"]
                points = r.get("points") if status == "present" else []
                if points is None or (status == "present" and not points):
                    raise ValueError(f"Present override requires complete point annotations: {name}/{query}")
                if query == target and (status != "present" or points != ds.points(name).tolist()):
                    raise ValueError("Do not overwrite primary FSC GT via overrides; version dataset explicitly")
            row = {"schema": 1, "dataset_fingerprint": ds.fingerprint,
                   "image": name, "split": split, "query": query, "status": status,
                   "width": w, "height": h, "gt_points": points,
                   "gt_count": len(points) if points is not None else None,
                   "evidence": evidence}
            row["id"] = digest([ds.fingerprint, split, name, query])[:24]
            validate_row(ds, row)
            rows.append(row)
    write_jsonl(path, rows)
    return {"rows": len(rows), "present": sum(r["status"] == "present" for r in rows),
            "absent": sum(r["status"] == "absent" for r in rows),
            "unknown": sum(r["status"] == "unknown" for r in rows)}


def validate_row(ds, r):
    import numpy as np
    if r["dataset_fingerprint"] != ds.fingerprint:
        raise ValueError("Manifest/dataset fingerprint mismatch")
    if ds.image_split.get(r["image"]) != r["split"] or r["query"] not in ds.class_sets[r["split"]]:
        raise ValueError("Manifest image/query violates split")
    if norm_query(r["query"]) != r["query"] or r["status"] not in ("present", "absent", "unknown"):
        raise ValueError("Invalid query/status")
    if ds.size(r["image"]) != (r["width"], r["height"]):
        raise ValueError("Manifest image size differs from actual file")
    if r["status"] == "unknown":
        if r["gt_count"] is not None or r["gt_points"] is not None:
            raise ValueError("Unknown query cannot have fabricated GT")
    else:
        pts = np.asarray(r["gt_points"], dtype=float).reshape(-1, 2)
        if not str(r.get("evidence", "")).strip() or not np.isfinite(pts).all():
            raise ValueError("Verified queries require evidence and finite points")
        eps = 1e-6
        if len(pts) != r["gt_count"] or ((pts < -eps).any() or (pts[:, 0] > r["width"] + eps).any()
                                          or (pts[:, 1] > r["height"] + eps).any()):
            raise ValueError("Invalid GT point/count/coordinate")
        if (r["status"] == "absent" and len(pts)) or (r["status"] == "present" and not len(pts)):
            raise ValueError("Status inconsistent with GT")


def make_region_review(ds, split, path, negatives=4, limit=None):
    if negatives < 0:
        raise ValueError("negatives must be nonnegative")
    rng = random.Random(ds.cfg["seed"])
    rows = []
    names = ds.splits[split][:limit] if limit else ds.splits[split]
    # Suggested negative classes also have positive exemplar ROIs in this subset.
    # This reduces a trivial lexical "this word is always negative" shortcut.
    regional_vocab = {ds.classes[n] for n in names if len(ds.boxes(n))}
    for name in names:
        ds.validate_image(name)
        target = ds.classes[name]
        candidates = sorted(regional_vocab - {target})
        for index, box in enumerate(ds.boxes(name).tolist()):
            negatives_here = rng.sample(candidates, min(negatives, len(candidates)))
            rows.append({"schema": 1, "dataset_fingerprint": ds.fingerprint,
                         "id": digest([ds.fingerprint, split, name, index, box])[:24],
                         "image": name, "split": split, "box_xyxy": box,
                         "source_class": target, "source": "FSC147 exemplar",
                         "labels": [{"query": target, "label": 1, "evidence": "FSC147 exemplar"}]
                         + [{"query": q, "label": None, "evidence": ""} for q in negatives_here]})
    write_jsonl(path, rows)
    return {"regions": len(rows), "negative_labels": "pending review; null is ignored"}


def checked_region_labels(ds, row, split):
    import numpy as np
    if row["dataset_fingerprint"] != ds.fingerprint or row["split"] != split:
        raise ValueError("Region review fingerprint/split mismatch")
    if ds.image_split.get(row["image"]) != split or row["source_class"] not in ds.class_sets[split]:
        raise ValueError("Region data violates class/image split")
    w, h = ds.validate_image(row["image"])
    box = np.asarray(row["box_xyxy"], dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or (box < 0).any() or box[2] > w or box[3] > h or (box[2:] <= box[:2]).any():
        raise ValueError("Invalid reviewed box")
    labels, seen = [], set()
    for item in row["labels"]:
        query = norm_query(item["query"])
        if query not in ds.class_sets[split] or query in seen:
            raise ValueError(f"Query repeated or outside {split} classes: {query}")
        seen.add(query)
        label = item["label"]
        if label is None:
            continue
        if type(label) is not int or label not in (0, 1) or not str(item.get("evidence", "")).strip():
            raise ValueError("Reviewed label must be integer 0/1 with evidence")
        labels.append({"query": query, "label": label, "evidence": item["evidence"]})
    return labels
