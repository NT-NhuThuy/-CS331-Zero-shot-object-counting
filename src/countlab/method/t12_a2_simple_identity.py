"""Experimental v2 A2 controls on A1-retained baseline proposals."""

from collections import defaultdict
import math


def _box(candidate):
    raw = candidate.get("box_xyxy")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(x) for x in raw)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _iou(a, b):
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    intersection = max(0, x1-x0) * max(0, y1-y0)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return intersection / (area_a + area_b - intersection)


def _score(candidate):
    value = candidate.get("confidence", candidate.get("score"))
    try:
        number = float(value)
    except (ValueError, TypeError):
        return float("-inf")
    return number if math.isfinite(number) else float("-inf")


def run_a2_simple_identity(
    manifest, observed_events, *, a1_rejected_ids, nms_iou=0.5
):
    if not isinstance(nms_iou, (int, float)):
        raise ValueError("invalid NMS threshold")
    if not math.isfinite(nms_iou) or not 0 < nms_iou <= 1:
        raise ValueError("invalid NMS threshold")

    baseline = manifest.get("baseline")
    if not isinstance(baseline, list):
        raise ValueError("baseline must be a list")
    if not isinstance(observed_events, list):
        raise ValueError("events must be a list")

    by_id = {}
    for item in baseline:
        if not isinstance(item, dict):
            raise ValueError("malformed candidate")
        cid = item.get("candidate_id")
        if not isinstance(cid, str) or not cid or cid in by_id:
            raise ValueError("invalid or duplicate candidate ID")
        by_id[cid] = item

    if manifest.get("baseline_count") != len(baseline):
        raise ValueError("frozen baseline count mismatch")

    rejected = set(a1_rejected_ids)
    if not rejected.issubset(by_id):
        raise ValueError("A1 rejected ID not in baseline")

    retained = {
        cid: item for cid, item in by_id.items()
        if cid not in rejected
    }

    ordered = sorted(
        retained,
        key=lambda cid: (-_score(retained[cid]), cid)
    )

    kept = []
    suppressed = []
    invalid_geometry = []

    for cid in ordered:
        current = _box(retained[cid])

        if current is None:
            invalid_geometry.append(cid)

        overlap = False
        if current is not None:
            for other in kept:
                previous = _box(retained[other])
                if previous is not None and _iou(current, previous) >= nms_iou:
                    overlap = True
                    break

        if overlap:
            suppressed.append(cid)
        else:
            kept.append(cid)

    relations = defaultdict(set)
    edge_refs = defaultdict(set)

    for event in observed_events:
        if not isinstance(event, dict):
            raise ValueError("malformed observed event")
        if event.get("op") != "PAIR_IDENTITY":
            continue

        a = event.get("a")
        b = event.get("b")
        relation = event.get("relation")
        ref = event.get("evidence_ref")

        if (
            not isinstance(a, str)
            or not isinstance(b, str)
            or a == b
            or relation not in {"SAME", "DIFFERENT", "UNKNOWN"}
            or not isinstance(ref, str)
            or not ref
        ):
            raise ValueError("malformed observed pair event")

        if a not in retained or b not in retained:
            continue

        key = tuple(sorted((a, b)))
        if relation != "UNKNOWN":
            relations[key].add(relation)
            edge_refs[key].add(ref)

    groups = []

    for cid in sorted(retained):
        placed = False

        for group in groups:
            if all(
                relations.get(tuple(sorted((cid, other)))) == {"SAME"}
                for other in group
            ):
                group.append(cid)
                placed = True
                break

        if not placed:
            groups.append([cid])

    same_refs = sorted({
        ref
        for group in groups
        for i, a in enumerate(group)
        for b in group[i+1:]
        for ref in edge_refs[tuple(sorted((a, b)))]
    })

    pair_conflicts = sorted([
        list(key)
        for key, values in relations.items()
        if len(values) > 1
    ])

    return {
        "schema": "countlab_v2_a2_preliminary_v1",
        "status": "EXPERIMENTAL_NOT_APPROVED",
        "scope": "A1_RETAINED_BASELINE_ONLY",
        "a1_retained_baseline_count": len(retained),
        "a1_rejected_candidate_ids": sorted(rejected),
        "A2_NMS": {
            "count": len(kept),
            "kept_candidate_ids": sorted(kept),
            "suppressed_candidate_ids": sorted(suppressed),
            "invalid_geometry_retained": sorted(invalid_geometry),
            "iou_threshold": nms_iou,
            "identity_evidence_used": False
        },
        "A2_COMPLETE_LINK": {
            "count": len(groups),
            "groups": groups,
            "direct_same_evidence_refs": same_refs,
            "pair_conflicts": pair_conflicts,
            "geometry_used": False
        },
        "rescue_additions": 0,
        "gt_used": False,
        "model_calls_by_a2": 0,
        "scene_count_certified": False
    }
