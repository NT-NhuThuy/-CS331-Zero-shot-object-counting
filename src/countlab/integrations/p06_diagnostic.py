"""P06: GT-free candidate diagnostics. No training or calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from countlab.integrations.thu_features import attach_features
from countlab.method.schema import _validate_no_gt, parse_input


def canonical_sha256(obj):
    raw = json.dumps(
        obj, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def extract_diagnostic_features(
    manifest,
    npz_path,
    candidate_map,
    *,
    exposure_status="UNKNOWN",
):
    if exposure_status not in {"EXPOSED", "UNKNOWN"}:
        raise ValueError("P06 diagnostic exposure must be EXPOSED or UNKNOWN")

    _validate_no_gt(manifest)
    _validate_no_gt(candidate_map)

    baseline, rescue, _ = parse_input(manifest)

    if not isinstance(candidate_map, dict):
        raise ValueError("candidate_map must be a dict")

    ids = {p.candidate_id for p in baseline + rescue}

    if set(candidate_map) != ids:
        raise ValueError("candidate_map must cover exactly all candidate IDs")

    mapped_keys = set()

    for cid, mapping in candidate_map.items():
        if not isinstance(mapping, dict) or set(mapping) != {
            "node_graph_id", "raw_query_index"
        }:
            raise ValueError(f"{cid}: invalid exact-key mapping")

        key = (
            mapping["node_graph_id"],
            mapping["raw_query_index"],
        )

        if any(type(v) is not int or v < 0 for v in key):
            raise ValueError(f"{cid}: mapping indices must be nonnegative integers")

        if key in mapped_keys:
            raise ValueError("two candidate IDs map to the same frozen node key")

        mapped_keys.add(key)

    joined = attach_features(
        manifest, Path(npz_path), candidate_map
    )

    source_meta = joined["metadata"]["thu_frozen_features"]
    diagnostics = source_meta["per_candidate"]
    rows = []

    for proposal in sorted(
        baseline + rescue, key=lambda p: p.candidate_id
    ):
        x0, y0, x1, y1 = proposal.box
        w, h = x1 - x0, y1 - y0
        values = diagnostics[proposal.candidate_id]
        confidence = proposal.confidence

        rows.append({
            "candidate_id": proposal.candidate_id,
            "role": proposal.role,
            "view_id": proposal.view_id,
            "box_xyxy": [x0, y0, x1, y1],
            "area_px2": w * h,
            "aspect_ratio": w / h,
            "detector_confidence": confidence,
            "confidence_uncertainty_baseline": (
                None if confidence is None else 1.0 - confidence
            ),
            "v4_contribution": values["v4_contribution"],
            "a3_score": values["a3_score"],
            "node_graph_id": values["node_graph_id"],
            "raw_query_index": values["raw_query_index"],
        })

    result = {
        "schema": "countlab_p06_diagnostic_features_v1",
        "image_id": str(manifest["image_id"]),
        "query_id": str(manifest["query_id"]),
        "n_candidates": len(rows),
        "candidate_pool_sha256": canonical_sha256(manifest),
        "feature_npz_sha256": source_meta["feature_sha256"],
        "candidate_map_sha256": source_meta["candidate_map_sha256"],
        "source_exposure_status": exposure_status,
        "risk_hint_generated": False,
        "calibration_used": False,
        "eligible_for_training": False,
        "eligible_for_final_sealed": False,
        "g0_decision": "HOLD",
        "candidates": rows,
    }

    _validate_no_gt(result)
    return result
