"""Require cached and reference one-item beams to have identical scored paths."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def compare(reference: Path, cached: Path, tolerance: float = 1e-4) -> dict:
    slow = json.loads(reference.read_text())
    fast = json.loads(cached.read_text())
    if slow["item_id"] != fast["item_id"] or slow["cohort"] != fast["cohort"] or \
            slow["snapshot_revision"] != fast["snapshot_revision"] or \
            slow["search_config"].get("search_engine", "reference") != "reference" or \
            fast["search_config"].get("search_engine") != "cached":
        raise ValueError("reference/cached identity or engine mismatch")
    for key, value in slow["search_config"].items():
        if key != "search_engine" and fast["search_config"].get(key) != value:
            raise ValueError(f"search configuration mismatch: {key}")
    result = {"item_id": slow["item_id"], "cohort": slow["cohort"],
              "snapshot_revision": slow["snapshot_revision"], "branches": {}}
    for branch in ("proper_beam", "guided_beam"):
        a, b = slow[branch], fast[branch]
        if (a is None) != (b is None):
            raise ValueError(f"{branch}: availability mismatch")
        if a is None:
            continue
        if len(a["paths"]) != len(b["paths"]):
            raise ValueError(f"{branch}: path count mismatch")
        deltas = []
        for index, (p, q) in enumerate(zip(a["paths"], b["paths"])):
            if p["tokens"] != q["tokens"] or p["valid_for_decision"] != q["valid_for_decision"] or \
                    p["count"] != q["count"] or p["points_xy"] != q["points_xy"]:
                raise ValueError(f"{branch}: decoded path differs at index {index}")
            delta = abs(p["log_probability"] - q["log_probability"])
            if not math.isfinite(delta) or delta > tolerance:
                raise ValueError(f"{branch}: log probability differs at index {index}: {delta}")
            deltas.append(delta)
        if a["decision_status"] != b["decision_status"] or \
                (a["decision"] or {}).get("map_points_xy") != (b["decision"] or {}).get("map_points_xy") or \
                (a["decision"] or {}).get("mbr_points_xy") != (b["decision"] or {}).get("mbr_points_xy"):
            raise ValueError(f"{branch}: MAP/MBR decision mismatch")
        for key in ("complete_sequences", "active_truncated", "valid_complete_sequences",
                    "invalid_complete_sequences", "expanded_options"):
            if a["stats"][key] != b["stats"][key]:
                raise ValueError(f"{branch}: search statistic mismatch: {key}")
        result["branches"][branch] = {
            "paths": len(deltas), "max_absolute_log_probability_delta": max(deltas, default=0.0),
            "reference_forward_calls": a["stats"]["replay_forward_calls"],
            "cached_forward_calls": b["stats"]["cached_forward_calls"],
            "reference_seconds": a["stats"]["elapsed_seconds"],
            "cached_seconds": b["stats"]["elapsed_seconds"]}
    result["cached_reference_equal"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--cached", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.cached)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
