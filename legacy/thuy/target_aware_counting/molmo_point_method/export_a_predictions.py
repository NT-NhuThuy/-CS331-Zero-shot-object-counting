"""Export a complete frozen-label-free A cohort run into method JSONL files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .native_greedy import COHORTS, load_manifest, sha256

METHODS = ("native", "proper_map", "proper_mbr", "guided_map", "guided_mbr")


def prediction_for_method(row: dict, method: str) -> dict:
    if method == "native":
        native = row["native"]
        if native["status"] != "complete":
            raise ValueError("native generation incomplete")
        points = native["points_xy"]
        trace = {"source": "native", "output_tokens": native["output_tokens"]}
    else:
        branch_name, decision_name = method.rsplit("_", 1)
        branch = row["proper_beam"] if branch_name == "proper" else row["guided_beam"]
        fallback = branch is None
        if fallback:
            if row["f1_raw_available"]:
                raise ValueError("guided result missing despite complete F1 pool")
            branch = row["proper_beam"]
        if branch["decision_status"] != "ready":
            raise ValueError("search has no valid complete decision")
        decision = branch["decision"]
        points = decision[f"{decision_name}_points_xy"]
        trace = {"source": "proper_fallback_no_f1" if fallback else branch_name,
                 "decision": decision_name,
                 "trace_id": decision[f"{decision_name}_trace_id"],
                 "bank_size_unique": decision["bank_size_unique"],
                 "complete_sequences": branch["stats"]["complete_sequences"],
                 "invalid_complete_sequences": branch["stats"]["invalid_complete_sequences"],
                 "active_truncated": branch["stats"]["active_truncated"]}
    points = [list(point) for point in points]
    return {"status": "complete", "count": len(points), "points_xy": points,
            "trace": trace}


def export(cohort: str, source: Path, output_dir: Path) -> dict:
    manifest, items = load_manifest(cohort)
    expected = {item["item_id"]: item for item in items}
    with source.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != len(items) or len({row["item_id"] for row in rows}) != len(items):
        raise ValueError("cohort run has missing or duplicate rows")
    contract = rows[0]["contract"]
    if contract.get("method") != "molmopoint_a_cached_beam_v1" or \
            contract.get("cohort") != cohort or contract.get("manifest_sha256") != sha256(manifest):
        raise ValueError("cohort run contract mismatch")
    for row in rows:
        item = expected.get(row["item_id"])
        if item is None or row["contract"] != contract or \
                row["query"] != item["query"] or \
                row["source_image_id"] != item["source_image_id"]:
            raise ValueError("cohort row identity or contract mismatch")
    outputs = {method: output_dir / f"{cohort}_{method}.jsonl" for method in METHODS}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("one or more method outputs already exist")
    source_hash = sha256(source)
    materialized = {}
    for method in METHODS:
        exported_contract = {**contract, "method": f"molmopoint_a_{method}",
                             "source_predictions": str(source.resolve()),
                             "source_predictions_sha256": source_hash}
        materialized[method] = [
            {"item_id": row["item_id"],
             "source_image_id": row["source_image_id"], "query": row["query"],
             "contract": exported_contract, **prediction_for_method(row, method)}
            for row in rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    for method, path in outputs.items():
        with path.open("x", encoding="utf-8") as stream:
            for row in materialized[method]:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return {"cohort": cohort, "items": len(items),
            "source_sha256": source_hash,
            "outputs": {method: str(path) for method, path in outputs.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.cohort, args.source, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
