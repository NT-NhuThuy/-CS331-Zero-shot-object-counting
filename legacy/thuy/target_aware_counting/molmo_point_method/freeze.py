"""Freeze a complete, identity-checked native prediction file before label access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .native_greedy import COHORTS, load_manifest, sha256


def verify_and_freeze(cohort: str, predictions: Path, output: Path) -> dict:
    manifest, items = load_manifest(cohort)
    expected = {row["item_id"]: row for row in items}
    rows = []
    with predictions.open(encoding="utf-8") as stream:
        for line in stream:
            rows.append(json.loads(line))
    if len(rows) != len(items):
        raise ValueError(f"incomplete predictions: {len(rows)}/{len(items)}")
    seen = set()
    contract = rows[0]["contract"]
    if contract["cohort"] != cohort or contract["manifest_sha256"] != sha256(manifest):
        raise ValueError("prediction contract does not match cohort manifest")
    for row in rows:
        item_id = row["item_id"]
        if item_id not in expected or item_id in seen:
            raise ValueError("unknown or duplicate prediction item")
        seen.add(item_id)
        source = expected[item_id]
        if row["source_image_id"] != source["source_image_id"] or \
                row["query"] != source["query"] or row["contract"] != contract:
            raise ValueError("prediction identity or configuration mismatch")
        if row["status"] != "complete" or row["count"] != len(row["points_xy"]):
            raise ValueError(f"incomplete or inconsistent prediction: {item_id}")
        if any(len(p) != 2 for p in row["points_xy"]):
            raise ValueError(f"malformed points: {item_id}")
    frozen = {
        "complete": True,
        "labels_opened_before_freeze": False,
        "cohort": cohort,
        "items": len(items),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256(manifest),
        "predictions": str(predictions.resolve()),
        "predictions_sha256": sha256(predictions),
        "contract": contract,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(frozen, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_and_freeze(args.cohort, args.predictions, args.output)
    print(json.dumps({"cohort": result["cohort"], "items": result["items"],
                      "predictions_sha256": result["predictions_sha256"]}))


if __name__ == "__main__":
    main()
