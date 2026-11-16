"""P03: Source-disjoint split and exposure audit. CPU only."""

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import re

REQUIRED = {
    "dataset", "source_image_id", "query_id", "split",
    "gt_available", "exposure_status",
    "artifact_sha256", "protocol_id",
}

SPLITS = {
    "TRAIN", "DEV", "CAL", "FINAL_SEALED",
    "EXPOSED_REFERENCE",
}
EXPOSURES = {"UNEXPOSED", "EXPOSED", "UNKNOWN"}
FORBIDDEN_GT = {
    "gt", "gt_count", "ground_truth",
    "count_target", "true_count", "label_count",
}


def audit_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must contain nonempty rows")

    issues = []
    by_source = defaultdict(set)
    by_image_hash = defaultdict(set)
    seen_queries = set()
    counts = Counter()

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {i}: expected object")

        missing = REQUIRED - set(row)
        if missing:
            raise ValueError(f"row {i}: missing {sorted(missing)}")

        if FORBIDDEN_GT & set(row):
            raise ValueError(f"row {i}: GT values forbidden")

        for key in (
            "dataset", "source_image_id",
            "query_id", "protocol_id",
        ):
            if not isinstance(row[key], str) or not row[key].strip():
                issues.append(f"row {i}: invalid {key}")

        split = row["split"]
        exposure = row["exposure_status"]

        if split not in SPLITS:
            issues.append(f"row {i}: invalid split {split}")
        if exposure not in EXPOSURES:
            issues.append(f"row {i}: invalid exposure {exposure}")
        if type(row["gt_available"]) is not bool:
            issues.append(f"row {i}: gt_available must be boolean")

        if not re.fullmatch(
            r"[0-9a-f]{64}", str(row["artifact_sha256"])
        ):
            issues.append(f"row {i}: invalid artifact SHA256")

        pair_key = (row["dataset"], row["query_id"])
        if pair_key in seen_queries:
            issues.append(f"row {i}: duplicate query key {pair_key}")
        seen_queries.add(pair_key)

        by_source[row["source_image_id"]].add(split)
        counts[split] += 1

        image_hash = row.get("source_image_sha256")
        if image_hash is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", str(image_hash)):
                issues.append(f"row {i}: invalid image SHA256")
            else:
                by_image_hash[image_hash].add(split)

        if split == "EXPOSED_REFERENCE":
            if exposure != "EXPOSED":
                issues.append(f"row {i}: reference must be EXPOSED")
        elif exposure != "UNEXPOSED":
            issues.append(f"row {i}: unsafe exposure in {split}")

        known_exposed = (
            "fsc147_test", "fsc147_official_test",
            "absent111", "hard_negative_111",
            "v13", "v14",
        )
        identity = (
            str(row["dataset"]) + " " + str(row["protocol_id"])
        ).lower()

        if any(tag in identity for tag in known_exposed):
            if exposure != "EXPOSED":
                issues.append(
                    f"row {i}: known exposed cohort misclassified"
                )

    for source, splits in by_source.items():
        if len(splits) > 1:
            issues.append(
                f"source-image leakage: {source} -> {sorted(splits)}"
            )

    for digest, splits in by_image_hash.items():
        if len(splits) > 1:
            issues.append(
                f"image-hash leakage: {digest[:12]} -> {sorted(splits)}"
            )

    return {
        "status": "PASS" if not issues else "FAIL",
        "n_rows": len(rows),
        "n_source_images": len(by_source),
        "split_counts": dict(sorted(counts.items())),
        "issues": issues,
        "note": (
            "PASS verifies declared metadata only; independent "
            "source identity and sealed status need provenance evidence."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(
        args.manifest.read_text(encoding="utf-8")
    )
    if payload.get("schema") != "countlab_split_manifest_v1":
        raise ValueError("unsupported manifest schema")

    report = audit_rows(payload["rows"])
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
