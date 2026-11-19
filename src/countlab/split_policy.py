"""P03 split policy: metadata checks, not G0 approval."""

from collections import Counter, defaultdict
import re

CLEAN = {"TRAIN", "DEV", "CAL", "FINAL_SEALED"}
ALLOWED = CLEAN | {"EXPOSED_REFERENCE", "UNASSIGNED"}
REQUIRED = {
    "dataset", "source_image_id", "source_group_id",
    "source_image_sha256", "query_id", "proposed_split",
    "exposure_status", "lineage_status",
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def audit_split_proposal(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a nonempty list")

    issues = []
    by_group = defaultdict(set)
    by_source = defaultdict(set)
    by_hash = defaultdict(set)
    hash_groups = defaultdict(set)
    group_exposures = defaultdict(set)
    queries = set()
    counts = Counter()

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            issues.append(f"row {i}: not an object")
            continue

        missing = REQUIRED - row.keys()
        if missing:
            issues.append(f"row {i}: missing {sorted(missing)}")
            continue

        split = row["proposed_split"]
        exposure = row["exposure_status"]
        lineage = row["lineage_status"]
        digest = row["source_image_sha256"]
        gid = row["source_group_id"]
        source = (row["dataset"], row["source_image_id"])
        query = (row["dataset"], row["query_id"])

        if split not in ALLOWED:
            issues.append(f"row {i}: invalid split")
        if exposure not in {"UNEXPOSED", "EXPOSED", "UNKNOWN"}:
            issues.append(f"row {i}: invalid exposure")
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            issues.append(f"row {i}: invalid image SHA256")
        if not isinstance(gid, str) or not gid.strip():
            issues.append(f"row {i}: missing source group")

        if any(
            not isinstance(x, str) or not x.strip()
            for x in source + query
        ):
            issues.append(f"row {i}: invalid dataset/source/query ID")

        if query in queries:
            issues.append(f"row {i}: duplicate query ID")
        queries.add(query)

        if split in CLEAN:
            if exposure != "UNEXPOSED" or lineage != "VERIFIED":
                issues.append(
                    f"row {i}: clean split requires verified unexposed source"
                )

            evidence = row.get("provenance_evidence_sha256")
            if not isinstance(evidence, str) or not HEX64.fullmatch(evidence):
                issues.append(
                    f"row {i}: missing provenance evidence SHA256"
                )

        elif split == "EXPOSED_REFERENCE" and exposure != "EXPOSED":
            issues.append(
                f"row {i}: historical reference must be exposed"
            )

        counts[str(split)] += 1
        by_group[str(gid)].add(str(split))
        by_source[source].add(str(split))

        if isinstance(digest, str) and HEX64.fullmatch(digest):
            by_hash[digest].add(str(split))
            hash_groups[digest].add(str(gid))

        group_exposures[str(gid)].add(str(exposure))

    for kind, entries in (
        ("source group", by_group),
        ("source image", by_source),
        ("image hash", by_hash),
    ):
        for key, splits in entries.items():
            if len(splits) > 1 and CLEAN.intersection(splits):
                issues.append(
                    f"{kind} spans clean split(s): {key}: {sorted(splits)}"
                )

    for digest, gids in hash_groups.items():
        if len(gids) > 1:
            issues.append(
                f"byte-identical image has different group IDs: {digest[:12]}"
            )

    for gid, exposure_states in group_exposures.items():
        if "UNEXPOSED" in exposure_states and len(exposure_states) > 1:
            issues.append(f"source-group exposure conflict: {gid}")

    return {
        "status": "PASS" if not issues else "FAIL",
        "n_rows": len(rows),
        "split_counts": dict(sorted(counts.items())),
        "issues": issues,
        "g0_decision": "HOLD",
        "note": (
            "Metadata validation only; provenance and "
            "independence are not certified."
        ),
    }
