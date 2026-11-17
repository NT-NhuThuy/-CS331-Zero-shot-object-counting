"""Build a historical source-group roster. No clean split assignment."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("reports/p03")
MANIFEST = ROOT / "p03_exposed_source_manifest.json"
LEDGER = ROOT / "p03_image_byte_hashes.csv"
IDENTITY = ROOT / "p03_source_identity_report.json"
OUTPUT = ROOT / "p03_split_proposal_roster.csv"
SUMMARY = ROOT / "p03_split_proposal_summary.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError("Proposal already exists; refusing overwrite")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))

    assert manifest["schema"] == "countlab_split_manifest_v1"
    assert sha256(LEDGER) == identity["ledger_sha256"]

    with LEDGER.open(encoding="utf-8-sig", newline="") as f:
        ledger = list(csv.DictReader(f))

    by_id = {r["source_image_id"]: r for r in ledger}
    if len(by_id) != len(ledger):
        raise ValueError("Duplicate source IDs in ledger")

    records = manifest["rows"]

    assert len(records) == identity["n_pairs"] == 25301
    assert len(by_id) == identity["n_unique_declared_source_ids"] == 13190

    fields = [
        "dataset", "source_image_id", "source_group_id",
        "source_image_sha256", "query_id", "original_cohort",
        "proposed_split", "exposure_status", "lineage_status",
        "artifact_sha256", "protocol_id",
    ]

    counts = Counter()
    groups = defaultdict(set)

    with OUTPUT.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in records:
            image = by_id.get(row["source_image_id"])

            if not image or image["exists"] != "True":
                raise ValueError("Missing source image")

            digest = image["image_sha256"]
            if len(digest) != 64:
                raise ValueError("Invalid image hash")

            if (
                row["split"] != "EXPOSED_REFERENCE"
                or row["exposure_status"] != "EXPOSED"
            ):
                raise ValueError("Unexpected historical exposure")

            group_id = "byte-sha256:" + digest
            groups[group_id].add(row["source_image_id"])
            counts[row["original_cohort"]] += 1

            writer.writerow({
                "dataset": row["dataset"],
                "source_image_id": row["source_image_id"],
                "source_group_id": group_id,
                "source_image_sha256": digest,
                "query_id": row["query_id"],
                "original_cohort": row["original_cohort"],
                "proposed_split": "EXPOSED_REFERENCE",
                "exposure_status": "EXPOSED",
                "lineage_status": "HISTORICAL_EXPOSED",
                "artifact_sha256": row["artifact_sha256"],
                "protocol_id": row["protocol_id"],
            })

    report = {
        "schema": "countlab_p03_split_proposal_summary_v1",
        "pair_rows": len(records),
        "declared_source_ids": len(by_id),
        "byte_exact_source_groups": len(groups),
        "multi_id_byte_groups": sum(
            len(ids) > 1 for ids in groups.values()
        ),
        "rows_per_historical_cohort": dict(sorted(counts.items())),
        "proposed_split": "EXPOSED_REFERENCE_ONLY",
        "new_train_dev_cal_final_sealed": "NONE_APPROVED",
        "g0_decision": "HOLD",
        "source_ledger_sha256": sha256(LEDGER),
        "roster_sha256": sha256(OUTPUT),
        "warning": (
            "Byte-exact grouping is provisional. Re-encoded "
            "duplicates and exposure history remain unverified."
        ),
    }

    with SUMMARY.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("P03 SPLIT PROPOSAL ROSTER CREATED")
    print("Pairs:", report["pair_rows"])
    print("Source IDs:", report["declared_source_ids"])
    print("Byte groups:", report["byte_exact_source_groups"])
    print("Duplicate groups:", report["multi_id_byte_groups"])
    print("G0:", report["g0_decision"])
    print("OUTPUT:", OUTPUT)


if __name__ == "__main__":
    main()
