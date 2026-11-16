"""P03: Byte-level source identity audit for FSC147 and LVIS.

CPU-only. Does not modify original datasets.
Historical datasets remain exposed; no new sealed split is claimed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


COHORTS = {
    "FSC147_1301": (
        "generated/fsc147_presence_v4/test_pairs_1301.csv",
        None,
    ),
    "LVIS_TRAIN": (
        "generated/joint_semantic_counting_v1/lvis_train_semantic_pairs_v1.csv",
        "data/lvis_v1/images/semantic_v1/train",
    ),
    "LVIS_DEV": (
        "generated/joint_semantic_counting_v1/lvis_semantic_dev_pairs_v1.csv",
        "data/lvis_v1/images/semantic_v1/val",
    ),
    "LVIS_TEST": (
        "generated/joint_semantic_counting_v1/lvis_semantic_test_pairs_v1.csv",
        "data/lvis_v1/images/semantic_v1/val",
    ),
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def collect(
    old_root: Path,
    manifest_path: Path | None = None,
) -> tuple[list[dict], dict]:

    old_root = old_root.resolve()

    indexed = {}
    input_artifacts = {}
    cohort_pair_counts = {}
    path_conflicts = []

    # 1. Read historical source manifests
    for cohort, (csv_rel, folder_rel) in COHORTS.items():

        csv_path = old_root / csv_rel

        input_artifacts[cohort] = {
            "path": csv_rel,
            "sha256": file_sha256(csv_path),
        }

        pairs = 0

        with csv_path.open(
            encoding="utf-8-sig",
            newline="",
        ) as f:

            for row in csv.DictReader(f):
                pairs += 1

                if cohort == "FSC147_1301":
                    identity = row["file_name"].strip()
                    raw_path = Path(row["image_path"])

                    image = (
                        raw_path
                        if raw_path.is_absolute()
                        else old_root / raw_path
                    )

                    sid = "FSC147:" + identity

                else:
                    identity = row["image_id"].strip()

                    image = (
                        old_root
                        / folder_rel
                        / row["file_name"].strip()
                    )

                    sid = "LVIS:" + identity

                if not identity:
                    raise ValueError(
                        f"Empty image identity: {cohort}:{pairs}"
                    )

                image = Path(os.path.abspath(image))

                try:
                    relative_path = (
                        image.relative_to(old_root).as_posix()
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Image outside dataset root: {image}"
                    ) from exc

                if sid in indexed:
                    if indexed[sid]["image_path"] != relative_path:
                        path_conflicts.append({
                            "source_image_id": sid,
                            "cohort": cohort,
                        })

                    indexed[sid]["cohorts"].add(cohort)

                else:
                    indexed[sid] = {
                        "source_image_id": sid,
                        "image_path": relative_path,
                        "cohorts": {cohort},
                    }

        cohort_pair_counts[cohort] = pairs

    # 2. Verify historical manifest provenance
    differences = {
        "missing_in_manifest": 0,
        "extra_in_manifest": 0,
    }

    if manifest_path is not None:

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        if manifest.get("schema") != "countlab_split_manifest_v1":
            raise ValueError("Unsupported manifest schema")

        for cohort, artifact in input_artifacts.items():
            expected = (
                manifest.get("artifacts", {})
                .get(cohort, {})
                .get("sha256")
            )

            if expected != artifact["sha256"]:
                raise ValueError(
                    f"Frozen CSV SHA mismatch: {cohort}"
                )

        declared = {
            r["source_image_id"]
            for r in manifest["rows"]
        }

        actual = set(indexed)

        differences = {
            "missing_in_manifest": len(actual - declared),
            "extra_in_manifest": len(declared - actual),
        }

    # 3. Hash each unique source image
    ledger = []
    missing = []
    digest_groups = defaultdict(list)

    for sid, src in sorted(indexed.items()):

        image = old_root / src["image_path"]
        exists = image.is_file()

        digest = file_sha256(image) if exists else ""

        result = {
            "source_image_id": sid,
            "cohorts": ";".join(sorted(src["cohorts"])),
            "image_path": src["image_path"],
            "exists": exists,
            "image_sha256": digest,
        }

        ledger.append(result)

        if exists:
            digest_groups[digest].append(result)
        else:
            missing.append({
                "source_image_id": sid,
                "image_path": src["image_path"],
            })

    # 4. Detect identical image bytes under different IDs
    duplicate_groups = []

    for digest, group in sorted(digest_groups.items()):

        if len(group) <= 1:
            continue

        duplicate_groups.append({
            "image_sha256": digest,
            "source_image_ids": [
                r["source_image_id"] for r in group
            ],
            "cohorts": sorted({
                c
                for r in group
                for c in r["cohorts"].split(";")
            }),
        })

    complete = (
        not missing
        and not path_conflicts
        and not any(differences.values())
    )

    report = {
        "schema": "countlab_p03_image_identity_report_v1",
        "kind": "byte_exact_identity_only",
        "source_csvs": input_artifacts,
        "cohort_pair_counts": cohort_pair_counts,
        "n_pairs": sum(cohort_pair_counts.values()),
        "n_unique_declared_source_ids": len(indexed),
        "n_images_hashed": len(ledger) - len(missing),
        "n_missing_images": len(missing),
        "missing_image_examples": missing[:30],
        "n_id_path_conflicts": len(path_conflicts),
        "id_path_conflict_examples": path_conflicts[:30],
        "historical_manifest_comparison": differences,
        "n_byte_duplicate_groups_across_distinct_source_ids":
            len(duplicate_groups),
        "byte_duplicate_examples": duplicate_groups[:30],
        "file_integrity_status":
            "PASS" if complete else "HOLD",
        "g0_decision": "HOLD",
        "limits": [
            "SHA256 only detects byte-identical images.",
            "Re-encoded or cropped duplicates may be missed.",
            "Historical cohorts remain exposed.",
            "No independent FINAL_SEALED split is established.",
            "Thuy/Loc original-image lineage is not yet verified.",
            "GT values are not used by this audit.",
        ],
    }

    return ledger, report


def write_outputs(
    ledger: list[dict],
    report: dict,
    output_dir: Path,
) -> tuple[Path, Path]:

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "p03_image_byte_hashes.csv"
    json_path = output_dir / "p03_source_identity_report.json"

    if csv_path.exists() or json_path.exists():
        raise FileExistsError(
            "P03 outputs exist. Choose a new output directory."
        )

    with csv_path.open(
        "x", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_image_id",
                "cohorts",
                "image_path",
                "exists",
                "image_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(ledger)

    report["ledger_sha256"] = file_sha256(csv_path)

    with json_path.open("x", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        f.write("\n")

    return csv_path, json_path


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--old-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--historical-manifest",
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/p03"),
    )

    args = parser.parse_args()

    ledger, report = collect(
        args.old_root,
        args.historical_manifest,
    )

    csv_path, json_path = write_outputs(
        ledger,
        report,
        args.output_dir,
    )

    fields = [
        "n_pairs",
        "n_unique_declared_source_ids",
        "n_images_hashed",
        "n_missing_images",
        "n_id_path_conflicts",
        "n_byte_duplicate_groups_across_distinct_source_ids",
        "historical_manifest_comparison",
        "file_integrity_status",
        "g0_decision",
    ]

    print(json.dumps(
        {k: report[k] for k in fields},
        indent=2,
    ))

    print("LEDGER:", csv_path)
    print("REPORT:", json_path)


if __name__ == "__main__":
    main()
