"""P03: Historical team lineage reconciliation. Read-only inputs."""

import csv
import hashlib
import json
from pathlib import Path
import os


ROOT = Path(os.environ["UNICOUNT_OLD_ROOT"]).expanduser()
LOC = Path("legacy/loc/data")
THUY = Path("legacy/thuy/experiments")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(image, query):
    return (str(image).strip(), str(query).strip().casefold())


def main():
    fsc_path = ROOT / (
        "generated/fsc147_presence_v4/test_pairs_1301.csv"
    )
    fsc = read_csv(fsc_path)

    fsc_all_images = {r["file_name"] for r in fsc}
    fsc_all_keys = {
        key(r["file_name"], r["query"]) for r in fsc
    }
    fsc_absent = [
        r for r in fsc
        if r["pair_type"] == "VERIFIED_ABSENT_HARD_NEGATIVE"
    ]
    absent_ids = {r["pair_id"] for r in fsc_absent}
    absent_keys = {
        key(r["file_name"], r["query"]) for r in fsc_absent
    }

    files = [
        "verified_hard_negatives_111.csv",
        "fresh_negative_126_absent.csv",
        "fresh_negative_160_manual_review.csv",
    ]

    loc_results = {}

    for name in files:
        path = LOC / name
        rows = read_csv(path)
        is_111 = name.startswith("verified_")

        query_col = "query_target" if is_111 else "target"

        images = {
            r["image_name"].strip() for r in rows
        }
        pairs = {
            key(r["image_name"], r[query_col])
            for r in rows
        }

        info = {
            "rows": len(rows),
            "sha256": sha(path),
            "unique_images": len(images),
            "image_overlap_with_fsc": len(
                images & fsc_all_images
            ),
            "image_query_overlap_with_fsc": len(
                pairs & fsc_all_keys
            ),
            "image_query_overlap_with_fsc_absent111": len(
                pairs & absent_keys
            ),
        }

        if is_111:
            ids = {r["pair_id"] for r in rows}
            info["pair_id_overlap_with_fsc_absent111"] = len(
                ids & absent_ids
            )
            info["unique_pair_ids"] = len(ids)

        loc_results[name] = info

    thuy_results = []

    for path in sorted(THUY.glob("E*/manifest.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        thuy_results.append({
            "experiment": path.parent.name,
            "sha256": sha(path),
            "status": data.get("status"),
            "cohort_id": data.get("cohort_id"),
            "image_id_count": len(data.get("image_ids") or []),
        })

    report = {
        "schema": "countlab_p03_team_lineage_v1",
        "fsc147": {
            "rows": len(fsc),
            "absent_pairs": len(fsc_absent),
            "sha256": sha(fsc_path),
        },
        "loc": loc_results,
        "thuy": thuy_results,
        "g0_decision": "HOLD",
        "limitations": [
            "Name and query matching do not prove shared protocol.",
            "Thuy manifests lack source image rosters.",
            "Fresh negative label verification is not established.",
            "No independent FINAL_SEALED cohort is proven.",
        ],
    }

    out = Path("reports/p03/p03_team_lineage_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("=== LOC LINEAGE ===")
    for name, info in loc_results.items():
        print(name, json.dumps(info, indent=2))

    print("\n=== THUY LINEAGE ===")
    print("Experiments:", len(thuy_results))
    print("With image rosters:", sum(
        r["image_id_count"] > 0 for r in thuy_results
    ))

    print("\nG0:", report["g0_decision"])
    print("REPORT:", out)


if __name__ == "__main__":
    main()
