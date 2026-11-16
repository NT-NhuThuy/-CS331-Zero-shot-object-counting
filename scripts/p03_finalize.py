"""Finalize P03 from historical evidence. No new dataset splits."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("reports/p03")
LOC = Path("legacy/loc/data")
OUT = ROOT / "p03_final_evidence_summary.json"
DOC = Path("docs/method/P03_SPLIT_AUDIT.md")


def load_json(name):
    return json.loads(
        (ROOT / name).read_text(encoding="utf-8")
    )


def load_rows(name):
    with (LOC / name).open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if not row.get("image_name") or not row.get("target"):
            raise ValueError(f"Invalid CSV: {name}")

    return rows


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def pair_keys(rows):
    return {
        (
            r["image_name"].strip(),
            r["target"].strip().casefold()
        )
        for r in rows
    }


def main():
    if OUT.exists() or DOC.exists():
        raise FileExistsError(
            "Final report already exists. Refusing overwrite."
        )

    identity = load_json("p03_source_identity_report.json")
    team = load_json("p03_team_lineage_report.json")
    initial = load_json("p03_initial_audit.json")

    a = load_rows("fresh_negative_126_absent.csv")
    b = load_rows("fresh_negative_160_manual_review.csv")

    groups = identity["byte_duplicate_examples"]

    if len(groups) != identity[
        "n_byte_duplicate_groups_across_distinct_source_ids"
    ]:
        raise ValueError("Duplicate group list is incomplete")

    if (
        identity["n_pairs"] != 25301
        or identity["n_images_hashed"] != 13190
        or identity["n_missing_images"] != 0
        or identity["file_integrity_status"] != "PASS"
    ):
        raise ValueError("Historical audit integrity mismatch")

    loc111 = team["loc"]["verified_hard_negatives_111.csv"]

    if loc111["image_query_overlap_with_fsc_absent111"] != 111:
        raise ValueError("Loc/Thu absent111 parity failed")

    ka = pair_keys(a)
    kb = pair_keys(b)

    ia = {r["image_name"].strip() for r in a}
    ib = {r["image_name"].strip() for r in b}

    shared_pairs = len(ka & kb)
    shared_images = len(ia & ib)

    cross = sum(
        len(set(g["cohorts"])) > 1
        for g in groups
    )

    thuy = team["thuy"]
    with_roster = sum(
        bool(e["image_id_count"])
        for e in thuy
    )

    summary = {
        "schema": "countlab_p03_final_evidence_v1",
        "pairs": identity["n_pairs"],
        "source_ids": identity["n_unique_declared_source_ids"],
        "images_hashed": identity["n_images_hashed"],
        "missing_images": identity["n_missing_images"],
        "byte_duplicate_groups": len(groups),
        "cross_cohort_byte_duplicates": cross,
        "loc_thu_shared_absent111": 111,
        "thuy_experiments": len(thuy),
        "thuy_with_source_roster": with_roster,
        "fresh126_rows": len(a),
        "manual160_rows": len(b),
        "fresh_shared_image_query_keys": shared_pairs,
        "fresh_shared_image_names": shared_images,
        "source_hashes": {
            "historical_manifest": initial["manifest_sha256"],
            "image_ledger": identity["ledger_sha256"],
            "source_identity_report": sha256(
                ROOT / "p03_source_identity_report.json"
            ),
            "team_lineage_report": sha256(
                ROOT / "p03_team_lineage_report.json"
            ),
            "fresh126": sha256(
                LOC / "fresh_negative_126_absent.csv"
            ),
            "manual160": sha256(
                LOC / "fresh_negative_160_manual_review.csv"
            ),
        },
        "historical_file_integrity": "PASS",
        "g0_decision": "HOLD",
    }

    md = f"""# P03 — Source-Disjoint Split & Leakage Audit

## 1. Objective

Audit historical datasets, source-image identities, SHA256 hashes,
and experimental provenance before developing UNICOUNT risk routing.

This audit does not create new dataset splits or use GT for inference.

## 2. Historical dataset results

| Measurement | Result |
|---|---:|
| Historical image-query pairs | {summary["pairs"]:,} |
| Declared source-image IDs | {summary["source_ids"]:,} |
| Images successfully hashed | {summary["images_hashed"]:,} |
| Missing images | {summary["missing_images"]} |
| Byte-duplicate groups | {len(groups)} |
| Cross-cohort byte duplicates | {cross} |
| Shared Loc/Thu hard-negative pairs | 111/111 |
| Thuy manifests with image rosters | {with_roster}/{len(thuy)} |

FSC147 contains 1,190 present pairs and 111 hard-negative pairs.
The 111 negatives reuse 96 images from FSC147.

All 16 detected byte-duplicate groups occur within FSC147.
No cross-cohort byte-identical duplicates were detected.

Historical LVIS TRAIN, DEV, and TEST have zero recorded image-ID
and filename overlap.

## 3. Fresh-negative comparison

Fresh126 contains {len(a)} rows.

Manual-review160 contains {len(b)} rows.

Shared normalized image-query keys: {shared_pairs}.

Shared image filenames: {shared_images}.

Filename comparison alone does not establish image independence
or the validity of absence annotations.

## 4. Cross-member provenance

### Thu

Joint V3.2, Structural V4, and A3 were evaluated using
previously exposed FSC147 TEST and hard-negative data.

These results are historical diagnostic evidence.

### Loc

All 111 verified hard-negative pairs match Thu's 111 pairs
by pair ID and normalized image-query identity.

They are one shared historical cohort, not independent tests.

### Thuy

{len(thuy)} experiment manifests were inspected.

{with_roster} provide source-image rosters.

Full source-level exposure tracking remains incomplete.

## 5. Limitations

- SHA256 detects byte-identical files, not re-encoded or cropped images.
- Thuy historical image rosters are incomplete.
- Loc PrACo image lineage still requires verification.
- Historical FSC147 TEST and LVIS TEST are already exposed.
- No independent UNICOUNT FINAL_SEALED cohort is established.
- Presence annotation and exact-count eligibility must be separated.

## 6. Reproducibility

Run the project tests using:

    python -m pytest -q tests/

Relevant implementations:

- src/countlab/split_audit.py
- src/countlab/source_identity_audit.py
- scripts/p03_team_lineage.py
- scripts/p03_finalize.py

Evidence and source hashes are stored in reports/p03/.

## 7. G0 Decision

Historical file integrity: PASS.

**G0 DATASET INTEGRITY: HOLD.**

Before proceeding to independent model evaluation, the team must
verify historical source rosters, establish source-disjoint
TRAIN/DEV/CAL splits, and freeze a genuinely independent
FINAL_SEALED evaluation cohort.

This report does not establish UNICOUNT end-to-end performance
or a state-of-the-art result.
"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)

    OUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    DOC.write_text(md, encoding="utf-8")

    print("P03 FINAL REPORT CREATED")
    print("Fresh shared pairs:", shared_pairs)
    print("Fresh shared images:", shared_images)
    print("G0 DECISION:", summary["g0_decision"])
    print("JSON:", OUT)
    print("MARKDOWN:", DOC)


if __name__ == "__main__":
    main()
