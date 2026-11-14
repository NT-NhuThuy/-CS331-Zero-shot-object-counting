"""Prepare and validate source-blinded independent-review label contracts."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "target_aware_counting/output/fqie_gate0_20261104_v1"
DEST = GATE / "independent_review"
CONFLICT = GATE / "audit/conflict_review_template.jsonl"
ADDITIONS = GATE / "audit/addition_review/review_manifest.jsonl"
SEMANTIC = ROOT / "target_aware_counting/output/fqie_h2_alpha_query_switch_20261104_v1/blind_review_template.jsonl"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in value)
    if path.exists() and path.read_text() != data:
        raise RuntimeError(f"review contract differs: {path}")
    path.write_text(data)


def prepare():
    conflict = rows(CONFLICT)
    # The original source-blind template predates provenance validation and
    # omitted review_date.  Add the field to the copied contract only; do not
    # modify labels or reveal the source key.
    for row in conflict:
        row.setdefault("review_date", None)
    additions = []
    allowed_add = ["target_single_instance", "target_multi_instance_box", "target_part_only",
                   "distractor", "insufficient_visible_evidence", "uncertain"]
    for card in rows(ADDITIONS):
        for candidate in card["candidates"]:
            additions.append({"item_id": card["item_id"], "card": card["card"],
                              "query": card["query"], "review_index": candidate["review_index"],
                              "label": "UNREVIEWED", "allowed_labels": allowed_add,
                              "reviewer_id": None, "review_date": None, "notes": None})
    semantic = rows(SEMANTIC)
    for row in semantic:
        card = Path(row["card_file"])
        if not card.is_absolute():
            row["card_file"] = str((SEMANTIC.parent / card).resolve())
    if len(conflict) != 30 or len(additions) != 300 or len(semantic) != 8:
        raise RuntimeError("unexpected review denominator")
    write_once(DEST / "conflict_labels.jsonl", conflict)
    write_once(DEST / "addition_labels.jsonl", additions)
    write_once(DEST / "semantic_labels.jsonl", semantic)
    contract = {"status": "SOURCE_BLINDED_REVIEW_READY", "created_date": "2026-11-04",
                "input_sha256": {"conflict": sha(CONFLICT), "additions": sha(ADDITIONS), "semantic": sha(SEMANTIC)},
                "denominators": {"conflict_pairs": 30, "addition_candidates": 300, "semantic_regions": 8},
                "rule": "reviewer must inspect only card and query before source keys, model scores, or GT audit"}
    p = DEST / "contract.json"
    payload = json.dumps(contract, indent=2) + "\n"
    if p.exists() and p.read_text() != payload:
        raise RuntimeError("review contract changed")
    p.write_text(payload)
    print(json.dumps(contract["denominators"]))


def repair_schema():
    """Repair only missing review metadata/path fields in an existing contract.

    This is deliberately not a general migration: existing labels and reviewer
    provenance are preserved byte-for-byte at the value level.
    """
    conflict_path = DEST / "conflict_labels.jsonl"
    semantic_path = DEST / "semantic_labels.jsonl"
    conflict = rows(conflict_path)
    semantic = rows(semantic_path)
    changed = {"conflict_review_date": 0, "semantic_card_path": 0}
    for row in conflict:
        if "review_date" not in row:
            row["review_date"] = None
            changed["conflict_review_date"] += 1
    for row in semantic:
        card = Path(row["card_file"])
        if not card.is_absolute():
            row["card_file"] = str((SEMANTIC.parent / card).resolve())
            changed["semantic_card_path"] += 1
    # These are mutable reviewer work files, unlike frozen model artifacts.
    # Rewriting is safe because every row/value is retained and only the two
    # schema defects above are changed.
    for path, values in ((conflict_path, conflict), (semantic_path, semantic)):
        path.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n"
                                for x in values))
    print(json.dumps(changed))


def valid_date(value):
    try:
        dt.date.fromisoformat(value)
        return True
    except (TypeError, ValueError):
        return False


def validate():
    errors = []
    for name in ("conflict_labels.jsonl", "addition_labels.jsonl"):
        for i, row in enumerate(rows(DEST / name), 1):
            if row["label"] not in row["allowed_labels"]:
                errors.append(f"{name}:{i}: invalid/unreviewed label")
            if not isinstance(row.get("reviewer_id"), str) or not row["reviewer_id"].strip():
                errors.append(f"{name}:{i}: missing reviewer_id")
            if not valid_date(row.get("review_date")):
                errors.append(f"{name}:{i}: missing/invalid review_date")
    for i, row in enumerate(rows(DEST / "semantic_labels.jsonl"), 1):
        for field in ("query_a_matches_region", "query_b_matches_region"):
            if row.get(field) not in ("YES", "NO", "UNKNOWN"):
                errors.append(f"semantic_labels.jsonl:{i}: invalid {field}")
        if not isinstance(row.get("reviewer_id"), str) or not row["reviewer_id"].strip():
            errors.append(f"semantic_labels.jsonl:{i}: missing reviewer_id")
        if not valid_date(row.get("review_date")):
            errors.append(f"semantic_labels.jsonl:{i}: missing/invalid review_date")
    report = {"status": "PASS" if not errors else "INCOMPLETE", "errors": errors,
              "labels_sha256": {name: sha(DEST / name) for name in
                                ("conflict_labels.jsonl", "addition_labels.jsonl", "semantic_labels.jsonl")}}
    p = DEST / "validation.json"
    p.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "errors": len(errors)}))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "repair-schema", "validate"))
    {"prepare": prepare, "repair-schema": repair_schema,
     "validate": validate}[parser.parse_args().action]()
