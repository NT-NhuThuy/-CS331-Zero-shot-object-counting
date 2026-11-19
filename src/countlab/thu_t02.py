"""Thư T02: exposed-source / presence-label / split-eligibility audit.

Only creates GT-SEPARATE research ledgers and an EXPOSED pilot roster.
It does NOT assign TRAIN/DEV/CAL/FINAL_SEALED or turn reported absences
into independently verified labels. Does not run models or read GT during
inference. Uses only Python stdlib and project split_policy when requested.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"empty CSV: {path}")
        result = list(reader)
    if not result:
        raise ValueError(f"CSV contains no rows: {path}")
    return result


def require_fields(rows: list[dict], fields: set[str], title: str) -> None:
    missing = fields - rows[0].keys()
    if missing:
        raise ValueError(f"{title}: missing columns {sorted(missing)}")


def clean(value: str | None) -> str:
    return str(value or "").strip()


def query_key(image: str, query: str) -> tuple[str, str]:
    return (clean(image).casefold(), " ".join(clean(query).casefold().split()))


def read_attested_ledger(ledger_path: Path | None, report_path: Path | None) -> tuple[dict[str, str], str]:
    """Use image hashes only if previous P03 report attests the ledger bytes."""
    if (ledger_path is None) != (report_path is None):
        raise ValueError("--source-ledger and --source-identity-report must be supplied together")
    if ledger_path is None:
        return {}, "NOT_PROVIDED"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("ledger_sha256") != file_sha256(ledger_path)
            or report.get("file_integrity_status") != "PASS"):
        raise ValueError("P03 image ledger SHA/integrity proof failed")
    rows = read_csv(ledger_path)
    require_fields(rows, {"source_image_id", "image_sha256", "exists"}, "P03 source ledger")
    ledger = {}
    for r in rows:
        sid = clean(r["source_image_id"])
        digest = clean(r["image_sha256"])
        if not sid or sid in ledger or clean(r["exists"]).lower() != "true" or not HEX64.fullmatch(digest):
            raise ValueError(f"invalid / duplicated attested source entry: {sid}")
        ledger[sid] = digest
    return ledger, "P03_SHA_ATTESTED"


def _validated_fsc(rows: list[dict]) -> tuple[list[dict], dict[tuple[str, str], dict], dict[str, dict]]:
    require_fields(rows, {"pair_id", "file_name", "image_path", "query", "pair_type",
                          "presence_label", "count_target", "num_exemplars", "annotation_source"}, "FSC1301")
    entries: list[dict] = []
    by_absent: dict[tuple[str, str], dict] = {}
    by_pair_id: dict[str, dict] = {}
    for r in rows:
        pair_id, filename, query = map(clean, (r["pair_id"], r["file_name"], r["query"]))
        if not pair_id or not filename or not query or pair_id in by_pair_id:
            raise ValueError(f"missing or repeated FSC pair identity: {pair_id}")
        try:
            presence, count, exemplars = (int(r["presence_label"]),
                                          int(r["count_target"]), int(r["num_exemplars"]))
        except ValueError as exc:
            raise ValueError(f"noninteger label/count/exemplars: {pair_id}") from exc
        if presence not in (0, 1) or (presence == 1 and count <= 0) or (presence == 0 and count != 0):
            raise ValueError(f"inconsistent FSC presence/count: {pair_id}")
        # The historical FSC1301 source contains positive pairs with 3, 4,
        # *or 5* annotated exemplars.  Requiring exactly 3 falsely rejects
        # valid source rows (e.g. POS__4962__tree_logs has 4).  Retain the
        # original value for auditing, never rewrite it to a nominal protocol.
        if (presence == 1 and exemplars not in (3, 4, 5)) or (presence == 0 and exemplars != 0):
            raise ValueError(f"historical FSC exemplar value unexpected: {pair_id} ({exemplars})")
        if presence == 0:
            if clean(r["pair_type"]) != "VERIFIED_ABSENT_HARD_NEGATIVE":
                raise ValueError(f"not declared hard-negative: {pair_id}")
            label_status = "HISTORICAL_REPORTED_ABSENT"
            key = query_key(filename, query)
            if key in by_absent:
                raise ValueError(f"duplicate absent image-query: {key}")
            by_absent[key] = r
        else:
            label_status = "FSC_ANNOTATED_PRESENT"
        source_id = f"FSC147:{filename}"
        entry = {
            "record_id": f"fsc:{pair_id}", "pair_id": pair_id,
            "original_cohort": "FSC147_EXTENDED_1301", "source_image_id": source_id,
            "file_name": filename, "image_path": clean(r["image_path"]),
            "query_text": query, "presence_label": presence, "count_target": count,
            "label_status": label_status,
            "annotation_provenance": clean(r["annotation_source"]),
            "num_exemplars": exemplars,
            "source_group_id": "", "source_identity_proof": "", "source_image_sha256": "",
            "exposure_status": "EXPOSED", "proposed_split": "EXPOSED_REFERENCE",
            "exact_count_eligible": presence == 1,
            "pilot_eligible": True,
        }
        entries.append(entry)
        by_pair_id[pair_id] = r
    return entries, by_absent, by_pair_id


def _validate_loc111(rows: list[dict], absent: dict[tuple[str, str], dict], by_pair_id: dict[str, dict]) -> None:
    require_fields(rows, {"pair_id", "image_name", "query_target", "presence_label",
                          "count_target", "assistant_self_check", "final_decision"}, "Lộc111")
    if len(rows) != len(absent):
        raise ValueError(f"Lộc111 vs FSC absent sizes disagree: {len(rows)} != {len(absent)}")
    seen = set()
    for r in rows:
        pid = clean(r["pair_id"])
        key = query_key(r["image_name"], r["query_target"])
        if pid in seen or pid not in by_pair_id or key not in absent or clean(absent[key]["pair_id"]) != pid:
            raise ValueError(f"Lộc111 mirror pair identity mismatch: {pid}")
        if clean(r["presence_label"]) != "0" or clean(r["count_target"]) != "0":
            raise ValueError(f"Lộc111 unexpected GT: {pid}")
        if clean(r["final_decision"]) != "KEEP_NEGATIVE" or clean(r["assistant_self_check"]) != "PASS_ABSENT":
            raise ValueError(f"Lộc111 missing historical label evidence: {pid}")
        seen.add(pid)
    if len(seen) != len(absent):
        raise ValueError("Lộc111 missing FSC historical negative entries")


def _validated_fresh(rows160: list[dict], rows126: list[dict]) -> list[dict]:
    expected = {"candidate_id", "image_name", "target", "presence_label", "label_source"}
    require_fields(rows160, expected, "fresh160")
    require_fields(rows126, expected, "fresh126")
    by_id = {}
    for r in rows160:
        cid = clean(r["candidate_id"])
        if not cid or cid in by_id:
            raise ValueError("duplicate/missing fresh160 candidate ID")
        by_id[cid] = r
    seen126 = set()
    for r in rows126:
        cid = clean(r["candidate_id"])
        if cid not in by_id or cid in seen126:
            raise ValueError(f"fresh126 not unique subset: {cid}")
        orig = by_id[cid]
        if any(query_key(r["image_name"], r["target"]) != query_key(orig["image_name"], orig["target"])
               or clean(r[k]) != clean(orig[k]) for k in ("presence_label", "label_source")):
            raise ValueError(f"fresh126 mismatch with 160: {cid}")
        if clean(r["presence_label"]) != "ABSENT":
            raise ValueError(f"fresh126 has non-absent entry: {cid}")
        seen126.add(cid)
    if {cid for cid, row in by_id.items() if clean(row["presence_label"]) == "ABSENT"} != seen126:
        raise ValueError("fresh126 must cover exactly the absent subset of fresh160")
    result = []
    for cid, r in by_id.items():
        label = clean(r["presence_label"]).upper()
        if label not in {"PRESENT", "ABSENT", "UNVERIFIED"}:
            raise ValueError(f"unexpected fresh160 label: {cid}")
        single = clean(r["label_source"]) == "single_human_reviewer"
        result.append({
            "record_id": "fresh160:" + cid, "pair_id": cid,
            "original_cohort": "LOC_FRESH160", "source_image_id": "",
            "file_name": clean(r["image_name"]), "image_path": "",
            "query_text": clean(r["target"]),
            "presence_label": {"PRESENT": 1, "ABSENT": 0}.get(label),
            "count_target": None,
            "label_status": ("SINGLE_REVIEW_" + label if single else "UNVERIFIED"),
            "annotation_provenance": clean(r["label_source"]),
            "num_exemplars": None,
            "source_group_id": "", "source_identity_proof": "UNRESOLVED_ORIGINAL_SOURCE",
            "source_image_sha256": "", "exposure_status": "EXPOSED",
            "proposed_split": "EXPOSED_REFERENCE", "exact_count_eligible": False,
            "pilot_eligible": False,
        })
    return result


def _choose_pilot(fsc_rows: list[dict], n_each: int) -> list[dict]:
    """Deterministic, source-group-disjoint *within pilot*, exposed only."""
    if n_each < 0:
        raise ValueError("n_each cannot be negative")
    if n_each == 0:
        return []
    chosen, occupied = [], set()
    for label in (0, 1):  # prioritize rarer absent stratum
        candidates = [r for r in fsc_rows if r["presence_label"] == label]
        candidates.sort(key=lambda r: hashlib.sha256(("T02_PILOT_V1:" + r["pair_id"]).encode()).hexdigest())
        added = 0
        for r in candidates:
            if r["source_group_id"] in occupied:
                continue
            chosen.append(r)
            occupied.add(r["source_group_id"])
            added += 1
            if added == n_each:
                break
    return chosen


def _validate_clean_proposal(path: Path | None, exposed: list[dict]) -> dict:
    if path is None:
        return {"status": "NOT_PROVIDED", "g2_decision": "HOLD", "note": "No approved independent TRAIN/DEV/CAL/FINAL_SEALED roster"}
    from .split_policy import audit_split_proposal
    rows = read_csv(path)
    result = audit_split_proposal(rows)
    exposed_hashes = {r["source_image_sha256"] for r in exposed if r["source_image_sha256"]}
    exposed_ids = {r["source_image_id"] for r in exposed if r["source_image_id"]}
    extra = []
    for i, row in enumerate(rows):
        if row.get("proposed_split") in {"TRAIN", "DEV", "CAL", "FINAL_SEALED"}:
            if row.get("source_image_sha256") in exposed_hashes or row.get("source_image_id") in exposed_ids:
                extra.append(f"row {i}: clean proposal reuses exposed FSC source")
    result["issues"].extend(extra)
    if extra:
        result["status"] = "FAIL"
    # Metadata validation cannot certify genuinely clean lineage or labels.
    result["g2_decision"] = "HOLD"
    result["note"] = "Only metadata / available FSC overlap checked; manual provenance and labels still required"
    return result


def build(fsc: Path, loc111: Path, fresh160: Path, fresh126: Path,
          out_dir: Path, *, source_ledger: Path | None = None,
          source_identity_report: Path | None = None,
          clean_proposal: Path | None = None, pilot_each: int = 16) -> dict:
    inputs = {"fsc": fsc, "loc111": loc111, "fresh160": fresh160, "fresh126": fresh126}
    if out_dir.exists():
        raise FileExistsError(f"T02 output is create-only: {out_dir}")
    for key, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{key}: {path}")
    attested, proof = read_attested_ledger(source_ledger, source_identity_report)
    records, absent, by_pair = _validated_fsc(read_csv(fsc))
    _validate_loc111(read_csv(loc111), absent, by_pair)
    fresh_records = _validated_fresh(read_csv(fresh160), read_csv(fresh126))
    for r in records:
        digest = attested.get(r["source_image_id"])
        if digest:
            r["source_group_id"] = "byte-sha256:" + digest
            r["source_image_sha256"] = digest
            r["source_identity_proof"] = "P03_SHA_ATTESTED"
        else:
            # Only a declared-file grouping: safe for EXPOSED pilot dedup,
            # insufficient to certify source disjoint from new datasets.
            r["source_group_id"] = "declared:FSC147:" + r["file_name"]
            r["source_identity_proof"] = "DECLARED_IMAGE_ID_ONLY"
    records += fresh_records
    proposed_clean = _validate_clean_proposal(clean_proposal, records)
    pilot = _choose_pilot([r for r in records if r["pilot_eligible"]], pilot_each)
    src_groups = [r["source_group_id"] for r in pilot]
    if len(src_groups) != len(set(src_groups)):
        raise AssertionError("pilot source-group collision")
    # Historical pair IDs embed POS/HARD labels. Do NOT expose those IDs to
    # inference. Use a neutral opaque ID and retain the mapping GT-side only.
    def opaque_id(r: dict) -> str:
        key = r["source_image_id"] + "\0" + r["query_text"]
        return "q_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]

    inference_items = [{
        "query_id": opaque_id(r), "image_id": r["source_image_id"],
        "source_group_id": r["source_group_id"],
        "image_path": r["image_path"], "query_text": r["query_text"],
    } for r in pilot]
    if len({item["query_id"] for item in inference_items}) != len(inference_items):
        raise ValueError("opaque image-query key collision")
    banned = ("presence_label", "count_target", "label_status", "gt_count", "ground_truth")
    if any(any(key in item for key in banned) for item in inference_items):
        raise AssertionError("GT leaked into pilot inference input")
    counts = Counter(r["label_status"] for r in records)
    identity_counts = Counter(r["source_identity_proof"] for r in records)
    pilot_labels = Counter(r["presence_label"] for r in pilot)
    exemplar_hist = {
        "present": dict(sorted(Counter(str(r["num_exemplars"]) for r in records
                                        if r["original_cohort"] == "FSC147_EXTENDED_1301"
                                        and r["presence_label"] == 1).items())),
        "absent": dict(sorted(Counter(str(r["num_exemplars"]) for r in records
                                       if r["original_cohort"] == "FSC147_EXTENDED_1301"
                                       and r["presence_label"] == 0).items())),
    }
    report = {
        "schema": "countlab_thu_t02_audit_v1", "task": "T02", "owner": "Thu",
        "technical_status": "PASS", "g0_data_decision": "HOLD", "g2_decision": "HOLD",
        "n_records": len(records), "n_fsc_pairs": len(by_pair),
        "n_historical_absent_mirror_matched": len(absent),
        "n_fresh160": len(fresh_records),
        "n_fresh126_subset": sum(r["presence_label"] == 0 for r in fresh_records),
        "fsc_exemplar_distribution": exemplar_hist,
        "label_status_counts": dict(sorted(counts.items())),
        "source_identity_proof_counts": dict(sorted(identity_counts.items())),
        "p03_ledger_verification": proof,
        "pilot": {"n_pairs": len(pilot), "present": pilot_labels.get(1, 0),
                  "absent": pilot_labels.get(0, 0),
                  "n_source_groups": len(set(src_groups)), "role": "EXPOSED_REFERENCE_ONLY"},
        "clean_proposal_audit": proposed_clean,
        "input_sha256": {k: file_sha256(v) for k, v in inputs.items()},
        "blockers": [
            "No independent clean source-disjoint TRAIN/DEV/CAL/FINAL_SEALED roster established by T02",
            "Fresh160 absent/present labels have one reviewer only; treat as diagnostic",
            "Historical reported absent111 were not independently re-verified by this script",
            "FSC present pairs have 3/4/5 historical exemplars while absent pairs have 0; NOT a matched text-only protocol",
            "Byte SHA group proves identity only when attested P03 ledger is supplied; re-encodings/crops still possible",
            "All T02 outputs are evaluation/planning artifacts; GT labels must never enter model inference",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    columns = ["record_id", "pair_id", "original_cohort", "source_image_id", "file_name", "image_path",
               "query_text", "presence_label", "count_target", "label_status", "annotation_provenance",
               "num_exemplars",
               "source_group_id", "source_identity_proof", "source_image_sha256", "exposure_status",
               "proposed_split", "exact_count_eligible", "pilot_eligible"]
    with (out_dir / "t02_label_ledger_GT_ONLY.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(records)
    with (out_dir / "t02_pilot_labels_GT_ONLY.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "historical_pair_id", "source_group_id",
                                               "presence_label", "count_target", "label_status"])
        writer.writeheader()
        for r in pilot:
            writer.writerow({"query_id": opaque_id(r), "historical_pair_id": r["pair_id"],
                             "source_group_id": r["source_group_id"], "presence_label": r["presence_label"],
                             "count_target": r["count_target"], "label_status": r["label_status"]})
    with (out_dir / "t02_pilot_inference_inputs_GT_FREE.json").open("x", encoding="utf-8") as f:
        json.dump({"schema": "countlab_exposed_pilot_queries_v1", "role": "EXPOSED_REFERENCE",
                   "items": inference_items}, f, indent=2, ensure_ascii=False); f.write("\n")
    report["output_sha256"] = {
        name: file_sha256(out_dir / name) for name in (
            "t02_label_ledger_GT_ONLY.csv", "t02_pilot_labels_GT_ONLY.csv",
            "t02_pilot_inference_inputs_GT_FREE.json")}
    with (out_dir / "t02_audit_report.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, sort_keys=True); f.write("\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Thư T02: exposed label/source/split eligibility audit. Never certifies clean final data.")
    for name in ("fsc", "loc111", "fresh160", "fresh126"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source-ledger", type=Path)
    ap.add_argument("--source-identity-report", type=Path)
    ap.add_argument("--clean-proposal", type=Path)
    ap.add_argument("--pilot-each", type=int, default=16)
    a = ap.parse_args()
    report = build(a.fsc, a.loc111, a.fresh160, a.fresh126, a.out_dir,
                   source_ledger=a.source_ledger, source_identity_report=a.source_identity_report,
                   clean_proposal=a.clean_proposal, pilot_each=a.pilot_each)
    print(json.dumps({k: report[k] for k in ("technical_status", "g0_data_decision", "n_records",
                                              "n_historical_absent_mirror_matched", "pilot", "p03_ledger_verification")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
