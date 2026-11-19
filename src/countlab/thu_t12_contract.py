"""C1 handoff gates for UNICOUNT T12 (CPU-only, no labels in inference).

This validates *declared* source identities and paired scoring metadata. It never
certifies an unexposed evaluation roster from self-asserted metadata alone.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict, Counter
from pathlib import Path

HASH = re.compile(r"^[0-9a-f]{64}$")
CLEAN = {"DEV", "CAL", "FINAL_SEALED"}
SPLITS = CLEAN | {"TRAIN", "EXPOSED_REFERENCE", "UNASSIGNED"}
ARMS = {"A0", "A1", "A3", "A4", "A5"}
LABEL_KEYS = {"presence_label", "count_target", "gt_count", "gt", "ground_truth", "source_gt_count", "true_count", "label_count"}
REQUIRED = {"dataset", "image_id", "query_id", "query_text", "source_image_id", "source_group_id", "image_sha256", "split", "exposure_status", "lineage_status", "eligibility", "provenance_evidence_sha256"}


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value(row: dict, key: str) -> str:
    return str(row.get(key) or "").strip()


def audit_roster(rows: list[dict]) -> dict:
    """Validate a roster, fail-closed on clean split claims.

    PASS is metadata-level only; clean-eligible rows additionally require
    external provenance approval, not something this function can grant.
    """
    errors: list[str] = []
    if not rows:
        errors.append("empty roster")
    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    hash_groups: dict[str, set[str]] = defaultdict(set)
    seen_pairs: set[tuple[str, str]] = set()
    split_counts: Counter = Counter()
    for i, row in enumerate(rows):
        missing = REQUIRED - set(row)
        if missing:
            errors.append(f"row {i}: missing columns {sorted(missing)}")
            continue
        for key in ("dataset", "image_id", "query_id", "query_text", "source_image_id", "source_group_id"):
            if not _value(row, key):
                errors.append(f"row {i}: empty {key}")
        image_hash = _value(row, "image_sha256").lower()
        if not HASH.fullmatch(image_hash):
            errors.append(f"row {i}: invalid image_sha256")
        split = _value(row, "split")
        if split not in SPLITS:
            errors.append(f"row {i}: invalid split {split}")
        exposure = _value(row, "exposure_status")
        lineage = _value(row, "lineage_status")
        eligibility = _value(row, "eligibility")
        if eligibility not in {"COUNT_ELIGIBLE", "PRESENCE_ONLY", "INELIGIBLE", "UNVERIFIED"}:
            errors.append(f"row {i}: invalid eligibility")
        pair = (_value(row, "dataset"), _value(row, "query_id"))
        if pair in seen_pairs:
            errors.append(f"row {i}: duplicate dataset/query_id {pair}")
        seen_pairs.add(pair)
        if split == "EXPOSED_REFERENCE":
            if exposure != "EXPOSED":
                errors.append(f"row {i}: exposed reference must be EXPOSED")
        elif split in CLEAN | {"TRAIN"}:
            if exposure != "UNEXPOSED" or lineage != "VERIFIED":
                errors.append(f"row {i}: clean split requires UNEXPOSED and VERIFIED lineage")
            if not HASH.fullmatch(_value(row, "provenance_evidence_sha256").lower()):
                errors.append(f"row {i}: missing provenance evidence hash")
        if split in CLEAN and eligibility == "UNVERIFIED":
            errors.append(f"row {i}: clean split has UNVERIFIED eligibility")
        if split == "FINAL_SEALED" and _value(row, "evaluation_access") != "SEALED":
            errors.append(f"row {i}: FINAL_SEALED requires SEALED evaluation access")
        if split in CLEAN | {"TRAIN"} and any(k in row and _value(row, k) for k in LABEL_KEYS):
            errors.append(f"row {i}: label values in shared clean roster; provide offline labels separately")
        sid = _value(row, "source_image_id")
        gid = _value(row, "source_group_id")
        group_splits[gid].add(split)
        source_splits[sid].add(split)
        if HASH.fullmatch(image_hash):
            hash_splits[image_hash].add(split)
            hash_groups[image_hash].add(gid)
        split_counts[split] += 1
    for kind, mapping in (("source group", group_splits), ("source image", source_splits), ("image SHA256", hash_splits)):
        for key, splits in mapping.items():
            if len(splits) > 1 and splits.intersection(CLEAN | {"TRAIN"}):
                errors.append(f"{kind} leakage {key}: {sorted(splits)}")
    for sha, groups in hash_groups.items():
        if len(groups) > 1:
            errors.append(f"identical image SHA256 with different source_group_id: {sha}")
    return {"schema": "thu_t12_roster_audit_v1", "metadata_status": "PASS" if not errors else "FAIL",
            "g0_data_decision": "HOLD", "n_pairs": len(rows), "split_counts": dict(sorted(split_counts.items())),
            "issues": errors, "limitations": ["Hash grouping is byte-exact only", "No clean cohort certified from manifest assertions", "Labels must remain offline"]}


def validate_scoring_contract(contract: dict) -> dict:
    """Validate protocol completeness, never infer that B5/B6/B7 mean A-IDs."""
    issues = []
    if contract.get("schema") != "thu_t12_scoring_contract_v1":
        issues.append("wrong schema")
    arms = contract.get("arms")
    if not isinstance(arms, dict) or set(arms) != ARMS:
        issues.append("arms must be exactly A0,A1,A3,A4,A5")
    else:
        for arm, spec in arms.items():
            if not isinstance(spec, dict):
                issues.append(f"{arm}: invalid specification")
                continue
            if not _value(spec, "implementation_id") or not _value(spec, "method_commit"):
                issues.append(f"{arm}: missing implementation_id/method_commit")
            for key in ("unary_budget", "pair_budget", "recovery_budget"):
                v = spec.get(key)
                if type(v) is not int or v < 0:
                    issues.append(f"{arm}: invalid {key}")
            if spec.get("budget_unit") != "observed_calls":
                issues.append(f"{arm}: budget_unit must be observed_calls")
    for key in ("frozen_pool_manifest_sha256", "paired_roster_sha256", "policy_sha256"):
        if not HASH.fullmatch(_value(contract, key)):
            issues.append(f"invalid/missing {key}")
    if contract.get("gt_access") != "OFFLINE_EVALUATOR_ONLY":
        issues.append("GT access must be OFFLINE_EVALUATOR_ONLY")
    if contract.get("missing_evidence") != "UNKNOWN_NOT_NO":
        issues.append("missing evidence must remain UNKNOWN")
    if contract.get("cost_accounting") != "MEASURED_NOT_REPLAY_ESTIMATE":
        issues.append("cost must be measured rather than assumed from masked slots")
    if contract.get("abstention_scoring") != "REPORT_RATE_AND_FORCED_FALLBACK":
        issues.append("report abstention rate and separate fallback count")
    if contract.get("metric_definitions") != {"present": "MAE_PRESENT_ONLY", "absent": "AZA_EXACT_ZERO", "false_zero": "FZR_PLUS_ON_PRESENT", "overall": "MAE_WITH_FALLBACK"}:
        issues.append("metric definitions do not match the locked dictionary")
    if contract.get("selection_split") not in {"CAL", "FIXED_PRE_REGISTERED"}:
        issues.append("policy selection must be CAL or pre-registered")
    if contract.get("source_disjoint_status") != "VERIFIED_EXTERNALLY":
        issues.append("no external source-disjoint verification")
    return {"schema": "thu_t12_contract_validation_v1", "status": "PASS" if not issues else "HOLD",
            "issues": issues, "interpretation": "PASS validates declared protocol fields, not dataset provenance or GPU observations"}


def main() -> None:
    parser = argparse.ArgumentParser(description="C1 T12 roster and scoring preflight; no GT inference")
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roster = audit_roster(_read(args.roster))
    result = {"roster": roster, "roster_sha256": _sha(args.roster), "scoring": None}
    if args.contract:
        result["scoring"] = validate_scoring_contract(json.loads(args.contract.read_text(encoding="utf-8")))
        result["scoring_contract_sha256"] = _sha(args.contract)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"metadata_status": roster["metadata_status"], "g0": roster["g0_data_decision"], "scoring": result["scoring"]["status"] if result["scoring"] else "NOT_PROVIDED", "out": str(args.out)}, indent=2))
    if roster["metadata_status"] == "FAIL" or (result["scoring"] and result["scoring"]["status"] != "PASS"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
