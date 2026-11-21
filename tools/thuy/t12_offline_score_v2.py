"""C3 T12 offline scoring v2 candidate. GT never enters inference.

This is an ADDITIVE, NOT TEAM-APPROVED evaluation adapter. Source roster and
scoring contract use C1's validator at runtime; no implicit G0 approval.
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("A0", "A1", "A3", "A4", "A5")
PROTOCOL_SHA = "7ee56cb1542b594c16d7a6d0f4c5e7f524a7f5407e1e4fa688eaaac2051be5c9"
LABEL_KEYS = {
    "dataset", "image_id", "query_id", "source_image_id",
    "source_group_id", "split", "presence_label", "gt_count",
}
PRESENT = "PRESENT"
ABSENT = "ABSENT"
COUNT = "COUNT_ELIGIBLE"
PRESENCE_ONLY = "PRESENCE_ONLY"
INELIGIBLE = "INELIGIBLE"
CLEAN_SPLIT = "DEV"
FORBIDDEN_PREDICTION_KEYS = {
    "gt", "gt_count", "ground_truth", "groundtruth", "count_target",
    "true_count", "reference_count", "gt_boxes", "gt_points",
    "gt_annotations", "annotations", "is_absent", "is_present",
    "count_label", "presence_label", "gold_count", "gold_labels",
}


def require(condition, code):
    if not condition:
        raise ValueError("STOP: " + code)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def read_jsonl(path, name):
    require(path.is_file(), name + "_MISSING")
    result = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        require(isinstance(obj, dict), f"{name}_NOT_DICT_LINE_{index}")
        result.append(obj)
    require(bool(result), name + "_EMPTY")
    return result


def read_roster(path):
    require(path.is_file(), "ROSTER_MISSING")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), "ROSTER_EMPTY")
    return rows


def key(row):
    for field in ("image_id", "query_id", "source_group_id"):
        require(nonempty(row.get(field)), "INVALID_" + field)
    return (row["image_id"], row["query_id"])


def unique_index(rows, name):
    result = {}
    for row in rows:
        row_key = key(row)
        require(row_key not in result, name + "_DUPLICATE_IMAGE_QUERY")
        result[row_key] = row
    return result


def score_label(roster_row, label):
    """Provisional v2 offline-label schema; never determine eligibility from GT."""
    require(set(label) == LABEL_KEYS, "LABEL_SCHEMA_INVALID")
    for field in ("dataset", "image_id", "query_id", "source_image_id", "source_group_id"):
        require(label.get(field) == roster_row.get(field), "LABEL_ROSTER_IDENTITY_MISMATCH_" + field)
    require(label["split"] == CLEAN_SPLIT, "LABEL_SPLIT_NOT_DEV")

    status = roster_row["eligibility"]
    gt = label["gt_count"]
    present = label["presence_label"]

    if status == COUNT:
        require(type(gt) is int and gt >= 0, "COUNT_ELIGIBLE_GT_INVALID")
        require(present in (PRESENT, ABSENT), "COUNT_ELIGIBLE_PRESENCE_MISSING")
        require((gt > 0) == (present == PRESENT), "COUNT_PRESENCE_CONTRADICTION")
        return gt, present
    if status == PRESENCE_ONLY:
        require(gt is None, "PRESENCE_ONLY_GT_COUNT_FORBIDDEN")
        require(present in (PRESENT, ABSENT), "PRESENCE_ONLY_LABEL_INVALID")
        return None, present
    if status == INELIGIBLE:
        require(gt is None and present is None, "INELIGIBLE_LABELS_MUST_BE_NULL")
        return None, None
    raise ValueError("STOP: UNVERIFIED_OR_UNKNOWN_ELIGIBILITY")


def reject_gt_in_prediction(value):
    if isinstance(value, dict):
        for field, child in value.items():
            require(str(field).lower() not in FORBIDDEN_PREDICTION_KEYS,
                    "GT_FIELD_IN_PREDICTION_" + str(field))
            reject_gt_in_prediction(child)
    elif isinstance(value, list):
        for child in value:
            reject_gt_in_prediction(child)


def validate_prediction(prediction, roster_row, split_id, pool_id):
    reject_gt_in_prediction(prediction)
    require(prediction.get("cohort_role") == "dev", "PREDICTION_NOT_DEV")
    require(prediction.get("pool_scope") == "full_query_pool", "PREDICTION_NOT_FULL_POOL")
    require(prediction.get("split_protocol_id") == split_id, "PREDICTION_SPLIT_ID_MISMATCH")
    require(prediction.get("frozen_pool_run_id") == pool_id, "PREDICTION_POOL_MISMATCH")
    require(prediction.get("t12_protocol_sha256") == PROTOCOL_SHA, "FROZEN_PROTOCOL_MISMATCH")
    require(prediction.get("source_group_id") == roster_row["source_group_id"], "PREDICTION_SOURCE_GROUP_MISMATCH")
    require(prediction.get("gt_used") is False, "GT_USED_IN_PREDICTION")
    require(prediction.get("scene_zero_certified") is False, "SCENE_ZERO_CERTIFICATION_FORBIDDEN")
    require(prediction.get("model_calls_in_this_replay") == 0, "REPLAY_MODEL_CALLS_NOT_ZERO")
    require(prediction.get("measured_verifier_cost_attached") is False, "UNEXPECTED_REPLAY_COST_FLAG")

    arms = prediction.get("arms")
    require(isinstance(arms, dict) and set(arms) == set(ARMS), "PREDICTION_ARM_SET_MISMATCH")
    for arm in ARMS:
        require(isinstance(arms[arm], dict), "ARM_NOT_DICT_" + arm)
        require(type(arms[arm].get("forced_count")) is int and arms[arm]["forced_count"] >= 0, "INVALID_FORCED_COUNT_" + arm)

    a0 = arms["A0"]["forced_count"]
    a5 = arms["A5"]
    require(a5.get("original_fallback_count") == a0, "A5_FALLBACK_NOT_A0")
    require(arms["A1"]["forced_count"] == arms["A3"]["forced_count"], "A1_A3_PARITY_FAILED")
    require(type(arms["A4"].get("search_exact")) is bool, "A4_EXACTNESS_FLAG_MISSING")
    require(a5.get("status") in ("ABSTAIN", "PROVISIONAL_POSITIVE"), "A5_STATUS_INVALID")
    if a5["status"] == "ABSTAIN":
        require(a5.get("selective_count") is None and a5["forced_count"] == a0, "A5_ABSTAIN_FALLBACK_INVALID")
    else:
        require(type(a5.get("selective_count")) is int and a5["selective_count"] > 0, "A5_SELECTIVE_INVALID")
        require(a5["forced_count"] == a5["selective_count"], "A5_POSITIVE_FORCED_MISMATCH")


def build_items(predictions, roster, labels, *, split_id, pool_id):
    """Return complete matched DEV rows, with C1 eligibility authoritative."""
    pred_idx = unique_index(predictions, "PREDICTIONS")
    label_idx = unique_index(labels, "LABELS")
    roster_idx = unique_index(roster, "ROSTER")
    dev_idx = {k: r for k, r in roster_idx.items() if r.get("split") == CLEAN_SPLIT}
    require(bool(dev_idx), "NO_DEV_ROWS")
    require(set(pred_idx) == set(label_idx) == set(dev_idx), "DEV_JOIN_NOT_EXACT")
    require(len({(r["dataset"], r["query_id"]) for r in roster}) == len(roster), "DATASET_QUERY_NOT_UNIQUE")

    items = []
    for record_key in sorted(dev_idx):
        row = dev_idx[record_key]
        label = label_idx[record_key]
        prediction = pred_idx[record_key]
        require(row.get("exposure_status") == "UNEXPOSED", "DEV_EXPOSURE_NOT_CLEAR")
        require(row.get("lineage_status") == "VERIFIED", "DEV_LINEAGE_NOT_VERIFIED")
        require(row.get("eligibility") in (COUNT, PRESENCE_ONLY, INELIGIBLE), "DEV_ELIGIBILITY_INVALID")
        gt, presence = score_label(row, label)
        validate_prediction(prediction, row, split_id, pool_id)
        items.append({"prediction": prediction, "gt_count": gt, "presence": presence,
                      "eligibility": row["eligibility"], "source_group_id": row["source_group_id"]})
    return items


def rate(n, d):
    return n / d if d else None


def metrics(items, arm):
    eligible = [i for i in items if i["eligibility"] == COUNT]
    count_positive = [i for i in eligible if i["gt_count"] > 0]
    presence_rows = [i for i in items if i["presence"] in (PRESENT, ABSENT)]
    present = [i for i in presence_rows if i["presence"] == PRESENT]
    absent = [i for i in presence_rows if i["presence"] == ABSENT]

    def forced(item):
        return item["prediction"]["arms"][arm]["forced_count"]

    report = {
        "n_dev_total": len(items),
        "n_count_eligible": len(eligible),
        "n_presence_only": sum(i["eligibility"] == PRESENCE_ONLY for i in items),
        "n_ineligible": sum(i["eligibility"] == INELIGIBLE for i in items),
        "n_present_count_eligible": len(count_positive),
        "n_absent_count_eligible": len(eligible) - len(count_positive),
        "n_presence_evaluable": len(presence_rows),
        "n_present_presence_evaluable": len(present),
        "n_absent_presence_evaluable": len(absent),
        "overall_mae": rate(sum(abs(forced(i) - i["gt_count"]) for i in eligible), len(eligible)),
        "present_mae": rate(sum(abs(forced(i) - i["gt_count"]) for i in count_positive), len(count_positive)),
        "aza": rate(sum(forced(i) == 0 for i in absent), len(absent)),
        "fzr_plus": rate(sum(forced(i) == 0 for i in present), len(present)),
        "false_zero_positive_queries": sum(forced(i) == 0 for i in present),
    }
    if arm == "A5":
        abstained = [i for i in items if i["prediction"]["arms"]["A5"]["status"] == "ABSTAIN"]
        selective_present = [i for i in count_positive if i["prediction"]["arms"]["A5"]["status"] == "PROVISIONAL_POSITIVE"]
        report.update({
            "abstention_rate_all_dev": rate(len(abstained), len(items)),
            "abstention_rate_count_eligible": rate(sum(i["prediction"]["arms"]["A5"]["status"] == "ABSTAIN" for i in eligible), len(eligible)),
            "coverage_count_eligible": rate(sum(i["prediction"]["arms"]["A5"]["status"] == "PROVISIONAL_POSITIVE" for i in eligible), len(eligible)),
            "selective_mae_present": rate(sum(abs(i["prediction"]["arms"]["A5"]["selective_count"] - i["gt_count"]) for i in selective_present), len(selective_present)),
            "n_selective_present": len(selective_present),
        })
    if arm == "A4":
        report["non_exact_search_queries"] = sum(i["prediction"]["arms"]["A4"]["search_exact"] is False for i in items)
    return report


def paired_interval(items, left, right, metric, draws=1000):
    groups = defaultdict(list)
    for item in items:
        groups[item["source_group_id"]].append(item)
    names = sorted(groups)

    def difference(sample):
        x = metrics(sample, left)[metric]
        y = metrics(sample, right)[metric]
        return None if x is None or y is None else x - y

    observed = difference(items)
    if observed is None:
        return {"estimate": None, "interval_95": None, "reason": "METRIC_STRATUM_ABSENT"}
    if len(names) < 10:
        return {"estimate": observed, "interval_95": None, "cluster_count": len(names),
                "reason": "FEWER_THAN_10_SOURCE_CLUSTERS"}

    rng = random.Random(19011)
    values = []
    for _ in range(draws):
        sample = [item for _ in names for item in groups[names[rng.randrange(len(names))]]]
        value = difference(sample)
        if value is not None:
            values.append(value)
    if len(values) < draws * 0.8:
        return {"estimate": observed, "interval_95": None, "valid_draws": len(values),
                "reason": "BOOTSTRAP_STRATUM_UNSTABLE"}
    values.sort()
    return {"estimate": observed, "interval_95": [values[int(0.025 * len(values))],
            values[int(0.975 * len(values)) - 1]], "resample_unit": "source_group_id",
            "cluster_count": len(names), "valid_draws": len(values),
            "interpretation": "DESCRIPTIVE_NOT_CONFIRMATORY"}


def paired_help_harm(items, left, right, *, present_only=False):
    """Query-level count absolute-error changes only on count-eligible cases."""
    eligible = [i for i in items if i["eligibility"] == COUNT and (not present_only or i["gt_count"] > 0)]
    better = worse = tied = 0
    for item in eligible:
        prediction = item["prediction"]["arms"]
        lerr = abs(prediction[left]["forced_count"] - item["gt_count"])
        rerr = abs(prediction[right]["forced_count"] - item["gt_count"])
        better += lerr < rerr
        worse += lerr > rerr
        tied += lerr == rerr
    return {"n": len(eligible), "help": better, "harm": worse, "tie": tied,
            "interpretation": "DESCRIPTIVE_PAIRED_QUERY_COUNTS"}


def c1_gate(roster, contract):
    try:
        from countlab.thu_t12_contract import audit_roster, validate_scoring_contract
    except ImportError as exc:
        raise ValueError("STOP: C1_CONTRACT_MODULE_NOT_IN_WORKTREE") from exc
    audit = audit_roster(roster)
    require(audit.get("metadata_status") == "PASS", "C1_ROSTER_AUDIT_FAILED:" + repr(audit.get("issues")))
    validated = validate_scoring_contract(contract)
    require(validated.get("status") == "PASS", "C1_SCORING_CONTRACT_HOLD:" + repr(validated.get("issues")))
    return audit, validated


def run(args):
    paths = (args.predictions, args.labels, args.roster, args.contract,
             args.attestation, args.pool_manifest, args.policy, args.out)
    require(len({p.resolve() for p in paths}) == len(paths), "INPUT_OUTPUT_PATH_COLLISION")
    require(not args.out.exists(), "OUTPUT_ALREADY_EXISTS")
    require(all(p.is_file() for p in paths[:-1]), "INPUT_FILE_MISSING")

    roster = read_roster(args.roster)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    require(isinstance(contract, dict), "SCORING_CONTRACT_NOT_OBJECT")
    c1_audit, c1_validation = c1_gate(roster, contract)
    require(contract.get("paired_roster_sha256") == digest(args.roster), "ROSTER_HASH_NOT_CONTRACT_BOUND")
    require(contract.get("frozen_pool_manifest_sha256") == digest(args.pool_manifest), "POOL_HASH_NOT_CONTRACT_BOUND")
    require(contract.get("policy_sha256") == digest(args.policy), "POLICY_HASH_NOT_CONTRACT_BOUND")

    approval = json.loads(args.attestation.read_text(encoding="utf-8"))
    require(isinstance(approval, dict), "ATTESTATION_NOT_OBJECT")
    require(approval.get("status") == "approved_for_dev_scoring", "EXTERNAL_DEV_APPROVAL_MISSING")
    for flag in ("source_disjoint_verified", "candidate_ids_attested", "source_image_hashes_attested"):
        require(approval.get(flag) is True, "ATTESTATION_MISSING_" + flag)
    require(nonempty(approval.get("approved_by")) and nonempty(approval.get("approval_reference")), "ATTESTATION_PROVENANCE_MISSING")
    require(approval.get("g0_decision") == "GO_FOR_DEV" and nonempty(approval.get("g0_evidence_reference")), "G0_INDEPENDENT_RELEASE_MISSING")
    for field, file in (
        ("prediction_file_sha256", args.predictions),
        ("label_file_sha256", args.labels),
        ("roster_file_sha256", args.roster),
        ("contract_file_sha256", args.contract),
        ("pool_manifest_sha256", args.pool_manifest),
        ("policy_file_sha256", args.policy),
    ):
        require(approval.get(field) == digest(file), "ATTESTATION_HASH_MISMATCH_" + field)
    require(approval.get("t12_protocol_sha256") == PROTOCOL_SHA, "ATTESTATION_PROTOCOL_MISMATCH")
    split_id = approval.get("split_protocol_id")
    pool_id = approval.get("frozen_pool_run_id")
    require(nonempty(split_id) and nonempty(pool_id), "APPROVED_RUN_ID_MISSING")
    require(contract.get("selection_split") in ("CAL", "FIXED_PRE_REGISTERED"), "POLICY_SELECTION_NOT_LOCKED")

    prediction_rows = read_jsonl(args.predictions, "PREDICTIONS")
    labels = read_jsonl(args.labels, "LABELS")
    items = build_items(prediction_rows, roster, labels, split_id=split_id, pool_id=pool_id)
    require(bool(items), "EMPTY_SCORE_ITEMS")

    report = {
        "schema": "c3_t12_offline_scoring_v2_candidate",
        "status": "DESCRIPTIVE_DEV_ONLY_APPROVAL_METADATA_DECLARED",
        "team_approval_status": "CANDIDATE_NOT_TEAM_APPROVED",
        "c1_roster_metadata_status": c1_audit["metadata_status"],
        "c1_scoring_contract_status": c1_validation["status"],
        "g0_source_provenance": "REQUIRES_INDEPENDENT_EXTERNAL_EVIDENCE",
        "metric_policy": "COUNT_MAE_ON_COUNT_ELIGIBLE_PRESENCE_ON_EXPLICIT_VERIFIED_LABELS",
        "n_source_clusters": len({i["source_group_id"] for i in items}),
        "n_dev_total": len(items),
        "eligibility_counts": dict(sorted(Counter(i["eligibility"] for i in items).items())),
        "files_sha256": {name: digest(path) for name, path in (
            ("predictions", args.predictions), ("labels", args.labels),
            ("roster", args.roster), ("contract", args.contract),
            ("attestation", args.attestation), ("pool_manifest", args.pool_manifest),
            ("policy", args.policy))},
        "t12_protocol_sha256": PROTOCOL_SHA,
        "arms": {arm: metrics(items, arm) for arm in ARMS},
        "paired": {f"{left}_minus_{right}": {
            "help_harm_count_eligible": paired_help_harm(items, left, right),
            "help_harm_present_count_eligible": paired_help_harm(items, left, right, present_only=True),
            "metrics": {metric: paired_interval(items, left, right, metric)
                        for metric in ("overall_mae", "present_mae", "aza", "fzr_plus")},
        } for left, right in (("A4", "A0"), ("A4", "A1"), ("A5", "A0"))},
        "limitations": [
            "Provisional offline-label schema requires C1/C2 signoff",
            "C1 metadata PASS and declared scoring PASS never independently prove G0",
            "Approval file provenance and source-image roster must be externally checked",
            "C3 exporter has no per-row image SHA; hash-bound approval must attest joins",
            "No replay resource claim: measured_verifier_cost_attached=false",
            "A2/B5 and A6/B6 A7/B7 are not executed by this scorer",
            "Bootstrap intervals are descriptive and source-clustered",
            "No scene-level certified zero",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.out),
                      "output_sha256": digest(args.out), "n_dev": len(items)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "labels", "roster", "contract", "attestation", "pool_manifest", "policy", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
