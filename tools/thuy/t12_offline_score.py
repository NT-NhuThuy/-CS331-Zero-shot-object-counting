"""Offline T12 DEV scorer. Never imports counting inference."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ARMS = ("A0", "A1", "A3", "A4", "A5")
PROTOCOL_SHA = "7ee56cb1542b594c16d7a6d0f4c5e7f524a7f5407e1e4fa688eaaac2051be5c9"


def require(ok, message):
    if not ok:
        raise ValueError("STOP: " + message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path, name):
    require(path.is_file(), name + "_MISSING")
    rows = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if line.strip():
            row = json.loads(line)
            require(
                isinstance(row, dict),
                f"{name}_ROW_{number}_NOT_OBJECT",
            )
            rows.append(row)
    require(bool(rows), name + "_EMPTY")
    return rows


def record_key(row, name):
    for field in ("image_id", "query_id", "source_group_id"):
        require(
            isinstance(row.get(field), str) and bool(row[field]),
            f"{name}_INVALID_{field}",
        )
    return row["image_id"], row["query_id"]


def index_rows(rows, name):
    result = {}
    for row in rows:
        key = record_key(row, name)
        require(key not in result, f"{name}_DUPLICATE_{key}")
        result[key] = row
    return result


def safe_rate(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(items, arm):
    absent = [(p, y) for p, y in items if y == 0]
    present = [(p, y) for p, y in items if y > 0]

    def forced(prediction):
        return prediction["arms"][arm]["forced_count"]

    report = {
        "n": len(items),
        "n_absent": len(absent),
        "n_present": len(present),
        "overall_mae": safe_rate(
            sum(abs(forced(p) - y) for p, y in items),
            len(items),
        ),
        "present_mae": safe_rate(
            sum(abs(forced(p) - y) for p, y in present),
            len(present),
        ),
        "aza": safe_rate(
            sum(forced(p) == 0 for p, _ in absent),
            len(absent),
        ),
        "fzr_plus": safe_rate(
            sum(forced(p) == 0 for p, _ in present),
            len(present),
        ),
        "false_zero_positive_queries": sum(
            forced(p) == 0 for p, _ in present
        ),
    }

    if arm == "A5":
        accepted = [
            (p, y) for p, y in items
            if p["arms"]["A5"]["status"] == "PROVISIONAL_POSITIVE"
        ]
        accepted_positive = [
            (p, y) for p, y in accepted if y > 0
        ]
        report.update({
            "coverage": safe_rate(len(accepted), len(items)),
            "abstention_rate": safe_rate(
                len(items) - len(accepted), len(items)
            ),
            "selective_present_count": len(accepted_positive),
            "selective_mae_present": safe_rate(
                sum(
                    abs(p["arms"]["A5"]["selective_count"] - y)
                    for p, y in accepted_positive
                ),
                len(accepted_positive),
            ),
        })

    if arm == "A4":
        report["non_exact_search_queries"] = sum(
            p["arms"]["A4"]["search_exact"] is False
            for p, _ in items
        )

    return report


def paired_interval(items, left, right, metric, draws=1000):
    groups = defaultdict(list)
    for prediction, gt in items:
        groups[prediction["source_group_id"]].append((prediction, gt))

    names = sorted(groups)

    def difference(sample):
        x = metrics(sample, left)[metric]
        y = metrics(sample, right)[metric]
        return None if x is None or y is None else x - y

    estimate = difference(items)

    if estimate is None:
        return {
            "estimate": None,
            "interval_95": None,
            "reason": "METRIC_STRATUM_ABSENT",
        }

    if len(names) < 10:
        return {
            "estimate": estimate,
            "interval_95": None,
            "cluster_count": len(names),
            "reason": "FEWER_THAN_10_SOURCE_CLUSTERS",
        }

    rng = random.Random(19011)
    values = []

    for _ in range(draws):
        sampled_groups = (
            names[rng.randrange(len(names))]
            for _ in names
        )
        sample = [
            item
            for group in sampled_groups
            for item in groups[group]
        ]
        delta = difference(sample)
        if delta is not None:
            values.append(delta)

    if len(values) < int(draws * 0.8):
        return {
            "estimate": estimate,
            "interval_95": None,
            "valid_draws": len(values),
            "reason": "BOOTSTRAP_STRATUM_UNSTABLE",
        }

    values.sort()
    return {
        "estimate": estimate,
        "interval_95": [
            values[int(0.025 * len(values))],
            values[int(0.975 * len(values)) - 1],
        ],
        "resample_unit": "source_group_id",
        "cluster_count": len(names),
        "valid_draws": len(values),
        "interpretation": "DESCRIPTIVE_NOT_CONFIRMATORY",
    }


def run(args):
    paths = (
        args.predictions,
        args.labels,
        args.roster,
        args.attestation,
        args.out,
    )

    require(
        len({p.resolve() for p in paths}) == len(paths),
        "INPUT_OUTPUT_PATH_COLLISION",
    )
    require(not args.out.exists(), "OUTPUT_ALREADY_EXISTS")

    for path in paths[:-1]:
        require(path.is_file(), f"INPUT_MISSING_{path}")

    prediction_sha = sha(args.predictions)
    label_sha = sha(args.labels)
    roster_sha = sha(args.roster)

    approval = json.loads(
        args.attestation.read_text(encoding="utf-8")
    )
    require(isinstance(approval, dict), "ATTESTATION_NOT_OBJECT")
    require(
        approval.get("status") == "approved_for_dev_scoring",
        "C1_C2_APPROVAL_MISSING",
    )

    for field in (
        "source_disjoint_verified",
        "candidate_ids_attested",
        "source_image_hashes_attested",
    ):
        require(
            approval.get(field) is True,
            "ATTESTATION_MISSING_" + field,
        )

    for field, expected in (
        ("prediction_file_sha256", prediction_sha),
        ("label_file_sha256", label_sha),
        ("roster_file_sha256", roster_sha),
        ("t12_protocol_sha256", PROTOCOL_SHA),
    ):
        require(
            approval.get(field) == expected,
            "ATTESTATION_HASH_MISMATCH_" + field,
        )

    split = approval.get("split_protocol_id")
    pool = approval.get("frozen_pool_run_id")

    require(
        isinstance(split, str) and bool(split),
        "SPLIT_ID_UNAPPROVED",
    )
    require(
        isinstance(pool, str) and bool(pool),
        "POOL_RUN_UNAPPROVED",
    )

    predictions = index_rows(
        load_rows(args.predictions, "PREDICTIONS"),
        "PREDICTIONS",
    )
    labels = index_rows(
        load_rows(args.labels, "LABELS"),
        "LABELS",
    )
    roster = index_rows(
        load_rows(args.roster, "ROSTER"),
        "ROSTER",
    )

    group_roles = defaultdict(set)

    for row in roster.values():
        role = row.get("cohort_role")
        require(
            role in {"dev", "cal", "final"},
            "ROSTER_ROLE_INVALID",
        )
        require(
            row.get("split_protocol_id") == split,
            "ROSTER_SPLIT_ID_MISMATCH",
        )
        group_roles[row["source_group_id"]].add(role)

    require(
        all(len(roles) == 1 for roles in group_roles.values()),
        "SOURCE_GROUP_CROSSES_SPLITS",
    )

    dev_keys = {
        key
        for key, row in roster.items()
        if row["cohort_role"] == "dev"
    }

    require(
        set(predictions) == set(labels) == dev_keys,
        "DEV_COHORT_JOIN_NOT_EXACT",
    )

    items = []
    exact_label_keys = {
        "image_id",
        "query_id",
        "source_group_id",
        "cohort_role",
        "split_protocol_id",
        "gt_count",
    }

    for key in sorted(dev_keys):
        prediction = predictions[key]
        label = labels[key]
        roster_row = roster[key]

        require(
            prediction.get("cohort_role") == "dev"
            and prediction.get("pool_scope") == "full_query_pool",
            "PREDICTION_NOT_CLEAN_DEV",
        )
        require(
            prediction.get("split_protocol_id") == split
            and prediction.get("frozen_pool_run_id") == pool,
            "PREDICTION_RUN_ID_MISMATCH",
        )
        require(
            prediction.get("t12_protocol_sha256") == PROTOCOL_SHA,
            "PREDICTION_PROTOCOL_HASH_MISMATCH",
        )
        require(
            prediction.get("gt_used") is False
            and prediction.get("scene_zero_certified") is False,
            "PREDICTION_GT_OR_ZERO_FLAG_INVALID",
        )
        require(
            prediction.get("model_calls_in_this_replay") == 0,
            "REPLAY_MODEL_CALLS_NONZERO",
        )

        require(
            prediction["source_group_id"]
            == label["source_group_id"]
            == roster_row["source_group_id"],
            "SOURCE_GROUP_JOIN_MISMATCH",
        )
        require(
            label.get("cohort_role") == "dev"
            and label.get("split_protocol_id") == split,
            "LABEL_SPLIT_MISMATCH",
        )
        require(
            set(label) == exact_label_keys,
            "LABEL_SCHEMA_EXTRA_FIELDS",
        )

        gt = label["gt_count"]
        require(
            type(gt) is int and gt >= 0,
            "GT_COUNT_INVALID",
        )

        arms = prediction.get("arms")
        require(
            isinstance(arms, dict)
            and set(arms) == set(ARMS),
            "PREDICTION_ARM_SET_MISMATCH",
        )

        for arm in ARMS:
            forced = arms[arm].get("forced_count")
            require(
                type(forced) is int and forced >= 0,
                "FORCED_COUNT_INVALID_" + arm,
            )

        require(
            arms["A0"]["forced_count"]
            == arms["A5"]["original_fallback_count"],
            "FALLBACK_CHANGED",
        )
        require(
            arms["A1"]["forced_count"]
            == arms["A3"]["forced_count"],
            "A1_A3_PARITY_FAILED",
        )
        require(
            type(arms["A4"].get("search_exact")) is bool,
            "A4_SEARCH_EXACT_STATUS_MISSING",
        )

        strict = arms["A5"]
        require(
            strict.get("status") in {
                "ABSTAIN", "PROVISIONAL_POSITIVE"
            },
            "A5_STATUS_INVALID",
        )

        if strict["status"] == "ABSTAIN":
            require(
                strict.get("selective_count") is None
                and strict["forced_count"]
                == arms["A0"]["forced_count"],
                "ABSTAIN_FALLBACK_MISMATCH",
            )
        else:
            require(
                type(strict.get("selective_count")) is int
                and strict["selective_count"] > 0
                and strict["forced_count"] == strict["selective_count"],
                "POSITIVE_COUNT_INVALID",
            )

        items.append((prediction, gt))

    report = {
        "status": "T12_OFFLINE_DEV_SCORE_DESCRIPTIVE_ONLY",
        "count": len(items),
        "source_clusters": len({
            p["source_group_id"] for p, _ in items
        }),
        "gt_supplied_in_separate_file": True,
        "selection_policy": "LOCKED_A0_A1_A3_A4_A5_FORCED_COUNTS",
        "approval_file_sha256": sha(args.attestation),
        "prediction_file_sha256": prediction_sha,
        "label_file_sha256": label_sha,
        "roster_file_sha256": roster_sha,
        "t12_protocol_sha256": PROTOCOL_SHA,
        "arms": {
            arm: metrics(items, arm)
            for arm in ARMS
        },
        "paired_deltas": {
            f"{left}_minus_{right}": {
                metric: paired_interval(items, left, right, metric)
                for metric in (
                    "overall_mae",
                    "present_mae",
                    "aza",
                    "fzr_plus",
                )
            }
            for left, right in (
                ("A4", "A0"),
                ("A4", "A1"),
                ("A5", "A0"),
            )
        },
        "limitations": [
            "DEV-only descriptive evaluation",
            "Approval metadata requires independent C1/C2 verification",
            "Not a confirmatory statistical test",
            "A6/A7 measured-cost evaluation not included",
            "No claim of certified scene-level zero",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")

    print(json.dumps({
        "status": report["status"],
        "queries": len(items),
        "source_clusters": report["source_clusters"],
        "output": str(args.out),
        "output_sha256": sha(args.out),
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
