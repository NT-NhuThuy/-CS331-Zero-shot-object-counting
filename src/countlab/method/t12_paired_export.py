"""GT-blind fixed-evidence paired prediction exporter for UNICOUNT v2 C3."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from countlab.method.c3_dual_channel import solve_c3_dual_channel
from countlab.method.joint_solver_v2 import validate_event_contract
from countlab.method.v14_rejection_shadow import run_v14_rejection_shadow

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/thuy/T12_C3_ABLATION_PROTOCOL_20261119.md"

FROZEN_MODULES = (
    "src/countlab/method/joint_solver_v2.py",
    "src/countlab/method/v14_rejection_shadow.py",
    "src/countlab/method/c3_dual_channel.py",
    "src/countlab/method/c3_rescue_arbitration.py",
    "src/countlab/method/c3_evidence_refinement.py",
    "src/countlab/method/c3_count_impact_hints.py",
)

FORBIDDEN_GT_KEYS = {
    "gt", "gt_count", "ground_truth", "groundtruth",
    "target_count", "true_count", "reference_count",
    "gt_boxes", "gt_points", "gt_annotations", "annotations",
    "is_absent", "is_present", "count_label",
    "presence_label", "gold_count", "gold_labels",
}

ROW_KEYS = {
    "manifest", "observed_events", "source_group_id",
    "cohort_role", "split_protocol_id", "frozen_pool_run_id",
    "verifier_log_id", "pool_scope",
}


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return digest_bytes(payload)


def check_gt_free(value, path="input"):
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_GT_KEYS:
                raise ValueError(f"GT-bearing key forbidden: {path}.{key}")
            check_gt_free(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_gt_free(child, f"{path}[{index}]")


def require_text(row, field, line):
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line}: missing nonempty {field}")
    return value


def freeze_check():
    if not PROTOCOL.is_file():
        raise ValueError(f"T12 protocol missing: {PROTOCOL}")

    source = PROTOCOL.read_text(encoding="utf-8")
    recorded = dict(re.findall(
        r"\|\s*`(src/countlab/method/[^`]+)`\s*\|\s*`([0-9a-f]{64})`",
        source,
    ))

    for relative in FROZEN_MODULES:
        if relative not in recorded:
            raise ValueError(f"T12 protocol lacks hash: {relative}")

        path = ROOT / relative
        if (
            not path.is_file()
            or digest_bytes(path.read_bytes()) != recorded[relative]
        ):
            raise ValueError(
                f"source differs from T12 frozen snapshot: {relative}"
            )

    return digest_bytes(PROTOCOL.read_bytes())


def evaluate_record(row, line, mode):
    if not isinstance(row, dict) or not set(row).issubset(ROW_KEYS):
        raise ValueError(f"line {line}: unexpected input fields")

    check_gt_free(row)

    group = require_text(row, "source_group_id", line)
    cohort = require_text(row, "cohort_role", line)
    split_id = require_text(row, "split_protocol_id", line)
    pool_run = require_text(row, "frozen_pool_run_id", line)
    pool_scope = require_text(row, "pool_scope", line)

    if pool_scope not in (
        "full_query_pool", "selected_task_subset", "unverified"
    ):
        raise ValueError(f"line {line}: invalid pool_scope")

    manifest = row.get("manifest")
    events = row.get("observed_events")

    if not isinstance(manifest, dict) or not isinstance(events, list):
        raise ValueError(f"line {line}: invalid manifest/events")

    for field in ("image_id", "query_id", "query_text"):
        if field not in manifest or not str(manifest[field]).strip():
            raise ValueError(f"line {line}: missing manifest.{field}")

    baseline = manifest.get("baseline")
    if not isinstance(baseline, list):
        raise ValueError(f"line {line}: invalid baseline")

    count = manifest.get("baseline_count")
    if type(count) is not int or count != len(baseline):
        raise ValueError(
            f"line {line}: baseline count differs from proposals"
        )

    validate_event_contract(events)

    if mode == "dev_eval":
        metadata = manifest.get("metadata")

        if (
            cohort != "dev"
            or pool_scope != "full_query_pool"
            or split_id.lower() in (
                "unknown", "unverified", "synthetic"
            )
        ):
            raise ValueError(f"line {line}: invalid DEV declaration")

        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("image_sha256"), str)
            or not re.fullmatch(
                r"[0-9a-f]{64}", metadata["image_sha256"]
            )
        ):
            raise ValueError(f"line {line}: image hash missing")

        if not metadata.get("backend_protocol"):
            raise ValueError(f"line {line}: backend protocol missing")

        if events and not row.get("verifier_log_id"):
            raise ValueError(
                f"line {line}: verifier log linkage missing"
            )

    elif cohort not in (
        "synthetic", "diagnostic_exposed", "pilot_exposed"
    ):
        raise ValueError(
            f"line {line}: diagnostic mode excludes DEV/CAL/FINAL"
        )

    a1 = run_v14_rejection_shadow(
        manifest, events, fallback_count=count
    )

    a3 = solve_c3_dual_channel(
        manifest,
        events,
        fallback_count=count,
        rescue_policy="rejection_only",
    )

    a4 = solve_c3_dual_channel(
        manifest,
        events,
        fallback_count=count,
        rescue_policy="guarded_identity",
    )

    strict = a4["strict_decision"]

    if (
        a1["operational_count"]
        != a3["operational_prediction"]["count"]
    ):
        raise AssertionError(f"line {line}: A1/A3 parity failed")

    if a3["strict_decision"] != strict:
        raise AssertionError(
            f"line {line}: strict solver changed between arms"
        )

    if strict["original_fallback_count"] != count:
        raise AssertionError(
            f"line {line}: frozen fallback was modified"
        )

    if strict["status"] not in (
        "ABSTAIN", "PROVISIONAL_POSITIVE"
    ):
        raise AssertionError(f"line {line}: invalid strict status")

    if strict["status"] == "ABSTAIN":
        if (
            strict["count"] is not None
            or strict["evaluation_fallback_count"] != count
        ):
            raise AssertionError(
                f"line {line}: invalid abstention fallback"
            )

    if strict["status"] == "PROVISIONAL_POSITIVE":
        if strict["count"] != strict["evaluation_fallback_count"]:
            raise AssertionError(
                f"line {line}: invalid positive forced count"
            )

    arbitration = a4["correction_ledger"]["identity_arbitration"]
    if arbitration is None:
        raise AssertionError(
            f"line {line}: missing identity arbitration"
        )

    return {
        "image_id": str(manifest["image_id"]),
        "query_id": str(manifest["query_id"]),
        "source_group_id": group,
        "cohort_role": cohort,
        "split_protocol_id": split_id,
        "frozen_pool_run_id": pool_run,
        "pool_scope": pool_scope,
        "verifier_log_id": row.get("verifier_log_id"),
        "manifest_sha256": digest(manifest),
        "observed_events_sha256": digest(events),
        "observed_event_fields": len(events),
        "model_calls_in_this_replay": 0,
        "measured_verifier_cost_attached": False,
        "arms": {
            "A0": {
                "forced_count": count,
                "status": "FROZEN_BASELINE",
            },
            "A1": {
                "forced_count": a1["operational_count"],
                "status": "OPERATIONAL_REJECTION_ONLY",
                "rejected": a1["rejected"],
            },
            "A3": {
                "forced_count": a3["operational_prediction"]["count"],
                "status": "M7_OPERATIONAL_REJECTION_ONLY",
            },
            "A4": {
                "forced_count": a4["operational_prediction"]["count"],
                "status": "M8_M10_GUARDED_OPERATIONAL",
                "search_exact": arbitration["search_exact"],
                "conflicts": arbitration["conflicts"],
                "rejection_count": len(
                    arbitration["effective_rejected"]
                ),
                "baseline_merge_reduction": arbitration[
                    "baseline_identity_merge_reduction"
                ],
                "rescue_additions": len(
                    arbitration["admitted_rescue_groups"]
                ),
                "correction_ledger": a4["correction_ledger"],
            },
            "A5": {
                "forced_count": strict["evaluation_fallback_count"],
                "status": strict["status"],
                "selective_count": strict["count"],
                "original_fallback_count": count,
                "blocking_issues": strict["blocking_issues"],
                "physical_instances": strict["physical_instances"],
            },
        },
        "not_executed_arms": ["A2_B5", "A6_B6", "A7_B7"],
        "scene_zero_certified": False,
        "gt_used": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("diagnostic", "dev_eval"),
        default="diagnostic",
    )
    args = parser.parse_args()

    if args.input.resolve() == args.out.resolve() or args.out.exists():
        raise SystemExit("STOP: output exists or equals input")

    if not args.input.is_file():
        raise SystemExit("STOP: input does not exist")

    frozen_sha = freeze_check()
    results = []
    seen = set()

    lines = args.input.read_text(encoding="utf-8").splitlines()

    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue

        row = json.loads(raw)
        record = evaluate_record(row, line_number, args.mode)

        key = (record["image_id"], record["query_id"])
        if key in seen:
            raise ValueError(
                f"line {line_number}: duplicate image/query {key}"
            )

        seen.add(key)
        record["t12_protocol_sha256"] = frozen_sha
        record["exporter_sha256"] = digest_bytes(
            Path(__file__).read_bytes()
        )
        results.append(record)

    if not results:
        raise ValueError("empty input cohort")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("x", encoding="utf-8") as handle:
        for record in results:
            handle.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ) + "\n"
            )

    print(json.dumps({
        "status": "T12_GT_BLIND_PAIRED_EXPORT_COMPLETE",
        "mode": args.mode,
        "queries": len(results),
        "arms": ["A0", "A1", "A3", "A4", "A5"],
        "a1_a3_parity": "PASS",
        "output": str(args.out),
        "output_sha256": digest_bytes(args.out.read_bytes()),
        "protocol_sha256": frozen_sha,
        "gt_used": False,
        "model_calls": 0,
        "mae_computed": False,
    }, sort_keys=True, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
