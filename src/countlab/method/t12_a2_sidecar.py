"""GT-blind T12 A2 comparator sidecar without modifying the frozen exporter."""

import argparse
import json
from pathlib import Path

from countlab.method.t12_a2_simple_identity import run_a2_simple_identity
from countlab.method.t12_paired_export import (
    digest_bytes,
    evaluate_record,
    freeze_check,
)


def evaluate_a2_sidecar(row, line, mode):
    base = evaluate_record(row, line, mode)
    rejected_items = base["arms"]["A1"]["rejected"]

    if not isinstance(rejected_items, list):
        raise ValueError(f"line {line}: malformed A1 rejection ledger")

    rejected_ids = []
    for item in rejected_items:
        if not isinstance(item, dict) or not isinstance(item.get("candidate_id"), str):
            raise ValueError(f"line {line}: malformed A1 rejected candidate")
        rejected_ids.append(item["candidate_id"])

    result = run_a2_simple_identity(
        row["manifest"],
        row["observed_events"],
        a1_rejected_ids=rejected_ids,
    )

    if result["a1_retained_baseline_count"] != base["arms"]["A1"]["forced_count"]:
        raise AssertionError(f"line {line}: A1/A2 survivor count mismatch")

    return {
        "schema": "countlab_v2_t12_a2_sidecar_v1",
        "research_status": "EXPERIMENTAL_NOT_TEAM_APPROVED",
        "image_id": base["image_id"],
        "query_id": base["query_id"],
        "source_group_id": base["source_group_id"],
        "cohort_role": base["cohort_role"],
        "split_protocol_id": base["split_protocol_id"],
        "frozen_pool_run_id": base["frozen_pool_run_id"],
        "pool_scope": base["pool_scope"],
        "manifest_sha256": base["manifest_sha256"],
        "observed_events_sha256": base["observed_events_sha256"],
        "verifier_log_id": base["verifier_log_id"],
        "observed_event_fields": base["observed_event_fields"],
        "a0_frozen_count": base["arms"]["A0"]["forced_count"],
        "a1_rejection_only_count": base["arms"]["A1"]["forced_count"],
        "a1_rejected_candidate_ids": sorted(rejected_ids),
        "A2_NMS": {
            "forced_count": result["A2_NMS"]["count"],
            "details": result["A2_NMS"],
        },
        "A2_COMPLETE_LINK": {
            "forced_count": result["A2_COMPLETE_LINK"]["count"],
            "details": result["A2_COMPLETE_LINK"],
        },
        "a1_a3_parity": "PASS",
        "scene_zero_certified": False,
        "new_model_calls": 0,
        "gt_used": False,
        "accuracy_evaluated": False,
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
        raise SystemExit("STOP: input missing")

    protocol_sha = freeze_check()
    records = []
    seen = set()

    for line, raw in enumerate(
        args.input.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip():
            continue

        row = json.loads(raw)
        rec = evaluate_a2_sidecar(row, line, args.mode)
        pair = (rec["image_id"], rec["query_id"])

        if pair in seen:
            raise ValueError(f"line {line}: duplicate image/query: {pair}")

        seen.add(pair)
        rec["t12_protocol_sha256"] = protocol_sha
        rec["a2_source_sha256"] = digest_bytes(
            Path(run_a2_simple_identity.__code__.co_filename).read_bytes()
        )
        rec["sidecar_exporter_sha256"] = digest_bytes(
            Path(__file__).read_bytes()
        )
        records.append(rec)

    if not records:
        raise ValueError("empty cohort")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("x", encoding="utf-8") as handle:
        for rec in records:
            handle.write(
                json.dumps(
                    rec,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                ) + "\n"
            )

    print(json.dumps({
        "status": "T12_A2_SIDECAR_GT_BLIND_COMPLETE",
        "mode": args.mode,
        "queries": len(records),
        "arms": ["A2_NMS", "A2_COMPLETE_LINK"],
        "a1_a3_parity": "PASS",
        "output": str(args.out),
        "output_sha256": digest_bytes(args.out.read_bytes()),
        "t12_protocol_sha256": protocol_sha,
        "gt_used": False,
        "new_model_calls": 0,
        "scientific_accuracy_evaluated": False,
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
