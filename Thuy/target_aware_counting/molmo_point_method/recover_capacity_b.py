"""Recover only truncated B rows with the official short pointing prompt.

The complete first-pass rows are copied bit-for-bit in meaning into a new
contract with explicit lineage. Retry rows use the same image/warp policy and
token cap. No partial points become final predictions.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .native_greedy import COHORTS, load_existing, load_manifest, sha256
from .run_capacity_b_cohort import infer_item, load_audit
from .snapshot_lock import verify


def first_pass_rows(path: Path, cohort: str, items: list[dict], audit_hash: str,
                    revision: str, token_cap: int, dtype: str) -> tuple[dict, dict]:
    rows = {}
    contract = None
    expected = {item["item_id"]: item for item in items}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            item_id = row["item_id"]
            if item_id not in expected or item_id in rows:
                raise ValueError(f"foreign or duplicate first-pass row: {item_id}")
            if row["source_image_id"] != expected[item_id]["source_image_id"] or \
                    row["query"] != expected[item_id]["query"]:
                raise ValueError(f"first-pass identity mismatch: {item_id}")
            if contract is None:
                contract = row["contract"]
            if row["contract"] != contract or row["status"] not in ("complete", "truncated"):
                raise ValueError(f"mixed contract or unexpected status: {item_id}")
            if row["status"] == "complete" and row["count"] != len(row["points_xy"]):
                raise ValueError(f"first-pass count mismatch: {item_id}")
            if row["status"] == "truncated" and row["count"] is not None:
                raise ValueError(f"truncated row has a count: {item_id}")
            rows[item_id] = row
    if rows.keys() != expected.keys() or contract is None or \
            contract["method"] != "molmopoint_capacity_b_v1" or \
            contract["cohort"] != cohort or contract["audit_sha256"] != audit_hash or \
            contract["revision"] != revision or contract["max_new_tokens"] != token_cap or \
            contract["load_dtype"] != dtype or contract["item_id_filter"] is not None:
        raise ValueError("first-pass file does not match full locked B policy")
    return contract, rows


def run(args):
    manifest, items = load_manifest(args.cohort)
    snapshot = args.snapshot.resolve()
    verify(snapshot, args.revision)
    audit_path = args.audit.resolve()
    audit_contract, audit = load_audit(audit_path, args.cohort, items,
                                       snapshot, args.revision, sha256(manifest))
    source_path = args.first_pass.resolve()
    source_contract, source = first_pass_rows(
        source_path, args.cohort, items, sha256(audit_path), args.revision,
        args.max_new_tokens, args.dtype)
    retry_ids = [item["item_id"] for item in items if source[item["item_id"]]["status"] == "truncated"]
    contract = {"method": "molmopoint_capacity_b_v2_retry_short_prompt",
                "cohort": args.cohort, "manifest_sha256": sha256(manifest),
                "first_pass_sha256": sha256(source_path),
                "first_pass_contract": source_contract,
                "audit_sha256": sha256(audit_path), "audit_geometry": audit_contract["geometry"],
                "snapshot": str(snapshot), "revision": args.revision,
                "snapshot_lock_sha256": sha256(snapshot / "molmo_point_snapshot_lock.json"),
                "max_new_tokens": args.max_new_tokens, "load_dtype": args.dtype,
                "retry_prompt_template": "Point to the {query_without_trailing_period}.",
                "retry_item_ids": retry_ids,
                "retry_policy": "retry only first-pass truncated rows; preserve complete rows"}
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError("output exists; pass --resume with the same contract")
    existing = load_existing(output, contract, {item["item_id"] for item in items})
    # The complete rows can be copied without model loading. Preserve manifest order.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        for item in items:
            item_id = item["item_id"]
            if item_id in existing or source[item_id]["status"] != "complete":
                continue
            row = dict(source[item_id])
            row["contract"] = contract
            row["generation_source"] = "first_pass_complete"
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            existing[item_id] = row
    remaining = [item for item in items if item["item_id"] in retry_ids and
                 item["item_id"] not in existing]
    if remaining and not args.prepare_only:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
        model = AutoModelForImageTextToText.from_pretrained(
            snapshot, trust_remote_code=True, local_files_only=True, dtype=dtype,
            device_map=args.device_map, offload_folder=args.offload_folder,
            offload_state_dict=True).eval()
        meta_parameters = [name for name, parameter in model.named_parameters()
                           if parameter.device.type == "meta"]
        if meta_parameters:
            raise RuntimeError(
                "MolmoPoint cannot run with parameters left on the meta device. "
                "Free enough VRAM and use --device-map cuda:0; first meta parameter: "
                f"{meta_parameters[0]}")
        processor = AutoProcessor.from_pretrained(
            snapshot, trust_remote_code=True, local_files_only=True,
            padding_side="left", use_fast=False)
        with output.open("a", encoding="utf-8") as stream:
            for item in remaining:
                result = infer_item(model, processor, item, audit[item["item_id"]],
                                    args.max_new_tokens, prompt_style="official_short")
                row = {"item_id": item["item_id"],
                       "source_image_id": item["source_image_id"],
                       "query": item["query"], "contract": contract,
                       "generation_source": "short_prompt_retry", **result}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                existing[item["item_id"]] = row
                print(f"retry {item['item_id']} {result['status']} "
                      f"{result['selection']}", flush=True)
    print(json.dumps({"rows": len(existing), "denominator": len(items),
                      "complete": sum(r["status"] == "complete" for r in existing.values()),
                      "truncated": sum(r["status"] == "truncated" for r in existing.values()),
                      "retries": len(retry_ids), "output": str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--first-pass", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--device-map", default="cuda:0",
        help="single-device placement; auto CPU/meta dispatch is unsupported by MolmoPoint")
    parser.add_argument("--offload-folder")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="copy the 92 complete rows without loading model weights")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    run(args)


if __name__ == "__main__":
    main()
