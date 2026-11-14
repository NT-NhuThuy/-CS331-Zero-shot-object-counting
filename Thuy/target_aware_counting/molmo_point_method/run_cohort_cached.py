"""Resumable GT-free MolmoPoint A run on a locked development cohort.

This runner is gated on a real-checkpoint equality check against the slow
reference beam. It never reads labels. Each completed item is fsynced before
the next one; failed items are logged separately and remain absent on resume.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .f1_sidecar import load_f1_sidecar
from .native_greedy import COHORTS, infer_one, load_existing, load_manifest, prepare_one, sha256
from .proposal_mapping import proposal_patch_indices
from .reference_search import spatial_search_cached
from .run_reference import decode_path, inference_autocast, summarize_decoded_paths
from .snapshot_lock import verify


def run_one(model, processor, item: dict, args) -> dict:
    native = infer_one(model, processor, item, args.max_new_tokens)
    if native["status"] != "complete":
        raise ValueError(f"native generation {native['status']}; choose a sufficient token limit")
    sidecar, candidates, available = load_f1_sidecar(args.cohort, item)
    inputs, metadata, image_size = prepare_one(processor, item)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    mapping = (proposal_patch_indices(candidates, metadata,
                                      max_candidates=args.max_candidates)
               if available else None)
    guided = tuple(mapping["ordered_patch_indices"][:args.guided_slots]) \
        if mapping is not None else ()
    dtype = next(model.parameters()).dtype
    with torch.inference_mode(), inference_autocast(device, dtype):
        proper, proper_stats = spatial_search_cached(
            model, inputs, beam_width=args.beam_width,
            max_new_tokens=args.max_new_tokens, max_language=args.max_language,
            max_spatial=args.max_spatial,
            tail_mass_epsilon=args.tail_mass_epsilon)
        if available:
            guided_paths, guided_stats = spatial_search_cached(
                model, inputs, beam_width=args.beam_width,
                max_new_tokens=args.max_new_tokens, max_language=args.max_language,
                max_spatial=args.max_spatial, guided_patch_indices=guided,
                tail_mass_epsilon=args.tail_mass_epsilon)
        else:
            guided_paths, guided_stats = [], None
    proper_decoded = [decode_path(model, processor, metadata, image_size, path)
                      for path in proper]
    proper_result = summarize_decoded_paths(
        proper, proper_decoded, proper_stats, args.gospa_cutoff_pixels)
    if available:
        guided_decoded = [decode_path(model, processor, metadata, image_size, path)
                          for path in guided_paths]
        guided_result = summarize_decoded_paths(
            guided_paths, guided_decoded, guided_stats, args.gospa_cutoff_pixels)
    else:
        guided_result = None
    if proper_result["decision"] is None or \
            (available and guided_result["decision"] is None):
        raise ValueError("no valid complete hypothesis for a required search branch")
    return {"native": native, "proper_beam": proper_result,
            "guided_beam": guided_result, "f1_raw_available": available,
            "f1_sidecar_sha256": sha256(sidecar),
            "proposal_mapping": mapping,
            "guided_patch_indices_used": guided}


def run(args) -> None:
    if args.guided_slots > args.max_spatial:
        raise ValueError("guided slots exceed spatial option budget")
    snapshot = Path(args.snapshot).resolve()
    lock = verify(snapshot, args.revision)
    manifest, items = load_manifest(args.cohort)
    contract = {"method": "molmopoint_a_cached_beam_v1", "cohort": args.cohort,
                "manifest_sha256": sha256(manifest), "snapshot": str(snapshot),
                "revision": lock["revision"], "dtype": args.dtype,
                "max_new_tokens": args.max_new_tokens, "beam_width": args.beam_width,
                "max_language": args.max_language, "max_spatial": args.max_spatial,
                "guided_slots": args.guided_slots, "max_candidates": args.max_candidates,
                "tail_mass_epsilon": args.tail_mass_epsilon,
                "gospa_cutoff_pixels": args.gospa_cutoff_pixels,
                "code_sha256": {name: sha256(Path(__file__).with_name(name))
                                for name in ("run_cohort_cached.py", "reference_replay.py",
                                             "reference_search.py", "head_scoring.py",
                                             "run_reference.py", "set_risk.py")}}
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError("output exists; pass --resume for same contract")
    output.parent.mkdir(parents=True, exist_ok=True)
    # A resumed file retains the original frozen contract. Record the current
    # runner hash on new rows so a code change cannot masquerade as the old run.
    current_code_sha256 = contract["code_sha256"]
    if args.resume and output.exists() and output.stat().st_size:
        with output.open(encoding="utf-8") as prior:
            original_contract = json.loads(prior.readline())["contract"]
        if {k: v for k, v in original_contract.items() if k != "code_sha256"} != \
                {k: v for k, v in contract.items() if k != "code_sha256"}:
            raise ValueError("existing output has a different run configuration")
        contract = original_contract
    existing = load_existing(output, contract, {item["item_id"] for item in items})
    skipped = set(args.skip_item_id)
    unknown_skips = skipped - {item["item_id"] for item in items}
    if unknown_skips:
        raise ValueError(f"skip item IDs outside cohort: {sorted(unknown_skips)}")
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True, dtype=dtype,
        device_map=args.device_map, offload_folder=args.offload_folder,
        offload_state_dict=True).eval()
    processor = AutoProcessor.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True, padding_side="left")
    error_path = output.with_suffix(output.suffix + ".errors.jsonl")
    with output.open("a", encoding="utf-8") as stream, \
            error_path.open("a", encoding="utf-8") as errors:
        for index, item in enumerate(items, 1):
            if item["item_id"] in existing:
                continue
            if item["item_id"] in skipped:
                print(f"{index}/{len(items)} {item['item_id']} SKIPPED explicit diagnostic exception",
                      flush=True)
                continue
            try:
                result = run_one(model, processor, item, args)
            except Exception as exc:
                errors.write(json.dumps({"item_id": item["item_id"],
                                         "error_type": type(exc).__name__,
                                         "error": str(exc)}) + "\n")
                errors.flush()
                os.fsync(errors.fileno())
                print(f"{index}/{len(items)} {item['item_id']} ERROR {type(exc).__name__}: {exc}",
                      flush=True)
                if args.fail_fast:
                    raise
                continue
            row = {"item_id": item["item_id"],
                   "source_image_id": item["source_image_id"],
                   "query": item["query"], "contract": contract,
                   "execution_code_sha256": current_code_sha256, **result}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            existing[item["item_id"]] = row
            print(f"{index}/{len(items)} {item['item_id']} complete "
                  f"native={result['native']['count']} "
                  f"proper={len(result['proper_beam']['decision']['mbr_points_xy'])} "
                  f"guided={len(result['guided_beam']['decision']['mbr_points_xy']) if result['guided_beam'] else 'unavailable'}",
                  flush=True)
    print(json.dumps({"cohort": args.cohort, "denominator": len(items),
                      "completed_rows": len(existing), "output": str(output)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--max-language", type=int, default=4)
    parser.add_argument("--max-spatial", type=int, default=8)
    parser.add_argument("--guided-slots", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--gospa-cutoff-pixels", type=float, default=32.0)
    parser.add_argument("--tail-mass-epsilon", type=float, default=1e-6,
                        help="stop when open-prefix mass is this fraction of completed mass")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"),
                        default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--offload-folder")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-item-id", action="append", default=[],
                        help="diagnostic exception; leaves an explicit missing row")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.beam_width < 1 or args.max_language < 1 or \
            args.max_spatial < 1 or args.guided_slots < 0 or \
            not 0 < args.tail_mass_epsilon < 1:
        parser.error("invalid search limits")
    run(args)


if __name__ == "__main__":
    main()
