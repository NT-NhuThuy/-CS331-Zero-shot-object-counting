"""One-item, GT-free reference parity and spatial-search diagnostic.

This intentionally replays each branch from the prompt. It is a correctness
probe, not a throughput implementation for full development cohorts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .f1_sidecar import load_f1_sidecar
from .native_greedy import COHORTS, load_manifest, prepare_one, sha256, validate_points
from .point_grammar import validate_point_tokens
from .proposal_mapping import proposal_patch_indices
from .reference_search import native_replay, spatial_search, spatial_search_cached
from .score_audit import audit_native_path, representative_steps
from .set_risk import Hypothesis, decide
from .snapshot_lock import verify


def inference_autocast(device: torch.device, load_dtype):
    """Match the mixed-precision context expected by MolmoPoint's connector.

    The frozen model intentionally forms a float32 token-pooling mask.  Its
    product with ViT features is therefore float32, while reduced-precision
    connector weights are bf16/fp16.  CUDA autocast brings the connector input
    back to the loaded weight dtype, as required by the upstream model code.
    """
    enabled = device.type == "cuda" and load_dtype in (torch.bfloat16, torch.float16)
    amp_dtype = load_dtype if enabled else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enabled)


def decode_path(model, processor, metadata, image_size, path) -> dict:
    tokens = torch.tensor([path.tokens], device=model.device, dtype=torch.long)
    text = processor.post_process_image_text_to_text(
        tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    base = {"tokens": list(path.tokens), "generated_text": text,
            "log_probability": path.log_probability if math.isfinite(path.log_probability) else None,
            "complete": path.complete}
    bounds = model.model.build_token_bounds(
        torch.as_tensor(metadata["token_pooling"])[None, :, :].to(model.device))
    try:
        points = model.extract_image_points(
            text, metadata["token_pooling"], metadata["subpatch_mapping"],
            metadata["image_sizes"])
        xy = validate_points(points, image_size)
        grammar = validate_point_tokens(list(path.tokens), bounds, len(xy))
    except (ValueError, IndexError) as exc:
        # The native logit processor enforces spatial-token order, not the
        # surrounding XML-like text. A beam can reach EOS with a point triple
        # that the official text parser does not recognize. Keep its trace but
        # never turn a malformed completion into an empty-set prediction.
        return {**base, "valid_for_decision": False, "points_xy": None,
                "count": None, "decode_error_type": type(exc).__name__,
                "decode_error": str(exc)}
    return {**base, "valid_for_decision": True, "points_xy": xy,
            "count": len(xy), **grammar}


def summarize_decoded_paths(paths, decoded: list[dict], stats: dict,
                            gospa_cutoff_pixels: float) -> dict:
    if len(paths) != len(decoded):
        raise ValueError("path/decode count mismatch")
    valid = []
    for index, (path, row) in enumerate(zip(paths, decoded)):
        row["path_index"] = index
        if row["valid_for_decision"]:
            if not path.complete or row["log_probability"] is None:
                raise ValueError("decision requires complete paths with finite probability")
            valid.append(row)
    stats = {**stats, "valid_complete_sequences": len(valid),
             "invalid_complete_sequences": len(decoded)-len(valid)}
    if valid:
        hypotheses = [Hypothesis(tuple(tuple(point) for point in row["points_xy"]),
                                 row["log_probability"], str(row["path_index"]))
                      for row in valid]
        decision = decide(hypotheses, gospa_cutoff_pixels)
    else:
        decision = None
    return {"paths": decoded, "stats": stats, "decision": decision,
            "decision_status": "ready" if valid else "no_valid_complete_sequences"}


def run(args) -> dict:
    snapshot = Path(args.snapshot).resolve()
    lock = verify(snapshot, args.revision)
    manifest, rows = load_manifest(args.cohort)
    item = next((r for r in rows if r["item_id"] == args.item_id), None)
    if item is None:
        raise ValueError("item_id not in locked development cohort")
    sidecar, f1_candidates, raw_available = load_f1_sidecar(args.cohort, item)
    load_dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True,
        dtype=load_dtype, device_map=args.device_map,
        offload_folder=args.offload_folder,
        offload_state_dict=True).eval()
    processor = AutoProcessor.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True,
        padding_side="left")
    inputs, metadata, image_size = prepare_one(processor, item)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    context_limit = getattr(getattr(model.config, "text_config", None),
                            "max_position_embeddings", None)
    if context_limit is not None and inputs["input_ids"].shape[1] + args.max_new_tokens > context_limit:
        raise ValueError("prompt plus output limit exceeds model context")
    if args.audit_native_path is not None:
        with args.audit_native_path.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        if len(records) != 1:
            raise ValueError("score audit requires exactly one native prediction row")
        record = records[0]
        contract = record.get("contract", {})
        if record.get("item_id") != args.item_id or record.get("source_image_id") != item["source_image_id"] or \
                record.get("query") != item["query"] or record.get("status") != "complete" or \
                contract.get("cohort") != args.cohort or contract.get("revision") != args.revision or \
                contract.get("manifest_sha256") != sha256(manifest) or \
                contract.get("load_dtype") != args.dtype or \
                Path(contract.get("snapshot", "")).resolve() != snapshot:
            raise ValueError("native prediction identity, completion or model contract mismatch")
        native_tokens = record.get("output_token_ids")
        if not isinstance(native_tokens, list) or not native_tokens or \
                not all(isinstance(token, int) for token in native_tokens) or \
                len(native_tokens) > args.max_new_tokens:
            raise ValueError("invalid native token sequence for audit")
        bounds = model.model.build_token_bounds(inputs["image_token_pooling"])
        eos = model.generation_config.eos_token_id
        eos_ids = (eos,) if isinstance(eos, int) else tuple(eos or ())
        if native_tokens[-1] not in eos_ids:
            raise ValueError("native path must end with EOS")
        steps = representative_steps(native_tokens, bounds, eos_ids)
        with torch.inference_mode(), inference_autocast(device, load_dtype):
            audit = audit_native_path(model, inputs, native_tokens, selected_steps=steps)
        return {"item_id": args.item_id, "cohort": args.cohort,
                "manifest_sha256": sha256(manifest),
                "snapshot_revision": lock["revision"],
                "native_prediction_sha256": sha256(args.audit_native_path),
                "native_token_count": len(native_tokens), **audit}
    with torch.inference_mode(), inference_autocast(device, load_dtype):
        native_generated = model.generate(
            **inputs, logits_processor=model.build_logit_processor_from_inputs(inputs),
            max_new_tokens=args.max_new_tokens, do_sample=False)
        native_tokens = native_generated[0, inputs["input_ids"].shape[1]:].tolist()
        replay_path = native_replay(model, inputs, max_new_tokens=args.max_new_tokens)
    if native_tokens != list(replay_path.tokens):
        raise ValueError("native greedy parity failed; spatial search must not run")
    native_record = decode_path(model, processor, metadata, image_size, replay_path) \
        if replay_path.complete else {"complete": False, "tokens": native_tokens}
    if replay_path.complete and not native_record["valid_for_decision"]:
        raise ValueError("native greedy completed but its point decode is invalid")
    if args.parity_only:
        return {"item_id": args.item_id, "cohort": args.cohort,
                "manifest_sha256": sha256(manifest),
                "snapshot_revision": lock["revision"],
                "native_parity": True, "native": native_record,
                "max_new_tokens": args.max_new_tokens, "load_dtype": args.dtype,
                "parity_only": True}
    mapping = (proposal_patch_indices(f1_candidates, metadata,
                                      max_candidates=args.max_candidates)
               if raw_available else None)
    guided = tuple(mapping["ordered_patch_indices"][:args.guided_slots]) \
        if mapping is not None else ()
    search = spatial_search_cached if args.search_engine == "cached" else spatial_search
    with torch.inference_mode(), inference_autocast(device, load_dtype):
        proper, proper_stats = search(
            model, inputs, beam_width=args.beam_width, max_new_tokens=args.max_new_tokens,
            max_language=args.max_language, max_spatial=args.max_spatial)
        if raw_available:
            guided_paths, guided_stats = search(
                model, inputs, beam_width=args.beam_width, max_new_tokens=args.max_new_tokens,
                max_language=args.max_language, max_spatial=args.max_spatial,
                guided_patch_indices=guided)
        else:
            guided_paths, guided_stats = [], None
    decoded_proper = [decode_path(model, processor, metadata, image_size, path)
                      for path in proper]
    decoded_guided = [decode_path(model, processor, metadata, image_size, path)
                      for path in guided_paths]
    return {"item_id": args.item_id, "cohort": args.cohort,
            "manifest_sha256": sha256(manifest),
            "snapshot_revision": lock["revision"],
            "native_parity": True, "native": native_record,
            "proposal_mapping": mapping, "f1_raw_available": raw_available,
            "f1_sidecar_sha256": sha256(sidecar),
            "guided_patch_indices_used": guided,
            "proper_beam": summarize_decoded_paths(
                proper, decoded_proper, proper_stats, args.gospa_cutoff_pixels),
            "guided_beam": (summarize_decoded_paths(
                guided_paths, decoded_guided, guided_stats, args.gospa_cutoff_pixels)
                            if raw_available else None),
            "search_config": {"beam_width": args.beam_width,
                              "search_engine": args.search_engine,
                              "max_new_tokens": args.max_new_tokens,
                              "max_language": args.max_language,
                              "max_spatial": args.max_spatial,
                              "guided_slots": args.guided_slots,
                              "max_candidates": args.max_candidates,
                              "load_dtype": args.dtype,
                              "gospa_cutoff_pixels": args.gospa_cutoff_pixels}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--audit-native-path", type=Path,
                        help="audit native spatial-head scores without rerunning generation")
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--search-engine", choices=("reference", "cached"),
                        default="reference")
    parser.add_argument("--max-language", type=int, default=8)
    parser.add_argument("--max-spatial", type=int, default=16)
    parser.add_argument("--guided-slots", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--gospa-cutoff-pixels", type=float, default=32.0)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.beam_width < 1 or args.guided_slots < 0:
        parser.error("positive output/beam limits and nonnegative guided slots required")
    if args.guided_slots > args.max_spatial:
        parser.error("guided_slots cannot exceed max_spatial")
    if args.audit_native_path is not None and args.parity_only:
        parser.error("--audit-native-path and --parity-only are separate modes")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if args.audit_native_path is not None:
        print(json.dumps({"item_id": args.item_id,
                          "score_audit_pass": result["score_audit_pass"],
                          "audited_steps": len(result["steps"]),
                          "max_absolute_delta": result["max_absolute_delta"]}))
    else:
        print(json.dumps({"item_id": args.item_id, "native_parity": True,
                          "native_complete": result["native"]["complete"],
                          "proper_complete": result.get("proper_beam", {}).get("stats", {}).get("complete_sequences"),
                          "proper_valid": result.get("proper_beam", {}).get("stats", {}).get("valid_complete_sequences"),
                          "proper_invalid": result.get("proper_beam", {}).get("stats", {}).get("invalid_complete_sequences"),
                          "guided_complete": (result["guided_beam"]["stats"]["complete_sequences"]
                                              if result.get("guided_beam") else None),
                          "guided_valid": (result["guided_beam"]["stats"]["valid_complete_sequences"]
                                           if result.get("guided_beam") else None),
                          "guided_invalid": (result["guided_beam"]["stats"]["invalid_complete_sequences"]
                                             if result.get("guided_beam") else None)}))


if __name__ == "__main__":
    main()
