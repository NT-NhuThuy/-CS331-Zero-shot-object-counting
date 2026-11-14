"""Resumable MolmoPoint native greedy control on existing development manifests.

Requires a locally pinned allenai/MolmoPoint-8B snapshot. Never downloads model code
or weights, and never opens evaluation labels. Run one process per output file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from contextlib import nullcontext
from pathlib import Path

from .point_grammar import validate_point_tokens

COHORTS = {
    "lvis_dev300": ("lvis_dev300.jsonl", 300),
    "fsc_primary_dev100": ("fsc_primary_dev100.jsonl", 100),
    "fsc_verified_absent111": ("fsc_verified_absent111.jsonl", 111),
}
ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "target_aware_counting/output/sink_target_selection_20261029_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(cohort: str) -> tuple[Path, list[dict]]:
    name, expected = COHORTS[cohort]
    path = MANIFEST_DIR / name
    registry = json.loads((MANIFEST_DIR / "baseline_registry.json").read_text())
    locked = registry["cohorts"][cohort]
    if locked["items"] != expected or locked["manifest_sha256"] != sha256(path):
        raise ValueError(f"{cohort}: manifest differs from frozen baseline registry")
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != expected or len({r["item_id"] for r in rows}) != expected:
        raise ValueError(f"{cohort}: manifest denominator or item IDs changed")
    for row in rows:
        if not all(k in row for k in ("image_path", "query", "source_image_id")):
            raise ValueError(f"malformed manifest item {row.get('item_id')}")
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
    return path, rows


def load_existing(path: Path, contract: dict, item_ids: set[str]) -> dict[str, dict]:
    if not path.exists():
        return {}
    found = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            item_id = row["item_id"]
            if row["contract"] != contract or item_id not in item_ids or item_id in found:
                raise ValueError("existing output has mixed config, foreign item, or duplicate item")
            found[item_id] = row
    return found


def validate_points(points, image_size: tuple[int, int]) -> list[list[float]]:
    import math
    width, height = image_size
    result = []
    for point in points:
        if len(point) != 4 or point[1] != 0:
            raise ValueError(f"unexpected point record: {point}")
        x, y = float(point[2]), float(point[3])
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= width and 0 <= y <= height):
            raise ValueError(f"point outside original image: {point}")
        result.append([x, y])
    return result


def prepare_one(processor, row: dict, image_override=None, *,
                prompt_style: str = "every_instance"):
    from PIL import Image

    if image_override is None:
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
    else:
        image = image_override.convert("RGB")
    width, height = image.size
    clean_query = row["query"].strip().rstrip(".").strip()
    if prompt_style == "every_instance":
        prompt = f"Point to every instance of {clean_query}."
    elif prompt_style == "official_short":
        prompt = f"Point to the {clean_query}."
    else:
        raise ValueError(f"unknown prompt style: {prompt_style}")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image", "image": image},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        return_dict=True, padding=True, return_pointing_metadata=True)
    metadata = inputs.pop("metadata")
    return inputs, metadata, (width, height)


def _inference_autocast(device, model_dtype):
    """Keep MolmoPoint connector inputs compatible with reduced precision weights."""
    import torch
    if device.type == "cuda" and model_dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type="cuda", dtype=model_dtype)
    return nullcontext()


def infer_one(model, processor, row: dict, max_new_tokens: int, image_override=None, *,
              prompt_style: str = "every_instance",
              decoding_style: str = "native") -> dict:
    import torch

    inputs, metadata, (width, height) = prepare_one(
        processor, row, image_override, prompt_style=prompt_style)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit is None:
        context_limit = getattr(getattr(model.config, "text_config", None), "max_position_embeddings", None)
    if context_limit is not None and inputs["input_ids"].shape[1] + max_new_tokens > context_limit:
        raise ValueError("prompt plus output limit exceeds model context")
    started = time.monotonic()
    model_dtype = next(model.parameters()).dtype
    logits_processors = model.build_logit_processor_from_inputs(inputs)
    if decoding_style == "strict_points_v1":
        from .constrained_point_decoding import strict_point_processor
        logits_processors.append(strict_point_processor(model, processor, inputs))
    elif decoding_style != "native":
        raise ValueError(f"unknown decoding style: {decoding_style}")
    with torch.inference_mode(), _inference_autocast(device, model_dtype):
        output = model.generate(
            **inputs,
            logits_processor=logits_processors,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated = output[:, inputs["input_ids"].shape[1]:]
    token_ids = generated[0].tolist()
    eos_ids = model.generation_config.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    complete = bool(token_ids and token_ids[-1] in eos_ids)
    text = processor.post_process_image_text_to_text(
        generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    points = model.extract_image_points(
        text, metadata["token_pooling"], metadata["subpatch_mapping"],
        metadata["image_sizes"])
    xy = validate_points(points, (width, height))
    bounds = model.model.build_token_bounds(inputs["image_token_pooling"])
    grammar = validate_point_tokens(token_ids, bounds, len(xy))
    return {
        "status": "complete" if complete else "truncated",
        "count": len(xy) if complete else None,
        "points_xy": xy,
        "raw_points": [[value.item() if hasattr(value, "item") else value
                        for value in point] for point in points],
        **grammar,
        "generated_text": text,
        "output_token_ids": token_ids,
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "output_tokens": len(token_ids),
        "elapsed_seconds": time.monotonic() - started,
        "image_size_wh": [width, height],
        "decoding_style": decoding_style,
    }


def run(args) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    snapshot = Path(args.snapshot).resolve()
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise FileNotFoundError("provide a local pinned MolmoPoint-8B snapshot")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision) or snapshot.name != args.revision or \
            snapshot.parent.name != "snapshots" or \
            snapshot.parent.parent.name != "models--allenai--MolmoPoint-8B":
        from .snapshot_lock import verify
        verify(snapshot, args.revision)
    config = json.loads((snapshot / "config.json").read_text())
    if "molmo" not in str(config.get("model_type", "")).lower() and \
            "Molmo" not in str(config.get("architectures", [])):
        raise ValueError("snapshot does not identify a Molmo model")
    manifest_path, items = load_manifest(args.cohort)
    if args.item_id is not None:
        items = [item for item in items if item["item_id"] == args.item_id]
        if not items:
            raise ValueError("item_id not in locked development cohort")
    contract = {
        "method": ("molmopoint_native_strict_points_v1"
                   if args.decoding_style == "strict_points_v1" else "molmopoint_native_greedy"),
        "cohort": args.cohort,
        "manifest_sha256": sha256(manifest_path),
        "snapshot": str(snapshot),
        "revision": args.revision,
        "config_sha256": sha256(snapshot / "config.json"),
        "processor_config_sha256": sha256(snapshot / "processor_config.json"),
        "tokenizer_config_sha256": sha256(snapshot / "tokenizer_config.json"),
        "snapshot_lock_sha256": sha256(snapshot / "molmo_point_snapshot_lock.json")
                                if (snapshot / "molmo_point_snapshot_lock.json").exists() else None,
        "weight_files": {p.name: p.stat().st_size for p in sorted(snapshot.glob("*.safetensors"))},
        "max_new_tokens": args.max_new_tokens,
        "load_dtype": args.dtype,
        "prompt_template": "Point to every instance of {query_without_trailing_period}.",
        "processor_default_hf_4_57_1": True,
        "decoding_style": args.decoding_style,
    }
    if args.item_id is not None:
        contract["item_id_filter"] = args.item_id
    if not contract["weight_files"]:
        raise FileNotFoundError("snapshot has no local safetensors weights")
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError("output exists; pass --resume after checking contract")
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(output, contract, {x["item_id"] for x in items})
    load_dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True, dtype=load_dtype,
        device_map=args.device_map, offload_folder=args.offload_folder,
        offload_state_dict=True).eval()
    processor = AutoProcessor.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True,
        padding_side="left")
    with output.open("a", encoding="utf-8") as stream:
        for item in items:
            if item["item_id"] in existing:
                continue
            try:
                result = infer_one(model, processor, item, args.max_new_tokens,
                                   decoding_style=args.decoding_style)
            except Exception as exc:
                result = {"status": "error", "count": None, "points_xy": [],
                          "error_type": type(exc).__name__, "error": str(exc)}
                if args.fail_fast:
                    raise
            record = {"item_id": item["item_id"], "source_image_id": item["source_image_id"],
                      "query": item["query"], "contract": contract, **result}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(f"{len(existing)+1}/{len(items)} {item['item_id']} {result['status']}", flush=True)
            existing[item["item_id"]] = record
    counts = {status: sum(r["status"] == status for r in existing.values())
              for status in ("complete", "truncated", "error")}
    print(json.dumps({"cohort": args.cohort, "denominator": len(items),
                      "rows": len(existing), "status_counts": counts,
                      "output": str(output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=COHORTS)
    parser.add_argument("--item-id", default=None,
                        help="optional one-item diagnostic within the locked cohort")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--revision", required=True, help="40-hex Hugging Face snapshot commit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--decoding-style", choices=("native", "strict_points_v1"),
                        default="native")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    run(args)


if __name__ == "__main__":
    main()
