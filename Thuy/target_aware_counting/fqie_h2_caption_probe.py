"""Bounded frozen CycleGRPO region-caption capability probe on development.

This tests query-free descriptions only. It is not an Alpha-CLIP comparator,
semantic decision rule, or population estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "target_aware_counting/output"
OLD = OUT / "cyclegrpo_capability_probe_20261027_v1"
DEST = OUT / "fqie_h2_caption_smoke_20261104_v1"
sys.path.insert(0, str(OLD))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"locked output differs: {path}")
    path.write_bytes(content)


def manifest():
    a = {x["source_image_id"]: x for x in rows(OUT / "a8_weak_singleton_val_20261102_v1/predictions.jsonl")}
    b = {x["source_image_id"]: x for x in rows(OUT / "b0_a8_paired_val_sm86_20261103_v2/predictions_b0.jsonl")}
    m = {x["source_image_id"]: x for x in rows(OUT / "b0_a8_paired_val_sm86_20261103_v2/input_queries.jsonl")}
    selections = [(111, "A4", 0), (380, "A4", 0), (1152, "B0", 0),
                  (197, "B0", 0), (280, "B0", 0), (980, "A4", 0),
                  (980, "B0", 0)]
    items = []
    for source_id, source, index in selections:
        item = m[source_id]
        candidate = (a[source_id]["a4_add"] if source == "A4" else b[source_id]["additions"])[index]
        items.append({"probe_id": f"P{len(items):02d}", "item_id": item["item_id"],
                      "image_path": item["image_path"], "image_sha256": item["image_sha256"],
                      "query_for_later_audit_only": item["canonical"],
                      "source": source, "candidate_id": candidate["candidate_id"],
                      "box_xyxy": candidate["box_xyxy"]})
    absent_dir = OUT / "a6_lvis_absent_safety_20261102_v1"
    absent_manifest = {x["item_id"]: x for x in rows(absent_dir / "input_manifest.jsonl")}
    absent_pred = {x["item_id"]: x for x in rows(absent_dir / "predictions.jsonl")}
    absent_id = "lvis_a6_absent:340089:1171"
    am, ap = absent_manifest[absent_id], absent_pred[absent_id]
    assert len(ap["a4_add"]) == 1 and ap["m0_count"] == 0
    candidate = ap["a4_add"][0]
    items.append({"probe_id": "P07", "item_id": absent_id,
                  "image_path": am["image_path"], "image_sha256": am["image_sha256"],
                  "query_for_later_audit_only": am["query"],
                  "source": "A4_absent", "candidate_id": candidate["candidate_id"],
                  "box_xyxy": candidate["box_xyxy"]})
    assert len(items) == 8
    return items


def prepare():
    items = manifest()
    for x in items:
        if sha(x["image_path"]) != x["image_sha256"]:
            raise RuntimeError(f"image hash mismatch: {x['probe_id']}")
    write_once(DEST / "manifest.json", {"status": "LOCKED_DEVELOPMENT_SMOKE",
        "items": items, "max_regions": 8, "caption_prompt": "native CycleGRPO query-free region caption",
        "max_new_tokens": 96, "input_query_sent_to_caption_model": False,
        "model_lock_sha256": sha(OLD / "runtime_lock.json"),
        "limitation": "cases selected from previously analyzed development cohorts; no population inference"})
    print(json.dumps({"prepared": len(items), "manifest": str(DEST / "manifest.json")}))


def gpu_stage(stage):
    from run_native_smoke import gpu_gate, load_vq, load_qwen
    from native_bridge import encode_rectangle_region_native, generate_native_message
    from capability_parser import native_caption_message, clean_jpeg_data_url
    lock = json.loads((OLD / "runtime_lock.json").read_text())
    prepared = json.loads((DEST / "manifest.json").read_text())
    assert prepared["model_lock_sha256"] == sha(OLD / "runtime_lock.json")
    items = prepared["items"]
    assert len(items) == 8 and prepared["max_new_tokens"] == 96
    stamp = json.loads((DEST / "preflight_pass.json").read_text())
    if stamp != {"state": "ready_for_separately_authorized_runtime",
                  "model_lock_sha256": sha(OLD / "runtime_lock.json")}:
        raise RuntimeError("CPU preflight stamp missing or lock changed")
    import torch
    torch.manual_seed(0)
    gpu_gate(18)
    started = time.monotonic()
    if stage == "encode":
        model = load_vq(lock)
        for x in items:
            path = DEST / "encodings" / f"{x['probe_id']}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["image_sha256"] != x["image_sha256"] or saved["candidate_id"] != x["candidate_id"]:
                    raise RuntimeError(f"resume mismatch: {path}")
                continue
            result = encode_rectangle_region_native(x["image_path"], x["box_xyxy"], model)
            write_once(path, {"probe_id": x["probe_id"], "image_sha256": x["image_sha256"],
                              "candidate_id": x["candidate_id"], **result})
            print(json.dumps({"encoded": x["probe_id"]}), flush=True)
    else:
        if not all((DEST / "encodings" / f"{x['probe_id']}.json").exists() for x in items):
            raise RuntimeError("complete encode stage first")
        model, processor = load_qwen(lock)
        for x in items:
            path = DEST / "captions" / f"{x['probe_id']}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["image_sha256"] != x["image_sha256"] or saved["candidate_id"] != x["candidate_id"]:
                    raise RuntimeError(f"resume mismatch: {path}")
                continue
            enc = json.loads((DEST / "encodings" / f"{x['probe_id']}.json").read_text())
            message = native_caption_message(enc["mask_tokens"], clean_jpeg_data_url(x["image_path"]))
            prompt = message[0]["content"][1]["text"]
            if x["query_for_later_audit_only"].strip(" .").lower() in prompt.lower():
                raise RuntimeError("query leakage into caption prompt")
            result = generate_native_message(model, processor, message, 96)
            write_once(path, {"probe_id": x["probe_id"], "image_sha256": x["image_sha256"],
                              "candidate_id": x["candidate_id"], "prompt_text": prompt, **result})
            print(json.dumps({"captioned": x["probe_id"], "seconds": result["generation_seconds"]}), flush=True)
    print(json.dumps({"stage": stage, "wall_seconds": time.monotonic() - started,
                      "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "preflight", "encode", "caption"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "preflight":
        from runtime_gate import inspect_lock
        report = inspect_lock(OLD / "runtime_lock.json")
        if report["blockers"]:
            raise RuntimeError(f"CycleGRPO lock blocked: {report['blockers']}")
        write_once(DEST / "preflight_pass.json", {"state": report["state"],
                   "model_lock_sha256": sha(OLD / "runtime_lock.json")})
        print(json.dumps({"preflight": report["state"]}))
    else:
        gpu_stage(args.stage)
