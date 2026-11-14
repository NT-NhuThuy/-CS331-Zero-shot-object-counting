"""Fixed full/context negative-confirmation experiment, inference without GT.

Uses frozen CycleGRPO full-image decisions and a new 1.5x local crop.
The candidate is removed only for explicit NO in both views. This is a
development hypothesis, not a demonstrated semantic/identity guarantee.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
import random
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "target_aware_counting/output"
OLD = OUT / "cyclegrpo_capability_probe_20261027_v1"
DEST = OUT / "fqie_dualview_dev_20261105_v1"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(ROOT / "target_aware_counting"))
SOURCES = {
    "lvis": (OUT / "fqie_h2_alpha_lvis_full_20261104_v1/inference_manifest.json",
             OUT / "fqie_h2_lvis306_cycle_unary_20261104_v1"),
    "fsc_top5": (OUT / "fqie_h2_alpha_absent111_top5_20261104_v1/inference_manifest.json",
                 OUT / "fqie_h2_absent111_cycle_top5_20261104_v1"),
}
POLICY = {"crop_factor": 1.5, "max_new_tokens": 8, "seed": 0,
          "remove_rule": "full == no AND local == no; all UNKNOWN/uncertain kept",
          "unknown_rule": "invalid parse, token ceiling, or unidentifiable view -> UNKNOWN",
          "crop_image": "JPEG quality95 subsampling0; original prediction box never changed",
          "selection": "all 306 LVIS and all 552 previously frozen FSC top5 candidates",
          "replay_count": 12, "training": "none", "threshold_sweep": "none"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def blob(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def atomic_once(path, value, raw=False):
    path = Path(path)
    data = value if raw else blob(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"write-once conflict: {path}")
        return
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise RuntimeError(f"concurrent write conflict: {path}")
    finally:
        os.unlink(tmp)


@contextmanager
def run_lock(destination):
    import fcntl
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another stage is active in this output directory")
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def geometry(box, width, height):
    if len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
        raise ValueError("nonfinite/invalid box")
    b = [max(0., min(float(width), box[0])), max(0., min(float(height), box[1])),
         max(0., min(float(width), box[2])), max(0., min(float(height), box[3]))]
    if b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError("empty clipped box")
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    dx, dy = (b[2] - b[0]) * .75, (b[3] - b[1]) * .75
    bounds = [max(0, math.floor(cx - dx)), max(0, math.floor(cy - dy)),
              min(width, math.ceil(cx + dx)), min(height, math.ceil(cy + dy))]
    local = [b[0] - bounds[0], b[1] - bounds[1], b[2] - bounds[0], b[3] - bounds[1]]
    return b, bounds, local


def strict_parse(raw, tokens, ceiling=8):
    if not isinstance(raw, str) or tokens >= ceiling:
        return "UNKNOWN"
    if "<think>" in raw and "</think>" not in raw:
        return "UNKNOWN"
    text = raw.split("</think>")[-1].strip().lower()
    return text if text in {"yes", "no", "uncertain"} else "UNKNOWN"


def keep(full, local):
    return not (full == "no" and local == "no")


def paired_bootstrap(base, proposed, repetitions=2000):
    """Resample source images, preserving all queries from each sampled source."""
    if [x["query_id"] for x in base] != [x["query_id"] for x in proposed]:
        raise ValueError("bootstrap query binding mismatch")
    groups = defaultdict(list)
    for i, (a, b) in enumerate(zip(base, proposed)):
        if (a["source_image_id"], a["status"], a["gt_count"]) != (b["source_image_id"], b["status"], b["gt_count"]):
            raise ValueError("bootstrap label/source binding mismatch")
        groups[a["source_image_id"]].append(i)
    keys, rng = sorted(groups), random.Random(20261105)
    deltas = {k: [] for k in ("mae", "rmse", "tp50", "false_zero", "absent_mass")}
    def metric(rows, indices):
        p = [rows[i] for i in indices if rows[i]["status"] == "present"]
        a = [rows[i] for i in indices if rows[i]["status"] == "verified_absent"]
        if not p: return None
        e = [x["count"] - x["gt_count"] for x in p]
        return {"mae": sum(abs(x) for x in e)/len(e), "rmse": math.sqrt(sum(x*x for x in e)/len(e)),
                "tp50": sum(x["tp50"] for x in p), "false_zero": sum(x["count"] == 0 for x in p),
                "absent_mass": sum(x["count"] for x in a)}
    for _ in range(repetitions):
        indices = [i for source in rng.choices(keys, k=len(keys)) for i in groups[source]]
        a, b = metric(base, indices), metric(proposed, indices)
        if a is None: continue
        for key in deltas:
            deltas[key].append(b[key] - a[key])
    intervals = {}
    for key, values in deltas.items():
        values.sort()
        if not values: raise ValueError("no valid bootstrap samples")
        intervals[key] = [values[int(.025*(len(values)-1))], values[int(.975*(len(values)-1))]]
    return {"seed": 20261105, "repetitions": repetitions, "unit": "source_image", "delta_95ci": intervals}


def prepare(destination):
    from PIL import Image
    rows, upstream, image_hashes = [], {}, {}
    for cohort, (source, cycle_dir) in SOURCES.items():
        full_manifest = read(cycle_dir / "manifest.json")
        frozen = read(cycle_dir / "freeze_yesno.json")
        if frozen["manifest_sha256"] != sha(cycle_dir / "manifest.json"):
            raise RuntimeError("upstream manifest changed")
        meta = {x["candidate_id"]: x for x in read(source)["items"]}
        full = {x["candidate_id"]: x for x in frozen["items"]}
        native = {x["candidate_id"]: x for x in full_manifest["items"]}
        if set(meta) != set(full) or set(full) != set(native):
            raise RuntimeError("full evidence binding mismatch")
        upstream[str(source)] = sha(source)
        upstream[str(cycle_dir / "freeze_yesno.json")] = sha(cycle_dir / "freeze_yesno.json")
        for cid, x in meta.items():
            response = cycle_dir / "yesno" / (full[cid]["probe_id"] + ".json")
            if sha(response) != full[cid]["response_sha256"]:
                raise RuntimeError("upstream response hash mismatch")
            response_data = read(response)
            if response_data["candidate_id"] != cid or response_data["query"] != x["query"]:
                raise RuntimeError("upstream query binding mismatch")
            # UNKNOWN for truncated output; report the original parse separately.
            decision = strict_parse(response_data["raw_response"], response_data["output_tokens"])
            path = x["image_path"]
            if path not in image_hashes:
                image_hashes[path] = sha(path)
            if image_hashes[path] != x["image_sha256"]:
                raise RuntimeError("source image changed")
            with Image.open(path) as source_image:
                image = source_image.convert("RGB")
                clipped, bounds, local = geometry(x["box_xyxy"], *image.size)
                buf = io.BytesIO()
                image.crop(tuple(bounds)).save(buf, format="JPEG", quality=95, subsampling=0)
            key = hashlib.sha256((cohort + ":" + cid).encode()).hexdigest()
            crop_path = destination / "crops" / (key + ".jpg")
            atomic_once(crop_path, buf.getvalue(), raw=True)
            rows.append({"key": key, "cohort": cohort, "candidate_id": cid,
                         "query_id": str(x.get("query_image_id", x.get("item_id"))),
                         "source_image_id": str(x["source_image_id"]), "query": x["query"],
                         "original_image": path, "original_image_sha256": image_hashes[path],
                         "original_box": x["box_xyxy"], "clipped_box": clipped,
                         "crop_bounds": bounds, "local_box": local,
                         "crop_path": str(crop_path.resolve()), "crop_sha256": sha(crop_path),
                         "full_decision": decision, "legacy_full_decision": full[cid]["strict_decision"],
                         "full_response_sha256": sha(response)})
    if Counter(x["cohort"] for x in rows) != {"lvis": 306, "fsc_top5": 552}:
        raise RuntimeError("cohort denominators changed")
    ordered = sorted(rows, key=lambda x: x["key"])
    replay_keys = [x["key"] for x in ordered[:POLICY["replay_count"]]]
    tasks = []
    for row in ordered:
        if row["key"] in replay_keys:
            tasks.append({"task_id": row["key"] + "_replay", "key": row["key"], "view": "replay"})
        tasks.append({"task_id": row["key"] + "_local", "key": row["key"], "view": "local"})
    manifest = {"status": "LOCKED_BEFORE_LOCAL_INFERENCE_NO_LABELS", "policy": POLICY,
                "code_sha256": sha(__file__), "runtime_lock_sha256": sha(OLD / "runtime_lock.json"),
                "source_hashes": upstream, "items": rows, "tasks": tasks,
                "replay_keys": replay_keys, "count": len(rows)}
    atomic_once(destination / "manifest.json", manifest)
    print(json.dumps({"candidates": len(rows), "local_views": len(rows),
                      "full_replay_views": len(replay_keys),
                      "full_decisions": dict(Counter(x["full_decision"] for x in rows))}))


def load_manifest(destination):
    path = destination / "manifest.json"
    m = read(path)
    if m["code_sha256"] != sha(__file__) or m["policy"] != POLICY:
        raise RuntimeError("code/policy changed since prepare; use a new output directory")
    if m["runtime_lock_sha256"] != sha(OLD / "runtime_lock.json"):
        raise RuntimeError("runtime lock changed")
    return m, sha(path)


def task_input(row, view):
    if view == "local":
        return row["crop_path"], row["local_box"], row["crop_sha256"]
    return row["original_image"], row["clipped_box"], row["original_image_sha256"]


def checked(path, task, fingerprint):
    if not path.exists():
        return None
    data = read(path)
    if data.get("task_id") != task["task_id"] or data.get("manifest_sha256") != fingerprint:
        raise RuntimeError(f"resume fingerprint mismatch: {path}")
    return data


def gpu(destination, stage, limit):
    import torch
    from runtime_gate import inspect_lock
    from run_native_smoke import load_vq, load_qwen
    from native_bridge import encode_rectangle_region_native, generate_native_message
    from capability_parser import clean_jpeg_data_url, unary_query_yesno_message
    m, fingerprint = load_manifest(destination)
    tasks = m["tasks"] if limit is None else m["tasks"][:limit]
    rows = {x["key"]: x for x in m["items"]}
    # Validate all prerequisites before loading GPU weights.
    checked_images = set()
    for task in tasks:
        path, box, digest = task_input(rows[task["key"]], task["view"])
        if path not in checked_images:
            if sha(path) != digest:
                raise RuntimeError("view image changed")
            checked_images.add(path)
        if stage == "verify":
            enc_path = destination / "encodings" / (task["task_id"] + ".json")
            if checked(enc_path, task, fingerprint) is None:
                raise RuntimeError(f"encode incomplete: {task['task_id']}")
    report = inspect_lock(OLD / "runtime_lock.json")
    if report["blockers"]:
        raise RuntimeError(report["blockers"])
    pending = [t for t in tasks if checked(destination / ("encodings" if stage == "encode" else "responses") /
                                          (t["task_id"] + ".json"), t, fingerprint) is None]
    if not pending:
        print(json.dumps({"stage": stage, "status": "already_complete", "tasks": len(tasks)}))
        return
    torch.manual_seed(POLICY["seed"])
    lock = read(OLD / "runtime_lock.json")
    started = time.monotonic()
    if stage == "encode":
        model, processor = load_vq(lock), None
    else:
        model, processor = load_qwen(lock)
    with torch.inference_mode():
        for i, task in enumerate(pending):
            row = rows[task["key"]]
            image_path, box, digest = task_input(row, task["view"])
            record = {"task_id": task["task_id"], "manifest_sha256": fingerprint,
                      "candidate_id": row["candidate_id"], "image_sha256": digest, "view": task["view"]}
            if stage == "encode":
                result = encode_rectangle_region_native(image_path, box, model)
                atomic_once(destination / "encodings" / (task["task_id"] + ".json"), {**record, **result})
            else:
                ep = destination / "encodings" / (task["task_id"] + ".json")
                enc = checked(ep, task, fingerprint)
                msg = unary_query_yesno_message(enc["mask_tokens"], row["query"], clean_jpeg_data_url(image_path))
                result = generate_native_message(model, processor, msg, POLICY["max_new_tokens"])
                atomic_once(destination / "responses" / (task["task_id"] + ".json"),
                            {**record, "query": row["query"], "encoding_sha256": sha(ep),
                             "prompt_text": msg[0]["content"][1]["text"], **result,
                             "decision": strict_parse(result["raw_response"], result["output_tokens"])})
            if i % 25 == 0:
                print(json.dumps({"stage": stage, "completed_this_run": i + 1, "pending_at_start": len(pending)}), flush=True)
    atomic_once(destination / "runs" / (stage + "_" + str(time.time_ns()) + ".json"),
                {"stage": stage, "tasks_completed": len(pending), "wall_seconds": time.monotonic() - started,
                 "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
                 "python": sys.version, "torch": torch.__version__, "manifest_sha256": fingerprint})
    print(json.dumps({"stage": stage, "completed_this_run": len(pending), "wall_seconds": time.monotonic() - started}))


def freeze(destination):
    m, fingerprint = load_manifest(destination)
    rows = {x["key"]: x for x in m["items"]}
    decisions, hashes = {}, {}
    for task in m["tasks"]:
        rp = destination / "responses" / (task["task_id"] + ".json")
        ep = destination / "encodings" / (task["task_id"] + ".json")
        r, e = checked(rp, task, fingerprint), checked(ep, task, fingerprint)
        if r is None or e is None:
            raise RuntimeError(f"incomplete task: {task['task_id']}")
        if r["encoding_sha256"] != sha(ep) or r["query"] != rows[task["key"]]["query"]:
            raise RuntimeError("response/encoding/query binding mismatch")
        parsed = strict_parse(r["raw_response"], r["output_tokens"])
        if parsed != r["decision"]:
            raise RuntimeError("stored strict decision mismatch")
        hashes[task["task_id"]] = {"response": sha(rp), "encoding": sha(ep)}
        decisions[task["task_id"]] = parsed
    items = [{**x, "local_decision": decisions[x["key"] + "_local"],
              "keep": keep(x["full_decision"], decisions[x["key"] + "_local"])} for x in m["items"]]
    replay = [{"key": key, "cached": rows[key]["full_decision"], "replayed": decisions[key + "_replay"]}
              for key in m["replay_keys"]]
    atomic_once(destination / "freeze.json", {"status": "FROZEN_BEFORE_NEW_LABEL_JOIN", "manifest_sha256": fingerprint,
                                              "items": items, "replay": replay, "hashes": hashes})
    print(json.dumps({"frozen_candidates": len(items), "kept": sum(x["keep"] for x in items),
                      "replay_agreement": sum(x["cached"] == x["replayed"] for x in replay), "replay_total": len(replay)}))


def evaluate(destination):
    # GT is accessed only in this post-freeze entry point.
    from fqie_h2_lvis_remove_e2e import SRC, xyxy, match50
    m, fingerprint = load_manifest(destination)
    f = read(destination / "freeze.json")
    if f["status"] != "FROZEN_BEFORE_NEW_LABEL_JOIN" or f["manifest_sha256"] != fingerprint:
        raise RuntimeError("prediction freeze mismatch")
    for task, hashes in f["hashes"].items():
        if sha(destination / "responses" / (task + ".json")) != hashes["response"]:
            raise RuntimeError("frozen response modified")
        if sha(destination / "encodings" / (task + ".json")) != hashes["encoding"]:
            raise RuntimeError("frozen encoding modified")
    records = f["items"]
    labels = {x["candidate_id"]: x for x in read(SOURCES["lvis"][0].parent / "eval_labels.json")["items"]}
    rules = {"no_op": lambda x: True,
             "full_yes_legacy": lambda x: x["legacy_full_decision"] == "yes",
             "full_not_no": lambda x: x["full_decision"] != "no",
             "local_not_no": lambda x: x["local_decision"] != "no",
             "dual_no_remove": lambda x: keep(x["full_decision"], x["local_decision"])}
    gt = read(SRC / "input_274/query_pairs_coco.json")
    gt_boxes, candidates = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        gt_boxes[str(a["image_id"])].append(xyxy(a["bbox"]))
    for r in records:
        if r["cohort"] == "lvis":
            candidates[r["query_id"]].append(r)
    result = {"status": "DEVELOPMENT_ONLY", "rules": {}, "per_query": {},
              "freeze_sha256": sha(destination / "freeze.json"), "replay": f["replay"],
              "limits": ["Previously exposed development cohorts; not independent confirmation.",
                         "FSC top-five only; lower-rank absent mass is unmeasured.",
                         "Same model in both views; no independence or identity guarantee."]}
    for name, fn in rules.items():
        per_query = []
        for image in gt["images"]:
            qid = str(image["id"])
            chosen = [r["original_box"] for r in candidates[qid] if fn(r)]
            status = image["query_status"]
            if status not in {"present", "verified_absent"}:
                raise RuntimeError("unexpected label status")
            per_query.append({"query_id": qid, "source_image_id": str(image["source_image_id"]),
                              "status": status, "count": len(chosen), "gt_count": image["gt_count"],
                              "tp50": match50(chosen, gt_boxes[qid]) if status == "present" else 0})
        present = [x for x in per_query if x["status"] == "present"]
        absent = [x for x in per_query if x["status"] == "verified_absent"]
        errors = [x["count"] - x["gt_count"] for x in present]
        top5 = [x for x in records if x["cohort"] == "fsc_top5"]
        active = {x["query_id"] for x in top5 if fn(x)}
        result["rules"][name] = {"lvis_mae": sum(abs(e) for e in errors) / len(errors),
            "lvis_rmse": math.sqrt(sum(e*e for e in errors)/len(errors)),
            "lvis_tp50": sum(x["tp50"] for x in present), "lvis_false_zero": sum(x["count"] == 0 for x in present),
            "lvis_absent_mass": sum(x["count"] for x in absent),
            "lvis_strict_targets_kept": sum(fn(x) and labels[x["candidate_id"]]["eval_label"] == "TARGET_IOU50"
                                            for x in records if x["cohort"] == "lvis"),
            "fsc_top5_absent_boxes": sum(fn(x) for x in top5), "fsc_top5_active_queries": len(active),
            "fsc_top5_queries_total": len({x["query_id"] for x in top5})}
        result["per_query"][name] = per_query
    base, proposed = result["rules"]["no_op"], result["rules"]["dual_no_remove"]
    full = result["rules"]["full_not_no"]
    replay_ok = all(x["cached"] == x["replayed"] for x in f["replay"])
    result["development_screen"] = {
        "replay_exact": replay_ok,
        "tp_not_below_base": proposed["lvis_tp50"] >= base["lvis_tp50"],
        "false_zero_not_above_base": proposed["lvis_false_zero"] <= base["lvis_false_zero"],
        "mae_better_than_base": proposed["lvis_mae"] < base["lvis_mae"],
        "rmse_not_above_base": proposed["lvis_rmse"] <= base["lvis_rmse"],
        "fsc_top5_activity_not_above_full": proposed["fsc_top5_active_queries"] <= full["fsc_top5_active_queries"],
        "fsc_top5_mass_not_above_full": proposed["fsc_top5_absent_boxes"] <= full["fsc_top5_absent_boxes"]}
    result["decision"] = "ELIGIBLE_FOR_NEW_SOURCE_CONFIRMATION" if all(result["development_screen"].values()) else "NOT_PROMOTED"
    if base["lvis_tp50"] != 163 or result["rules"]["full_yes_legacy"]["lvis_tp50"] != 157:
        raise RuntimeError("cached LVIS control failed to reproduce historical TP50")
    result["bootstrap"] = {name: paired_bootstrap(result["per_query"][name], result["per_query"]["dual_no_remove"])
                           for name in ("no_op", "full_not_no")}
    atomic_once(destination / "evaluation.json", result)
    print(json.dumps({"rules": result["rules"], "screen": result["development_screen"], "decision": result["decision"]}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "encode", "verify", "freeze", "evaluate"))
    p.add_argument("--out", type=Path, default=DEST)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    if a.limit is not None and (a.limit <= 0 or a.stage not in {"encode", "verify"}):
        p.error("--limit is only for bounded GPU smoke")
    with run_lock(a.out):
        if a.stage == "prepare": prepare(a.out)
        elif a.stage in {"encode", "verify"}: gpu(a.out, a.stage, a.limit)
        elif a.stage == "freeze": freeze(a.out)
        else: evaluate(a.out)


if __name__ == "__main__":
    main()
