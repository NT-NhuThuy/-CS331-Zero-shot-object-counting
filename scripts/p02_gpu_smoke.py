"""Create-only P02 GPU smoke artifacts; no evaluation labels or model download.

Use an already hash-locked local config and an explicitly selected image/query.
Raw detector capture returns predictions unchanged. V14 parity is not tested.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time


def write_json(path: Path, data) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--query", required=True)
    ap.add_argument("--image-id", required=True)
    ap.add_argument("--query-id", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("smoke output must be a new directory")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    from countlab.integrations.countgd_official import OfficialCountGD, PROTOCOL_ID, file_sha
    from countlab.integrations.online import execute_online
    from countlab.integrations.preflight import inspect
    from countlab.method.schema import _validate_no_gt
    _validate_no_gt(config)
    if (config.get("protocol_id") != PROTOCOL_ID or config.get("verification_budget") != 0
            or config.get("identity", {}).get("enabled", False)
            or config.get("recovery", {}).get("trigger_baseline_le", 0) >= 0):
        raise ValueError("P02 smoke isolates official baseline: disable recovery/Qwen/identity")
    if not args.image.is_file() or not args.query.strip():
        raise ValueError("real image and explicit query required")
    # Existing local assets only. Do not modify the caller's persistent environment.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.out_dir / "config.lock.json", config)
    request = {"image_id": args.image_id, "query_id": args.query_id, "query_text": args.query,
               "image_sha256": file_sha(args.image), "image_path": str(args.image),
               "protocol_sha256": config["official_countgd"]["protocol_sha256"]}
    write_json(args.out_dir / "input_request.json", request)
    summary = {"gpu_smoke": "FAIL", "v14_count_parity": "NOT MEASURED",
               "v14_bbox_parity": "NOT MEASURED", "request": request,
               "smoke_script_sha256": file_sha(Path(__file__))}
    started = time.perf_counter()
    try:
        readiness = inspect(config)
        write_json(args.out_dir / "preflight.json", readiness)
        if readiness["blockers"]:
            raise RuntimeError("preflight BLOCKED: " + "; ".join(readiness["blockers"]))
        provider = OfficialCountGD(config["official_countgd"])
        summary["model_load_sec"] = time.perf_counter() - started
        raw_calls = []
        original_forward = provider.engine.forward

        def capture(caption, exemplars):
            raw = original_forward(caption, exemplars)
            _validate_no_gt(raw)
            raw_calls.append({"caption_sha256": hashlib.sha256(caption.encode()).hexdigest(),
                              "positive_exemplars_resized_xyxy": copy.deepcopy(exemplars),
                              "prediction": copy.deepcopy(raw)})
            return raw

        provider.engine.forward = capture
        manifest, result = execute_online(image_path=args.image, query=args.query,
            image_id=args.image_id, query_id=args.query_id, config=config, backend=provider)
        if len(raw_calls) != 2 or len(provider.runs) != 1 or provider.runs[0]["detector_calls"] != 2:
            raise RuntimeError("expected exactly two real detector calls")
        _validate_no_gt(manifest)
        write_json(args.out_dir / "raw_detector_predictions.json", raw_calls)
        write_json(args.out_dir / "frozen_candidates.json", manifest)
        write_json(args.out_dir / "prediction.json", result)
        summary.update(gpu_smoke="PASS", candidate_count=len(manifest["baseline"]),
                       detector_run=provider.runs[0], runtime=provider.provenance["runtime"],
                       evidence_level="REAL_GPU_SMOKE_NOT_V14_PARITY")
    except Exception as error:
        summary["error_type"] = type(error).__name__
        summary["error"] = str(error)
        raise
    finally:
        summary["total_wall_sec"] = time.perf_counter() - started
        write_json(args.out_dir / "smoke_summary.json", summary)
        hashes = {p.name: file_sha(p) for p in sorted(args.out_dir.iterdir()) if p.is_file()}
        write_json(args.out_dir / "SHA256SUMS.json", hashes)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
