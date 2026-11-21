"""Create-only T12 matched offline replay; NO GT, GPUs, Qwen, or actual cost.

Example: py tools/thuy/t12_b5_b6_replay_v1.py --inputs ... --out-dir ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from countlab.method.t12_fixed_baselines_v1 import run_t12_matched


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def read_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("config must be JSON object")
    return cfg


def replay(inputs: Path, b5_path: Path, b6_path: Path, out_dir: Path) -> dict:
    if out_dir.exists():
        raise FileExistsError(f"HOLD: create-only output already exists: {out_dir}")
    source = inputs.read_bytes()
    b5 = read_config(b5_path)
    b6 = read_config(b6_path)
    seen: set[tuple[str, str]] = set()
    rows = []
    for number, line in enumerate(source.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {"schema", "manifest", "observed_events"}:
            raise ValueError(f"line {number}: unexpected row fields")
        if row["schema"] != "countlab_t12_evidence_bank_row_v1":
            raise ValueError(f"line {number}: invalid row schema")
        manifest = row["manifest"]
        key = (manifest["image_id"], manifest["query_id"])
        if key in seen:
            raise ValueError(f"line {number}: duplicate image/query pair")
        seen.add(key)
        rows.append(run_t12_matched(manifest, row["observed_events"], b5_config=b5, b6_config=b6))
    if not rows:
        raise ValueError("input roster must be nonempty")
    prediction_bytes = ("\n".join(canonical(x) for x in rows) + "\n").encode("utf-8")
    meta = {"schema": "countlab_t12_b5_b6_replay_lock_v1", "status":
            "CPU_CONTRACT_REPLAY_ONLY_PENDING_C2_COST_AND_B7",
            "roster_pairs": len(rows), "input_sha256": sha_bytes(source),
            "b5_config_sha256": sha_bytes(b5_path.read_bytes()),
            "b6_config_sha256": sha_bytes(b6_path.read_bytes()),
            "predictions_sha256": sha_bytes(prediction_bytes),
            "gt_used": False, "actual_model_calls": None,
            "matched_to_b7_measured_cost": False}
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "B5_B6_predictions.jsonl").write_bytes(prediction_bytes)
    (out_dir / "replay_lock.json").write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--b5-config", type=Path, default=ROOT / "configs/method/B5_config.json")
    ap.add_argument("--b6-config", type=Path, default=ROOT / "configs/method/B6_config.json")
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    print(json.dumps(replay(a.inputs, a.b5_config, a.b6_config, a.out_dir),
                     ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
