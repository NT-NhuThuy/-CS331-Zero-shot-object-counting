"""Create-only, GT-blind batch replay for Thuy's T1 arbitration study.

The input index is a JSON object with schema ``countlab_t1_manifest_index_v1``
and a non-empty ``manifest_paths`` list. Each path points to one frozen
``countlab_candidate_pool_v1`` manifest. No annotations are accepted here:
labels belong to the separate offline evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .baselines import ARM_IDS
from .t1_comparison import run_matched_t1_batch


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_batch(*, index_path: Path, config_path: Path, out_dir: Path) -> dict[str, Any]:
    """Create one immutable, GT-blind decision-summary artifact."""
    if out_dir.exists():
        raise FileExistsError(f"create-only output exists: {out_dir}")
    index = _read_object(index_path)
    if index.get("schema") != "countlab_t1_manifest_index_v1":
        raise ValueError("unsupported T1 manifest index")
    relative_paths = index.get("manifest_paths")
    if (not isinstance(relative_paths, list) or not relative_paths or
            any(not isinstance(item, str) or not item for item in relative_paths)):
        raise ValueError("manifest_paths must be a non-empty string list")
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("manifest index contains duplicate paths")
    config = _read_object(config_path)
    if config.get("schema") != "countlab_t1_experiment_config_v1" or config.get("claims_allowed") is not False:
        raise ValueError("batch replay requires a no-claim T1 experiment config")
    arms = tuple(config.get("arms", ()))
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARM_IDS for arm in arms):
        raise ValueError("config arms must be unique registered B0-B9 identifiers")
    manifest_paths = [(index_path.parent / raw).resolve() for raw in relative_paths]
    manifests = [_read_object(path) for path in manifest_paths]
    summary = run_matched_t1_batch(
        manifests, budget=config.get("verification_budget"),
        pairwise_budget=config.get("pairwise_verification_budget"),
        policy=config.get("policy"), arms=arms)
    result = {
        "schema": "countlab_t1_batch_replay_artifact_v1",
        "status": "FROZEN_EVIDENCE_BATCH_REPLAY_NO_CLAIM",
        "uses_gt_for_replay": False,
        "index_sha256": _sha256(index_path),
        "config_sha256": _sha256(config_path),
        "manifest_records": [
            {"path": raw, "sha256": _sha256(path)}
            for raw, path in zip(relative_paths, manifest_paths)
        ],
        "summary": summary,
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    target = out_dir / "t1_batch_replay.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    lock = {
        "schema": "countlab_t1_batch_replay_lock_v1",
        "status": result["status"],
        "index_sha256": result["index_sha256"],
        "config_sha256": result["config_sha256"],
        "output_sha256": _sha256(target),
        "uses_gt_for_replay": False,
    }
    with (out_dir / "replay_lock.json").open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay_batch(index_path=args.index, config_path=args.config, out_dir=args.out_dir), indent=2))


if __name__ == "__main__":
    main()
