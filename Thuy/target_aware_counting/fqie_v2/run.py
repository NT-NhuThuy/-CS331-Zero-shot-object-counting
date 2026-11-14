"""Apply a locked FQIE decision to complete frozen candidate/evidence files.

Inputs are JSON objects: candidates={items:[...]}, evidence={items:[...]},
config={tau: number, input_sha256: {candidates: ..., evidence: ...}}.
No GT, labels, or source-key path is accepted by this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .decide import filter_candidates


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_once(path: Path, value: dict) -> None:
    blob = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != blob:
            raise RuntimeError(f"locked output differs: {path}")
        return
    path.write_bytes(blob)


def run(candidates_path: Path, evidence_path: Path, config_path: Path, output_path: Path) -> dict:
    cfg = load(config_path)
    expected = cfg["input_sha256"]
    actual = {"candidates": sha(candidates_path), "evidence": sha(evidence_path)}
    if expected != actual:
        raise RuntimeError("frozen input hash mismatch")
    candidates = load(candidates_path)["items"]
    evidence = load(evidence_path)["items"]
    if not isinstance(candidates, list) or not isinstance(evidence, list):
        raise ValueError("items must be arrays")
    ids = [x["candidate_id"] for x in evidence]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evidence ID")
    kept, ledger = filter_candidates(candidates, {x["candidate_id"]: x for x in evidence}, float(cfg["tau"]))
    result = {"status": "FROZEN_REMOVE_ONLY_DECISION", "input_sha256": actual,
              "config_sha256": sha(config_path), "base_count": len(candidates),
              "count": len(kept), "candidate_ids": [x["candidate_id"] for x in kept],
              "items": kept, "ledger": ledger}
    write_once(output_path, result)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("candidates", "evidence", "config", "output"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    result = run(args.candidates, args.evidence, args.config, args.output)
    print(json.dumps({k: result[k] for k in ("status", "base_count", "count")}))


if __name__ == "__main__":
    main()

