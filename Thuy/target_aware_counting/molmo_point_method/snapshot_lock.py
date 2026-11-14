"""Hash a complete official MolmoPoint local snapshot once after download."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .native_greedy import sha256

LOCK_NAME = "molmo_point_snapshot_lock.json"
REPO = "allenai/MolmoPoint-8B"


def lock(snapshot: Path, revision: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full 40-hex commit")
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    index = snapshot / "model.safetensors.index.json"
    config = snapshot / "config.json"
    if not index.is_file() or not config.is_file():
        raise FileNotFoundError("complete model index and config required")
    if json.loads(config.read_text())["model_type"] != "molmo_point":
        raise ValueError("not a MolmoPoint model")
    shards = set(json.loads(index.read_text())["weight_map"].values())
    files = sorted(path for path in snapshot.iterdir()
                   if path.is_file() and path.name != LOCK_NAME)
    if any(not (snapshot / name).is_file() for name in shards):
        raise FileNotFoundError("missing checkpoint shard or source file")
    if not any(path.name == "modeling_molmo_point.py" for path in files) or \
            not any(path.name == "tokenizer.json" for path in files):
        raise FileNotFoundError("missing official model source or tokenizer")
    result = {"repo": REPO, "revision": revision,
              "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                        for path in files}}
    with (snapshot / LOCK_NAME).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return result


def verify(snapshot: Path, revision: str) -> dict:
    path = snapshot / LOCK_NAME
    result = json.loads(path.read_text())
    if result["repo"] != REPO or result["revision"] != revision:
        raise ValueError("snapshot repository/revision mismatch")
    for name, expected in result["files"].items():
        file = snapshot / name
        if file.stat().st_size != expected["bytes"] or sha256(file) != expected["sha256"]:
            raise ValueError(f"snapshot file changed: {name}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(json.dumps({"files": len(lock(args.snapshot, args.revision)["files"])}))


if __name__ == "__main__":
    main()
