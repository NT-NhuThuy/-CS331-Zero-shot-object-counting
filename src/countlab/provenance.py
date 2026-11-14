"""Hash utility for CREATE-ONLY artifact manifests; never overwrites silently."""
import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def freeze_manifest(out: str | Path, files: dict[str, str | Path], *, metadata: dict | None = None) -> dict:
    paths = {k: str(Path(v).resolve()) for k, v in sorted(files.items())}
    record = {"schema": "countlab_hash_manifest_v1", "artifacts": {k: {"path": v, "sha256": sha256_file(v)} for k, v in paths.items()},
              "metadata": metadata or {}}
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as f:
        json.dump(record, f, sort_keys=True, ensure_ascii=False, indent=2)
        f.write("\n")
    return record
