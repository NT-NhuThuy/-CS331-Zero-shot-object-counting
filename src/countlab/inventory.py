"""Read-only legacy inventory utility."""
import argparse
import hashlib
import json
from pathlib import Path


def scan(root: Path) -> dict:
    sections = {}
    for name in ("loc", "thu", "thuy"):
        p = root / "legacy" / name
        files = sorted(x for x in p.rglob("*") if x.is_file() and not set(x.parts).intersection({".git", "__pycache__", ".pytest_cache"}) and x.suffix != ".pyc")
        sections[name] = {"count": len(files), "total_bytes": sum(f.stat().st_size for f in files),
                          "sample_paths": [str(x.relative_to(root)) for x in files[:8]]}
    return sections


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    a = p.parse_args()
    print(json.dumps(scan(a.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
