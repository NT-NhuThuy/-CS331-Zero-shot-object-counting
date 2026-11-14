"""Write a hash-locked FQIE decision config without reading labels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .run import sha, write_once


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("candidates", type=Path)
    p.add_argument("evidence", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--tau", type=float, default=0.0)
    a = p.parse_args()
    if not math.isfinite(a.tau):
        raise ValueError("tau must be finite")
    result = {"status": "LOCKED_BEFORE_EVALUATION", "tau": a.tau,
              "input_sha256": {"candidates": sha(a.candidates), "evidence": sha(a.evidence)}}
    write_once(a.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
