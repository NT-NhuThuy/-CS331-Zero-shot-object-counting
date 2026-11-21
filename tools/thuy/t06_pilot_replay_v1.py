"""Run T06 GT-blind CPU graph replay; outputs are create-only and unverified."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from countlab.method.t06_pilot_replay_v1 import run_pilot_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_pilot_file(args.inputs, args.config, args.out_dir)
    print(json.dumps(result, sort_keys=True, indent=2))
    print("T06_CPU_CONTRACT_REPLAY_PASS_REAL_PILOT_HOLD")


if __name__ == "__main__":
    main()
