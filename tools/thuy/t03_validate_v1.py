"""T03 local-only boundary checker for reviewed example fixtures or a JSON pair.

Run with PYTHONPATH=src from the repository root. Does not inspect GT files.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from countlab.method.t03_contract_v1 import validate_candidate_pool, validate_method_output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prediction", type=Path)
    args = parser.parse_args()
    if bool(args.examples) == bool(args.manifest):
        parser.error("supply --examples or --manifest")
    if args.examples:
        doc = json.loads(args.examples.read_text(encoding="utf-8"))
        examples = doc["examples"]
        for row in examples:
            a = validate_candidate_pool(row["manifest"])
            b = validate_method_output(row["manifest"], row["prediction"])
            print(json.dumps({"case": row["case"], "input": a["input_sha256"],
                              "decision": b["decision"], "fallback": b["fallback_count"]}, sort_keys=True))
        print("T03_EXAMPLE_SUITE_PASS", len(examples))
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        print(json.dumps(validate_candidate_pool(manifest), sort_keys=True))
        if args.prediction:
            prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
            print(json.dumps(validate_method_output(manifest, prediction), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
