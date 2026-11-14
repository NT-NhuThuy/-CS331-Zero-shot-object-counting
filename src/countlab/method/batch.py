"""Immutable batch replay after conversion from frozen candidate/evidence artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .pipeline import run_method


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a directory of GT-blind frozen candidate JSON files")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--budget", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("method_id") != "UNICOUNT_V1_CPU_REPLAY":
        raise ValueError("unsupported method config/version")
    files = sorted(args.inputs.glob("*.json"))
    if not files:
        raise ValueError("no JSON files in inputs")
    args.outputs.mkdir(parents=True, exist_ok=True)
    expected = [args.outputs / x.name for x in files]
    if any(x.exists() for x in expected):
        raise FileExistsError("output already exists; cannot overwrite frozen results")
    results = []
    ids = set()
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        key = (str(data.get("image_id")), str(data.get("query_id")))
        if key in ids:
            raise ValueError(f"duplicate image-query pair: {key}")
        ids.add(key)
        results.append(run_method(data, budget=args.budget, policy=config["policy"]))
    for out, destination in zip(results, expected):
        with destination.open("x", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
    print(json.dumps({"n_pairs": len(results), "n_abstained": sum(x["query_abstained"] for x in results),
                      "n_partial": sum(x["count_status"] == "partial" for x in results),
                      "method_policy_sha256": results[0]["method_policy_sha256"],
                      "outputs": str(args.outputs)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
