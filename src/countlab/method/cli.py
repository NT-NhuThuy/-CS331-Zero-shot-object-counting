"""countlab-method: one entrypoint for frozen-evidence replay, create-only output."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .pipeline import run_method


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified CountLab v1 replay (does not run CountGD++/Qwen)")
    parser.add_argument("--input", type=Path, required=True, help="GT-free frozen candidate/evidence JSON")
    parser.add_argument("--out", type=Path, required=True, help="create-only JSON output")
    parser.add_argument("--budget", type=int, default=None, help="maximum number of unary evidence slots to use")
    parser.add_argument("--config", type=Path, help="versioned method policy config JSON")
    parser.add_argument("--no-recovery", action="store_true")
    parser.add_argument("--no-identity", action="store_true")
    parser.add_argument("--abstain-on-conflict", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    if config is not None and config.get("method_id") != "UNICOUNT_V1_CPU_REPLAY":
        raise ValueError("unsupported method config/version")
    output = run_method(source, budget=args.budget, policy=config["policy"] if config else None,
                        enable_recovery=not args.no_recovery,
                        enable_identity=not args.no_identity,
                        abstain_on_conflict=args.abstain_on_conflict)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    print(json.dumps({"count": output["count"], "status": output["count_status"],
                      "sha256": output["input_sha256"], "output": str(args.out)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
