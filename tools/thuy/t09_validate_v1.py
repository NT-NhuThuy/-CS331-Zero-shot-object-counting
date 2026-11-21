"""CLI for T09 strict replay and deterministic trace acceptance."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from countlab.method.t09_delivery_v1 import run_t09_delivery


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: str) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/method/solver_config.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--examples")
    mode.add_argument("--manifest")
    parser.add_argument("--events")
    parser.add_argument("--fallback", type=int)
    parser.add_argument("--out")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.examples:
        if args.events is not None or args.fallback is not None or args.out is not None:
            parser.error("--examples does not accept --events/--fallback/--out")
        n = 0
        for line in Path(args.examples).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            actual = run_t09_delivery(item["manifest"], item["observed_events"],
                                     fallback_count=item["fallback_count"], config=config)
            if actual != item["solver_output"]:
                raise SystemExit("HOLD: T09_TRACE_MISMATCH: " + item["case"])
            n += 1
            print(json.dumps({"case": item["case"], "decision": actual["decision"],
                              "count": actual["count"],
                              "fallback": actual["evaluation_fallback_count"],
                              "observed_events_sha256": actual["observed_events_sha256"]},
                             sort_keys=True), flush=True)
        if n == 0:
            raise SystemExit("HOLD: T09_EMPTY_TRACE_SUITE")
        print("T09_TRACE_SUITE_PASS", n)
    else:
        if args.events is None or args.fallback is None or args.out is None:
            parser.error("--manifest requires --events --fallback --out")
        manifest, events = read_json(args.manifest), read_json(args.events)
        outcome = run_t09_delivery(manifest, events, fallback_count=args.fallback, config=config)
        payload = (json.dumps(outcome, ensure_ascii=False, indent=2, sort_keys=True,
                              allow_nan=False) + "\n").encode("utf-8")
        path = Path(args.out)
        with path.open("xb") as handle:
            handle.write(payload)
        print("T09_REPLAY_WRITTEN", str(path))
        print("T09_OUTPUT_SHA256", digest(payload))
        print("T09_DECISION", outcome["decision"], outcome["count"])


if __name__ == "__main__":
    main()
