"""Validate the structural contract of the ZOC research archive."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"README.md", "config.yaml", "command.sh", "manifest.json"}


def main() -> int:
    experiments = ROOT / "experiments"
    failures: list[str] = []
    for path in sorted(experiments.glob("E[0-9][0-9]_*/")):
        missing = sorted(name for name in REQUIRED if not (path / name).exists())
        if not (path / "results" / "metrics.json").exists():
            missing.append("results/metrics.json")
        if missing:
            failures.append(f"{path.name}: missing {', '.join(missing)}")
        manifest = path / "manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text())
                for key in ("status", "dataset_version", "split", "cohort_id", "checkpoint_sha256", "git_commit", "command", "evaluation_protocol", "metric_file"):
                    if key not in data:
                        failures.append(f"{path.name}: manifest missing {key}")
            except json.JSONDecodeError as exc:
                failures.append(f"{path.name}: invalid manifest JSON ({exc})")
        metrics_path = path / "results" / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
                if not metrics.get("metrics"):
                    failures.append(f"{path.name}: metrics.json has no reported metrics/conclusion")
            except json.JSONDecodeError as exc:
                failures.append(f"{path.name}: invalid metrics JSON ({exc})")
    if failures:
        print("Archive validation failed:")
        print("\n".join(f"- {item}" for item in failures))
        return 1
    print("Archive structure is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
