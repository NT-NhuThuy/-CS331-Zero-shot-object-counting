"""Summarize five paired A controls on all three locked development cohorts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .native_greedy import COHORTS

METHODS = {"native": "a_native", "proper_map": "proper_map",
           "proper_mbr": "proper_mbr", "guided_map": "guided_map",
           "guided_mbr": "guided_mbr"}


def summarize(evaluation_dir: Path) -> dict:
    result = {"cohorts": {}}
    for cohort, (_, expected_n) in COHORTS.items():
        entry = {"n": expected_n, "methods": {}}
        m0_reference = None
        for file_method, label in METHODS.items():
            path = evaluation_dir / f"{cohort}_{file_method}.json"
            data = json.loads(path.read_text())
            summary = data["summary"]
            if summary["cohort"] != cohort or summary["n"] != expected_n or \
                    len(data["per_item"]) != expected_n:
                raise ValueError(f"evaluation denominator mismatch: {path}")
            m0 = summary["methods"]["m0"]
            if m0_reference is None:
                m0_reference = m0
            elif m0_reference != m0:
                raise ValueError(f"M0 changed between paired evaluations: {cohort}")
            entry["methods"][file_method] = summary["methods"][label]
            entry["methods"][file_method]["delta_mae_vs_m0"] = summary[f"{label}_delta_mae_vs_m0"]
            entry["methods"][file_method]["delta_mae_ci95"] = summary[
                f"{label}_delta_mae_ci95_source_bootstrap"]
        entry["methods"]["m0"] = m0_reference
        result["cohorts"][cohort] = entry
    return result


def markdown_table(result: dict) -> str:
    lines = ["# MolmoPoint A development-cohort results", "",
             "Count and point metrics only. LVIS box IoU50 remains a separate contract.", "",
             "| Cohort | n | Method | MAE | RMSE | Exact | TP | FP | FN | False zero | Absent positive | ΔMAE vs M0 |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cohort, entry in result["cohorts"].items():
        for method in ("m0", *METHODS):
            metrics = entry["methods"][method]
            delta = metrics.get("delta_mae_vs_m0", 0)
            lines.append(f"| {cohort} | {entry['n']} | {method} | "
                         f"{metrics['mae']:.3f} | {metrics['rmse']:.3f} | "
                         f"{metrics['exact']} | {metrics['tp']} | {metrics['fp']} | "
                         f"{metrics['fn']} | {metrics['false_zero']} | "
                         f"{metrics['absent_positive']} | {delta:+.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.evaluation_dir)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    with args.markdown.open("x", encoding="utf-8") as stream:
        stream.write(markdown_table(result))
    print(markdown_table(result))


if __name__ == "__main__":
    main()
