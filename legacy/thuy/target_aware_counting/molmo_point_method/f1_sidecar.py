"""Read the locked, GT-free F1 proposal sidecar for either method branch."""
from __future__ import annotations

import json
from pathlib import Path


def load_f1_sidecar(cohort: str, item: dict) -> tuple[Path, list[dict], bool]:
    name = ("candidates_fsc_primary_dev100_full.jsonl"
            if cohort == "fsc_primary_dev100" else f"candidates_{cohort}.jsonl")
    path = Path(__file__).resolve().parents[1] / \
        f"output/reference_correspondence_rc01_20261029/{name}"
    with path.open(encoding="utf-8") as stream:
        control = next((row for line in stream if
                        (row := json.loads(line))["item_id"] == item["item_id"]), None)
    if control is None or control["source_image_id"] != item["source_image_id"] or \
            control["query"] != item["query"]:
        raise ValueError("F1 sidecar/manifest identity mismatch")
    f1 = [candidate for candidate in control["candidates"]
          if candidate["forward_id"] == "F1_raw"]
    available = bool(control["raw_pool_complete"] and f1)
    return path, f1, available
