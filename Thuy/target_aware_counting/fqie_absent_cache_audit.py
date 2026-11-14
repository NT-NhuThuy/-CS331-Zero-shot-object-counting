"""Audit frozen verified-absent cohorts separately; no model inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "target_aware_counting/output"
DEST = OUT / "fqie_gate0_20261104_v1/audit/absent_cache.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cohort(name, methods):
    path = OUT / name
    freeze = json.loads((path / "freeze.json").read_text())
    lock = json.loads((path / "input_lock.json").read_text())
    assert freeze["status"] == "FROZEN_BEFORE_GT"
    assert sha(path / "input_lock.json") == freeze["input_lock_sha256"]
    assert sha(path / "predictions.jsonl") == freeze["predictions_sha256"]
    assert sha(path / "input_manifest.jsonl") == lock["manifest_sha256"]
    labels = rows(path / "labels.jsonl")
    pred = rows(path / "predictions.jsonl")
    manifest = rows(path / "input_manifest.jsonl")
    assert len(labels) == len(pred) == len(manifest) == 100
    assert [x["item_id"] for x in labels] == [x["item_id"] for x in pred]
    assert [x["item_id"] for x in manifest] == [x["item_id"] for x in pred]
    assert len({x["source_image_id"] for x in manifest}) == 100
    assert all(x["status"] == "verified_absent" and x["gt_count"] == 0 and
               x["evidence"] == "LVIS_v1_val.explicit_neg_category_ids" for x in labels)
    result = {"cohort": name, "n": 100, "status": "verified_absent",
              "prediction_sha256": sha(path / "predictions.jsonl"), "methods": {}}
    for method, field in methods.items():
        counts = [int(x[field]) for x in pred]
        assert all(v >= 0 for v in counts)
        zero = [i for i, x in enumerate(pred) if x["m0_count"] == 0]
        result["methods"][method] = {
            "activation_all": sum(v > 0 for v in counts),
            "count_mass_all": sum(counts),
            "activation_on_m0_zero": sum(counts[i] > 0 for i in zero),
            "count_mass_on_m0_zero": sum(counts[i] for i in zero),
            "max_count": max(counts),
        }
    result["n_m0_zero"] = len(zero)
    result["n_m0_positive"] = 100 - len(zero)
    return result


if __name__ == "__main__":
    result = {
        "warning": "Separate LVIS cohorts; do not pool count mass or compare as paired.",
        "a6": cohort("a6_lvis_absent_safety_20261102_v1", {"M0": "m0_count", "A6": "a6_count"}),
        "a9": cohort("a9_lvis_absent_safety_20261103_v1", {"M0": "m0_count", "A4": "a4_count", "A9": "a9_count"}),
    }
    content = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if DEST.exists() and DEST.read_bytes() != content:
        raise RuntimeError(f"locked output differs: {DEST}")
    DEST.write_bytes(content)
    print(json.dumps(result, sort_keys=True))
