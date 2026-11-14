"""CPU validation of frozen Alpha-CLIP smoke scores, with no GT access."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "target_aware_counting/output/fqie_h2_caption_smoke_20261104_v1"
DEST = ROOT / "target_aware_counting/output/fqie_h2_alpha_smoke_20261104_v2"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    lock = json.loads((DEST / "lock.json").read_text())
    manifest = json.loads((INPUT / "manifest.json").read_text())
    assert lock["input_manifest_sha256"] == sha(INPUT / "manifest.json")
    assert lock["caption_freeze_sha256"] == sha(INPUT / "freeze_raw.json")
    files = {}
    unknown = []
    for x in manifest["items"]:
        file = DEST / "scores" / f"{x['probe_id']}.json"
        row = json.loads(file.read_text())
        assert row["candidate_id"] == x["candidate_id"]
        assert row["image_sha256"] == x["image_sha256"]
        assert row["box_xyxy"] == x["box_xyxy"]
        assert row["semantic_decision"] == "NOT_MADE"
        assert set(row["views"]) == {"full", "context"}
        for view in row["views"].values():
            assert view["alpha_positive_pixels_after_transform"] > 0
            assert math.isfinite(view["target_cosine"])
            if row["descriptor_status"] == "AVAILABLE":
                assert math.isfinite(view["descriptor_cosine"])
                assert math.isfinite(view["descriptor_minus_target"])
            else:
                assert view["descriptor_cosine"] is None
                assert view["descriptor_minus_target"] is None
        if row["descriptor_status"] != "AVAILABLE":
            unknown.append(x["probe_id"])
        files[str(file.relative_to(DEST))] = sha(file)
    result = {"status": "COMPLETE_RAW_SCORES_NO_DECISION", "items": len(files),
              "unknown_descriptor_probe_ids": unknown,
              "lock_sha256": sha(DEST / "lock.json"), "score_file_hashes": files}
    out = DEST / "freeze_scores.json"
    content = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if out.exists() and out.read_bytes() != content:
        raise RuntimeError("raw score freeze differs")
    out.write_bytes(content)
    print(json.dumps({"items": len(files), "unknown": unknown,
                      "status": result["status"]}))


if __name__ == "__main__":
    main()
