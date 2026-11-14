"""CPU integrity validation and freeze ledger for the bounded H2 caption smoke."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "target_aware_counting/output/fqie_h2_caption_smoke_20261104_v1"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = json.loads((DEST / "manifest.json").read_text())
    assert manifest["status"] == "LOCKED_DEVELOPMENT_SMOKE"
    assert len(manifest["items"]) == manifest["max_regions"] == 8
    file_hashes = {}
    token_ceiling = []
    for x in manifest["items"]:
        pid = x["probe_id"]
        epath = DEST / "encodings" / f"{pid}.json"
        cpath = DEST / "captions" / f"{pid}.json"
        e, c = json.loads(epath.read_text()), json.loads(cpath.read_text())
        assert e["candidate_id"] == c["candidate_id"] == x["candidate_id"]
        assert e["image_sha256"] == c["image_sha256"] == x["image_sha256"]
        assert e["source_box_xyxy"] == x["box_xyxy"]
        assert e["mask_origin"] == "adapter_rectangle_from_raw_box_not_object_mask"
        assert e["mask_tokens"] in c["prompt_text"]
        assert x["query_for_later_audit_only"].strip(" .").lower() not in c["prompt_text"].lower()
        assert 0 < c["output_tokens"] <= manifest["max_new_tokens"]
        assert isinstance(c["raw_response"], str) and c["raw_response"].strip()
        if c["output_tokens"] == manifest["max_new_tokens"]:
            token_ceiling.append(pid)
        file_hashes[str(epath.relative_to(DEST))] = sha(epath)
        file_hashes[str(cpath.relative_to(DEST))] = sha(cpath)
    result = {"status": "COMPLETE_RAW_EVIDENCE_NO_SEMANTIC_LABELS",
              "manifest_sha256": sha(DEST / "manifest.json"),
              "preflight_sha256": sha(DEST / "preflight_pass.json"),
              "items": 8, "token_ceiling_probe_ids": token_ceiling,
              "file_hashes": file_hashes}
    path = DEST / "freeze_raw.json"
    content = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError("frozen raw evidence changed")
    path.write_bytes(content)
    print(json.dumps({"items": 8, "token_ceiling": token_ceiling,
                      "status": result["status"]}))


if __name__ == "__main__":
    main()
