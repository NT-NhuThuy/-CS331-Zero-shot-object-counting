"""Join frozen model outputs into a complete GT-free decision evidence file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gpu import file_for, load_manifest
from .parse_descriptor import compatibility
from .run import sha, write_once


def build(manifest: Path, gpu_output: Path, disjoint_path: Path, destination: Path) -> dict:
    items = load_manifest(manifest)
    frozen = json.loads((gpu_output / "freeze.json").read_text())
    if frozen.get("status") != "FROZEN_BEFORE_LABEL_JOIN" or frozen["manifest_sha256"] != sha(manifest):
        raise RuntimeError("GPU evidence is not frozen for this manifest")
    if frozen["candidate_count"] != len(items):
        raise RuntimeError("frozen candidate count mismatch")
    disjoint = json.loads(disjoint_path.read_text())
    if disjoint.get("status") not in {"REVIEWED_DISJOINT_PAIRS", "NO_REVIEWED_DISJOINT_PAIRS"} or not isinstance(disjoint.get("pairs"), list):
        raise RuntimeError("disjoint pair file lacks review status")
    if disjoint["status"] == "NO_REVIEWED_DISJOINT_PAIRS" and disjoint["pairs"]:
        raise RuntimeError("unreviewed disjoint pairs are not permitted")
    pairs = set()
    for row in disjoint["pairs"]:
        if not all(row.get(k) for k in ("descriptor", "query", "reviewer_id", "review_date")):
            raise RuntimeError("incomplete reviewed disjoint pair")
        pair = tuple(sorted((row["descriptor"].casefold().strip(), row["query"].casefold().strip())))
        if pair[0] == pair[1]:
            raise RuntimeError("self-disjoint pair")
        pairs.add(pair)
    evidence = []
    for item in items:
        cid = item["candidate_id"]
        for stage in ("encodings", "captions", "scores"):
            path = file_for(gpu_output, stage, cid)
            if sha(path) != frozen["hashes"][stage][cid]:
                raise RuntimeError(f"frozen {stage} hash mismatch: {cid}")
        cp, sp = file_for(gpu_output, "captions", cid), file_for(gpu_output, "scores", cid)
        cap, score = json.loads(cp.read_text()), json.loads(sp.read_text())
        if cap["candidate_id"] != cid or score["candidate_id"] != cid or score["caption_sha256"] != sha(cp):
            raise RuntimeError(f"caption/score binding mismatch: {cid}")
        evidence.append({"candidate_id": cid, "caption_status": cap["caption_status"],
                         "descriptor": cap["descriptor"],
                         "compatibility": compatibility(cap["descriptor"], item["query"], known_disjoint=pairs),
                         "views": score["views"], "caption_sha256": sha(cp), "score_sha256": sha(sp)})
    payload = {"status": "FROZEN_GT_FREE_EVIDENCE", "manifest_sha256": sha(manifest),
               "gpu_freeze_sha256": sha(gpu_output / "freeze.json"),
               "reviewed_disjoint_sha256": sha(disjoint_path), "items": evidence}
    write_once(destination, payload)
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ("manifest", "gpu_output", "reviewed_disjoint", "destination"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    result = build(a.manifest, a.gpu_output, a.reviewed_disjoint, a.destination)
    print(json.dumps({"evidence_items": len(result["items"])}))


if __name__ == "__main__":
    main()
