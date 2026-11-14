"""Apply LVIS positive-retention thresholds to frozen absent111 top-5 scores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "target_aware_counting/output/fqie_h2_alpha_absent111_top5_20261104_v1"
CAL = ROOT / "target_aware_counting/output/fqie_h2_alpha_lvis_full_20261104_v1/evaluation.json"


def main():
    manifest = json.loads((RUN / "inference_manifest.json").read_text())
    freeze = json.loads((RUN / "freeze_scores.json").read_text())
    if hashlib.sha256((RUN / "inference_manifest.json").read_bytes()).hexdigest() != freeze["inference_manifest_sha256"]:
        raise RuntimeError("manifest changed after freeze")
    rows = []
    for item in manifest["items"]:
        rel = "scores/" + item["candidate_id"].replace(":", "_") + ".json"
        p = RUN / rel
        if hashlib.sha256(p.read_bytes()).hexdigest() != freeze["score_file_hashes"][rel]:
            raise RuntimeError("score changed after freeze")
        rows.append(json.loads(p.read_text()))
    calibration = json.loads(CAL.read_text())
    result = {"status": "DEVELOPMENT_TRANSFER_DIAGNOSTIC", "threshold_source": str(CAL),
              "selection": manifest["selection"], "queries": 111,
              "scored_candidates": len(rows), "total_candidates": manifest["total_frozen_candidates"],
              "unscored_candidates": manifest["unscored_candidates"], "metrics": {}}
    for source_field, score_field in (("native_score", "native_score"), ("alpha_full", "alpha_full")):
        points = []
        for op in calibration["metrics"][source_field]["operating_points"]:
            threshold = op["threshold"]
            kept = [r for r in rows if r[score_field] >= threshold]
            points.append({"LVIS_requested_target_retention": op["requested_target_retention"],
                           "threshold": threshold, "absent_candidates_kept": len(kept),
                           "absent_candidates_total_scored": len(rows),
                           "absent_queries_active": len({r["item_id"] for r in kept}),
                           "absent_top1_active": sum(r["native_rank"] == 1 for r in kept)})
        result["metrics"][source_field] = points
    result["limitations"] = [
        "All 111 queries are verified absent by the frozen cohort contract.",
        "Only top five native candidates per query were scored; one query has two candidates.",
        "The remaining 6390 candidates were not scored, so this does not measure full absent count mass.",
        "Thresholds come from strict IoU50 positives in a separate LVIS development cohort, not independent confirmation.",
        "FSC and LVIS score distributions may differ; no threshold is promoted from this transfer diagnostic."]
    out = RUN / "evaluation.json"
    payload = json.dumps(result, indent=2) + "\n"
    if out.exists() and out.read_text() != payload:
        raise RuntimeError("evaluation differs")
    out.write_text(payload)
    print(json.dumps(result["metrics"]))


if __name__ == "__main__":
    main()
