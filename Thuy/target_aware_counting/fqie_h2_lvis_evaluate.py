"""Evaluate frozen H2 scores at equal strict-positive retention."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "target_aware_counting/output/fqie_h2_alpha_lvis_full_20261104_v1"


def auc(pos, neg):
    wins = sum(a > b for a in pos for b in neg)
    ties = sum(a == b for a in pos for b in neg)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def operating(rows, field, retention):
    pos = sorted((r[field] for r in rows if r["eval_label"] == "TARGET_IOU50"), reverse=True)
    need = math.ceil(retention * len(pos))
    threshold = pos[need - 1]
    kept = [r for r in rows if r[field] >= threshold]
    kp = [r for r in kept if r["eval_label"] == "TARGET_IOU50"]
    kn = [r for r in kept if r["eval_label"] == "VERIFIED_ABSENT"]
    ku = [r for r in kept if r["eval_label"] == "UNKNOWN"]
    return {"requested_target_retention": retention, "threshold": threshold,
            "target_kept": len(kp), "target_total": len(pos),
            "target_retention": len(kp) / len(pos),
            "absent_candidates_kept": len(kn),
            "absent_candidates_total": sum(r["eval_label"] == "VERIFIED_ABSENT" for r in rows),
            "absent_query_pairs_active": len({r["query_image_id"] for r in kn}),
            "absent_source_images_active": len({r["source_image_id"] for r in kn}),
            "unknown_kept": len(ku),
            "unknown_total": sum(r["eval_label"] == "UNKNOWN" for r in rows)}


def main():
    freeze = json.loads((RUN / "freeze_scores.json").read_text())
    labels = json.loads((RUN / "eval_labels.json").read_text())
    manifest = json.loads((RUN / "inference_manifest.json").read_text())
    if freeze["inference_manifest_sha256"] != __import__("hashlib").sha256((RUN / "inference_manifest.json").read_bytes()).hexdigest():
        raise RuntimeError("manifest changed after raw-score freeze")
    label_by_id = {x["candidate_id"]: x for x in labels["items"]}
    rows = []
    for item in manifest["items"]:
        rel = "scores/" + item["candidate_id"].replace(":", "_") + ".json"
        p = RUN / rel
        if __import__("hashlib").sha256(p.read_bytes()).hexdigest() != freeze["score_file_hashes"][rel]:
            raise RuntimeError(f"raw score changed: {rel}")
        score = json.loads(p.read_text())
        lab = label_by_id[item["candidate_id"]]
        rows.append({**lab, "native_score": score["native_score"],
                     "alpha_context": score["views"]["context"]["target_cosine"],
                     "alpha_full": score["views"]["full"]["target_cosine"]})
    pos = [r for r in rows if r["eval_label"] == "TARGET_IOU50"]
    neg = [r for r in rows if r["eval_label"] == "VERIFIED_ABSENT"]
    result = {"status": "DEVELOPMENT_DIAGNOSTIC", "denominators": {
        "all_candidates": len(rows), "strict_target_iou50": len(pos),
        "verified_absent_candidates": len(neg),
        "unknown_present_candidates": sum(r["eval_label"] == "UNKNOWN" for r in rows),
        "verified_absent_query_pairs_with_predictions": len({r["query_image_id"] for r in neg}),
        "verified_absent_source_images_with_predictions": len({r["source_image_id"] for r in neg})},
        "metrics": {}, "interpretation_guard": [
            "Present proposals below IoU 0.5 are UNKNOWN and excluded from AUC/retention labels.",
            "Verified-absent denominator includes only query pairs where frozen M0 emitted candidates.",
            "Thresholds are descriptive development operating points, not a locked rule.",
            "Candidate-level values are clustered within source images and are not independent samples."]}
    for field in ("native_score", "alpha_context", "alpha_full"):
        result["metrics"][field] = {
            "auc_target_vs_verified_absent": auc([r[field] for r in pos], [r[field] for r in neg]),
            "target_mean": sum(r[field] for r in pos) / len(pos),
            "absent_mean": sum(r[field] for r in neg) / len(neg),
            "operating_points": [operating(rows, field, x) for x in (1.0, 0.95, 0.90)]}
    out = RUN / "evaluation.json"
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if out.exists() and out.read_text() != payload:
        raise RuntimeError("locked evaluation differs")
    out.write_text(payload)
    print(json.dumps(result["denominators"]))
    for field, value in result["metrics"].items():
        print(json.dumps({"field": field, "auc": value["auc_target_vs_verified_absent"],
                          "operating_points": value["operating_points"]}))


if __name__ == "__main__":
    main()
