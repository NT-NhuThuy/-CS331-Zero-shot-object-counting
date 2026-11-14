"""Bridge from legacy FQIE frozen verification tasks to CountLab candidate manifests.

Input ONLY: bidirectional_task_index.json and identity_unary_decisions.jsonl.
No ground truth, eval CSV, target counts, or post-hoc thresholds are read.
Pairwise identity edges are deliberately omitted unless separately verified.
"""
from __future__ import annotations
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
from countlab.method.schema import parse_input, _validate_no_gt


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(index: dict, verdicts: list[dict], *, hashes: dict[str, str]) -> list[dict]:
    _validate_no_gt(index)
    _validate_no_gt(verdicts)
    if index.get("schema") != "fqie_bidirectional_verification_manifest_v1" or index.get("uses_gt") is not False:
        raise ValueError("expected GT-blind frozen FQIE task index")
    by_task = {}
    for v in verdicts:
        key = v.get("candidate_id", v.get("task_id"))
        if not key or key in by_task:
            raise ValueError("invalid/duplicate verdict id")
        by_task[key] = v
    grouped = defaultdict(lambda: {"baseline": [], "rescue": []})
    info = {}
    seen_tasks = set()
    for task in index["tasks"]:
        task_id = str(task["task_id"])
        if task_id in seen_tasks:
            raise ValueError("duplicate task_id")
        seen_tasks.add(task_id)
        role = task["role"]
        if role not in {"baseline", "rescue"}:
            raise ValueError("unknown task role")
        qid = str(task["query_id"])
        image_id = str(task["source_image_id"])
        query = str(task["query_caption"])
        if qid in info and info[qid] != (image_id, query):
            raise ValueError("inconsistent image/query metadata")
        info[qid] = (image_id, query)
        verdict = by_task.get(task_id, {})
        semantic = verdict.get("semantic", "UNKNOWN")
        whole = verdict.get("whole", verdict.get("whole_instance", "UNKNOWN"))
        score = task.get("proposal_score")
        # A source score is a ROUTING HEURISTIC, not calibrated probability.
        has_score = isinstance(score, (int, float)) and 0 <= score <= 1
        p = {"candidate_id": task_id, "view_id": "full" if role == "baseline" else str(task.get("view_id", "unknown-rescue-view")),
             "box_xyxy": list(task["box_xyxy"]), "evidence": {"semantic": semantic, "whole_instance": whole},
             "evidence_refs": list(task.get("evidence_refs", [])) + [f"unary_decisions:{task_id}"]}
        if has_score:
            p["confidence"] = float(score)
        grouped[qid][role].append(p)
    result = []
    for qid in sorted(grouped):
        image_id, query = info[qid]
        group = grouped[qid]
        manifest = {"schema": "countlab_candidate_pool_v1", "image_id": image_id,
                    "query_id": qid, "query_text": query,
                    "baseline_count": len(group["baseline"]),
                    "baseline": group["baseline"], "rescue": group["rescue"],
                    "identity_edges": [],
                    "metadata": {"origin": "legacy_fqie_bidirectional_verification",
                                 "source_artifact_hashes": hashes,
                                 "identity_state": "NOT_PROVIDED_DO_NOT_ASSUME_NO_DUPLICATES"}}
        parse_input(manifest)
        result.append(manifest)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--unary", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    idx = json.loads(args.index.read_text(encoding="utf-8"))
    lines = [json.loads(x) for x in args.unary.read_text(encoding="utf-8").splitlines() if x.strip()]
    outputs = convert(idx, lines, hashes={"index_sha256": sha256(args.index), "unary_sha256": sha256(args.unary)})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.out_dir / ("q_" + hashlib.sha256(x["query_id"].encode()).hexdigest()[:16] + ".json") for x in outputs]
    if any(p.exists() for p in paths):
        raise FileExistsError("a target manifest already exists; no overwrite allowed")
    for obj, path in zip(outputs, paths):
        with path.open("x", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")
    print(json.dumps({"queries": len(paths), "output_folder": str(args.out_dir), "source_hashes": outputs[0]["metadata"]["source_artifact_hashes"] if outputs else {}}, ensure_ascii=False))

if __name__ == "__main__":
    main()
