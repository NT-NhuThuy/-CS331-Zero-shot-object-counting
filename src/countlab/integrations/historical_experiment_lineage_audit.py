"""Recover shareable, GT-free lineage facts from retained historical artifacts.

This is intentionally conservative: a filename, numeric image ID, or query
text never becomes an image hash or a source-disjoint assertion. For E00-E10
and V13, statuses are carried from the external inventory, not rescanned here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPERIMENT_IDS = [f"E{i:02d}" for i in range(11)] + ["V13", "V14"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError("artifact is outside workspace; refusing absolute-path disclosure") from exc


def v14_rows(*, private: Path, workspace: Path) -> tuple[list[dict], dict]:
    original = private / "outputs/v14_original_cpu_repro_20261114/bidirectional_output.json"
    imported = private / "outputs/v14_unicount_import_20261114"
    manifests, results = imported / "manifests", imported / "results"
    if not original.is_file() or not manifests.is_dir() or not results.is_dir():
        return [], {"experiment_id": "V14", "run_id": "v14_original_cpu_repro_20261114",
                    "evidence_status": "NOT_LOCATED", "detail": "required V14 artifact set is incomplete"}
    source = json.loads(original.read_text(encoding="utf-8"))
    queries = source.get("queries", {})
    rows = []
    for query_id, output in sorted(queries.items(), key=lambda item: int(item[0])):
        manifest = manifests / f"q_{hashlib.sha256(str(query_id).encode()).hexdigest()[:16]}.json"
        # The importer uses a stable query content hash rather than qid; find
        # its matching manifest by its embedded query_id without trusting name.
        matching = [path for path in manifests.glob("q_*.json")
                    if str(json.loads(path.read_text(encoding="utf-8")).get("query_id")) == str(query_id)]
        manifest = matching[0] if len(matching) == 1 else None
        image_id = None
        query_text = None
        for instance in output.get("accepted_instances", []) + output.get("rejected_regions", []) + output.get("unknown_regions", []):
            image_id = instance.get("source_image_id", image_id)
            query_text = instance.get("query_caption", query_text)
            if image_id is not None and query_text is not None:
                break
        if manifest is not None:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            image_id = payload.get("image_id", image_id)
            query_text = payload.get("query_text", query_text)
        result_paths = [path for path in results.glob("q_*.json")
                        if str(json.loads(path.read_text(encoding="utf-8")).get("query_id")) == str(query_id)]
        artifact_paths = [rel(original, workspace)]
        if manifest is not None: artifact_paths.append(rel(manifest, workspace))
        if len(result_paths) == 1: artifact_paths.append(rel(result_paths[0], workspace))
        rows.append({
            "experiment_id": "V14", "run_id": "v14_original_cpu_repro_20261114",
            "image_id": None if image_id is None else str(image_id),
            "image_sha256": None, "query_id": str(query_id), "query_text": query_text,
            "dataset": "LVIS-v1-val (claimed by legacy artifact; source image unverified)",
            "source_group": None, "artifact_path": artifact_paths,
            "evidence_status": "ARTIFACT_QUERY_VERIFIED_SOURCE_UNVERIFIED",
            "evidence": {"original_output_sha256": sha256(original),
                         "import_manifest_present": manifest is not None,
                         "per_query_result_present": len(result_paths) == 1,
                         "source_image_path": None,
                         "source_image_hash_evidence": "absent",
                         "task_index_sha256_claim": source.get("input_hashes", {}).get("task_index"),
                         "unary_sha256_claim": source.get("input_hashes", {}).get("unary_decisions")},
        })
    return rows, {"experiment_id": "V14", "run_id": "v14_original_cpu_repro_20261114",
                  "evidence_status": "PARTIAL_RECOVERED", "n_entries": len(rows),
                  "detail": "query/manifests/results recovered; source image paths, source hashes and task index bytes absent"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows, v14_summary = v14_rows(private=args.private, workspace=args.workspace)
    experiments = []
    for experiment_id in EXPERIMENT_IDS:
        if experiment_id == "V14":
            experiments.append(v14_summary)
        else:
            experiments.append({"experiment_id": experiment_id, "run_id": None,
                                "evidence_status": "NOT_LOCATED_IN_SCOPED_ROOTS",
                                "detail": "no manifest, task index, per-query output, or source-image lineage artifact located in ZOC/, ZOC_research/, UNICOUNT_PRIVATE/"})
    result = {"schema": "historical_experiment_lineage_audit_v1", "uses_gt": False,
              "shareable": True,
              "scope": ["ZOC/", "ZOC_research/ (if present)", "UNICOUNT_PRIVATE/"],
              "policy": {"source_disjoint_inference": "FORBIDDEN_FROM_FILENAME_OR_QUERY_ONLY",
                         "image_hash_required_for_cross_check": True,
                         "private_images_or_checkpoints_embedded": False},
              "experiments": experiments, "verified_entries": [], "missing_provenance_entries": rows,
              "summary": {"experiments_requested": EXPERIMENT_IDS, "experiments_with_recovered_entries": ["V14"] if rows else [],
                          "n_verified_entries": 0, "n_missing_provenance_entries": len(rows),
                          "g0_status": "BLOCKED_SOURCE_HASH_AND_HISTORICAL_SPLIT_USAGE_INCOMPLETE"}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
