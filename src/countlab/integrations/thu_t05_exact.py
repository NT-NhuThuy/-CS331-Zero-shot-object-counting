"""C1/T05: strict, diagnostic-only V4/A3 join to an *identical* frozen pool.

A crosswalk is independent evidence, not reconstructed by matching integer indices
or spatial proximity. No GT, calibration, risk routing or count modification here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from countlab.method.schema import parse_input
from countlab.integrations.thu_features import attach_features

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
LABEL_KEYS = frozenset({
    "gt", "gt_count", "ground_truth", "ground_truth_count", "true_count",
    "target_count", "count_target", "label_count", "presence_label",
    "source_gt_count", "label_source", "annotation_label", "is_present",
    "correct_count", "pair_type", "evaluation_gt",
})
LABEL_PREFIX = re.compile(r"^(?:POS|HARD|ABSENT|PRESENT|NEGATIVE)__", re.I)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _no_gt(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in LABEL_KEYS:
                raise ValueError(f"GT/label forbidden at {path}.{key}")
            _no_gt(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _no_gt(child, f"{path}[{i}]")


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256")
    return value


def _projection(manifest: dict) -> tuple[dict, list[dict]]:
    _no_gt(manifest)
    baseline, rescues, _ = parse_input(manifest)
    meta = manifest.get("metadata")
    if not isinstance(meta, dict):
        raise ValueError("frozen pool needs metadata")
    image_sha = _hash(meta.get("image_sha256"), "image_sha256")
    protocol = meta.get("backend_protocol")
    if not isinstance(protocol, str) or not protocol.strip():
        raise ValueError("backend_protocol must be pinned")
    query_id = manifest["query_id"]
    if LABEL_PREFIX.match(str(query_id)):
        raise ValueError("query_id encodes presence label; use neutral ID")
    rows = []
    for role in ("baseline", "rescue"):
        for item in manifest.get(role, []):
            if "candidate_id" not in item or not isinstance(item["candidate_id"], str):
                raise ValueError("candidate ID must be a nonempty string")
            native_idx = item.get("native_query_index")
            if native_idx is not None and (type(native_idx) is not int or native_idx < 0):
                raise ValueError("native_query_index must be nonnegative integer")
            rows.append({
                "candidate_id": item["candidate_id"],
                "role": role,
                "view_id": item.get("view_id", "full"),
                "box_xyxy": list(item["box_xyxy"]),
                "native_query_index": native_idx,
            })
    rows.sort(key=lambda r: r["candidate_id"])
    if len(rows) != len(baseline) + len(rescues):
        raise ValueError("malformed candidate roster")
    source = {
        "image_id": manifest["image_id"],
        "query_id": query_id,
        "query_text_sha256": hashlib.sha256(manifest["query_text"].encode("utf-8")).hexdigest(),
        "image_sha256": image_sha,
        "backend_protocol": protocol,
    }
    return source, rows


def build_binding_request(manifest: dict) -> dict:
    """Make a GT-free crosswalk *request*, never a fabricated mapping."""
    source, rows = _projection(manifest)
    return {
        "schema": "thu_t05_exact_binding_request_v1",
        "source": source,
        "candidate_pool_sha256": canonical_sha(manifest),
        "candidate_identity_sha256": canonical_sha({"source": source, "candidates": rows}),
        "n_candidates": len(rows),
        "candidate_rows": [
            dict(row, node_graph_id=None, raw_query_index=None)
            for row in rows
        ],
        "binding_status": "BLOCKED_CROSSWALK_NOT_PROVIDED",
        "allowed_scope": "EXPOSED_REFERENCE_DIAGNOSTIC_ONLY",
        "required_actual_files": ["frozen_feature_npz", "crosswalk_evidence", "source_run_manifest"],
        "warning": "native_query_index is NOT a frozen V4/A3 node key; do not fill indices by position",
    }


def bind_exact(manifest: dict, binding: dict, npz: Path,
               crosswalk_evidence: Path, run_manifest: Path) -> tuple[dict, dict]:
    """Attach features iff file-locked and original-pixel candidate crosswalk matches.

    The evidence file's *semantic* provenance must still be independently audited.
    This code validates bytes, IDs, coordinates, scope and source metadata only.
    """
    request = build_binding_request(manifest)
    _no_gt(binding)
    if binding.get("schema") != "thu_t05_exact_binding_v1":
        raise ValueError("wrong T05 binding schema")
    if binding.get("source_cohort_status") != "EXPOSED_REFERENCE":
        raise ValueError("T05 currently supports exposed diagnostic binding only")
    for field in ("source", "candidate_pool_sha256", "candidate_identity_sha256"):
        if binding.get(field) != request[field]:
            raise ValueError(f"{field}: source/namespace/pool mismatch")
    for field, path in (("feature_npz_sha256", npz),
                        ("crosswalk_evidence_sha256", crosswalk_evidence),
                        ("source_run_manifest_sha256", run_manifest)):
        if _hash(binding.get(field), field) != file_sha(Path(path)):
            raise ValueError(f"{field}: source file hash mismatch")
    # Check actual run-manifest fields, not merely its declared hash.
    run_doc = _unique_json(Path(run_manifest))
    if (run_doc.get("image_sha256") != request["source"]["image_sha256"]
            or run_doc.get("query_text") != manifest["query_text"]
            or run_doc.get("protocol_id") != request["source"]["backend_protocol"]):
        raise ValueError("run manifest image/query/protocol mismatch")
    rows = binding.get("candidate_rows")
    expected = request["candidate_rows"]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("candidate mapping must cover exactly the frozen candidate IDs")
    by_id = {}
    keys = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(expected[0] if expected else {
                "candidate_id", "role", "view_id", "box_xyxy", "native_query_index",
                "node_graph_id", "raw_query_index"}):
            raise ValueError("invalid candidate row keys")
        cid = row["candidate_id"]
        if not isinstance(cid, str) or cid in by_id:
            raise ValueError("duplicate/invalid candidate ID")
        a, b = row["node_graph_id"], row["raw_query_index"]
        if any(type(x) is not int or x < 0 for x in (a, b)):
            raise ValueError("frozen node keys must be nonnegative integer (no guessed mapping)")
        if (a, b) in keys:
            raise ValueError("two candidate IDs map to the same frozen node")
        keys.add((a, b))
        by_id[cid] = row
    expected_by_id = {row["candidate_id"]: row for row in expected}
    if set(by_id) != set(expected_by_id):
        raise ValueError("candidate ID namespace mismatch")
    for cid, template in expected_by_id.items():
        row = by_id[cid]
        for field in ("role", "view_id", "native_query_index", "box_xyxy"):
            if row[field] != template[field]:
                raise ValueError(f"{cid}: {field} mismatch with original candidate")
    # Crosswalk evidence must independently carry the identical exact mapping.
    cross_doc = _unique_json(Path(crosswalk_evidence))
    _no_gt(cross_doc)
    if cross_doc.get("schema") != "thu_t05_crosswalk_evidence_v1":
        raise ValueError("crosswalk evidence schema mismatch")
    if (cross_doc.get("source") != request["source"]
            or cross_doc.get("candidate_identity_sha256") != request["candidate_identity_sha256"]
            or cross_doc.get("feature_npz_sha256") != binding["feature_npz_sha256"]
            or cross_doc.get("candidate_rows") != binding.get("candidate_rows")):
        raise ValueError("crosswalk evidence does not bind to source/features/candidate rows")
    candidate_map = {cid: {"node_graph_id": row["node_graph_id"],
                           "raw_query_index": row["raw_query_index"]}
                     for cid, row in by_id.items()}
    original_candidates = {role: manifest.get(role, []) for role in ("baseline", "rescue")}
    joined = attach_features(manifest, Path(npz), candidate_map, calibration=None)
    for role in ("baseline", "rescue"):
        if joined[role] != original_candidates[role]:
            raise ValueError("T05 must not mutate candidates, risk_hint or evidence")
    joined.setdefault("metadata", {})["thu_t05_exact_binding"] = {
        "status": "PASS_TECHNICAL_DIAGNOSTIC_ONLY",
        "binding_sha256": canonical_sha(binding),
        "candidate_identity_sha256": request["candidate_identity_sha256"],
        "crosswalk_evidence_sha256": binding["crosswalk_evidence_sha256"],
        "source_run_manifest_sha256": binding["source_run_manifest_sha256"],
        "g0_data_decision": "HOLD",
        "calibration_used": False,
        "transfer_or_clean_dev_certified": False,
    }
    _no_gt(joined)
    parse_input(joined)
    report = {
        "schema": "thu_t05_exact_binding_report_v1",
        "technical_status": "PASS_EXACT_JOIN",
        "n_candidates": len(rows),
        "candidate_pool_sha256": request["candidate_pool_sha256"],
        "bound_manifest_sha256": canonical_sha(joined),
        "feature_npz_sha256": binding["feature_npz_sha256"],
        "crosswalk_evidence_sha256": binding["crosswalk_evidence_sha256"],
        "g0_data_decision": "HOLD",
        "risk_hint_generated": False,
        "routing_changed": False,
        "count_changed": False,
        "independent_crosswalk_semantics_verified": False,
        "scope": "EXPOSED_REFERENCE_DIAGNOSTIC_ONLY",
    }
    return joined, report


def _unique_json(path: Path) -> dict:
    def strict(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError(f"duplicate JSON key {k}")
            d[k] = v
        return d
    out = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=strict,
                     parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"invalid constant {x}")))
    if not isinstance(out, dict):
        raise ValueError("JSON root must be object")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="C1 T05: GT-blind frozen CountGD++ -> exact V4/A3 diagnostic binding")
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=("request", "bind"), default="request")
    ap.add_argument("--binding", type=Path)
    ap.add_argument("--feature-npz", type=Path)
    ap.add_argument("--crosswalk-evidence", type=Path)
    ap.add_argument("--run-manifest", type=Path)
    a = ap.parse_args()
    if a.out_dir.exists():
        raise FileExistsError("create-only output directory required")
    manifest = _unique_json(a.pool)
    if a.mode == "request":
        request = build_binding_request(manifest)
        payload = {"binding_request.json": request}
        shown = {"technical_status": "BLOCKED_CROSSWALK_NOT_PROVIDED",
                 "n_candidates": request["n_candidates"], "g0_data_decision": "HOLD"}
    else:
        if any(x is None for x in (a.binding, a.feature_npz, a.crosswalk_evidence, a.run_manifest)):
            ap.error("bind requires --binding --feature-npz --crosswalk-evidence --run-manifest")
        bound, report = bind_exact(manifest, _unique_json(a.binding), a.feature_npz,
                                   a.crosswalk_evidence, a.run_manifest)
        payload = {"frozen_candidates_t05.json": bound, "t05_binding_report.json": report}
        shown = {"technical_status": report["technical_status"],
                 "n_candidates": report["n_candidates"], "g0_data_decision": "HOLD"}
    a.out_dir.mkdir(parents=True, exist_ok=False)
    for name, data in payload.items():
        with (a.out_dir / name).open("x", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
            f.write("\n")
    print(json.dumps(shown, indent=2))


if __name__ == "__main__":
    main()
