"""T06 G1: deterministic, GT-blind batch graph replay (CPU, observed evidence).

This boundary is additive and never changes the frozen T03 or T09 solver.
External handoffs are only *declared* until authenticated by C1/C2; all outputs
remain provisional, and no online CountGD/Qwen calls are executed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .t03_contract_v1 import _no_gt, validate_candidate_pool
from .joint_solver_v2 import validate_event_contract, validate_solver_config
from .t09_delivery_v1 import run_t09_delivery

ROW_SCHEMA = "countlab_t06_pilot_input_row_v1"
PRED_SCHEMA = "countlab_t06_pilot_prediction_v1"
TRACE_SCHEMA = "countlab_t06_graph_trace_v1"
CONFIG_SCHEMA = "countlab_t06_pilot_replay_config_v1"
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exact(obj: Any, keys: set[str], label: str) -> None:
    if not isinstance(obj, dict) or set(obj) != keys:
        raise ValueError(f"{label}: expected exact keys {sorted(keys)}")


def _sha(s: Any, label: str) -> None:
    if not isinstance(s, str) or SHA_PATTERN.fullmatch(s) is None:
        raise ValueError(f"{label}: expected lowercase SHA256")


def validate_config(config: dict) -> dict:
    _no_gt(config)
    _exact(config, {"schema", "solver_config", "claims_allowed"}, "T06 config")
    if config["schema"] != CONFIG_SCHEMA or config["claims_allowed"] is not False:
        raise ValueError("T06 config must disable scientific claims")
    return {"schema": CONFIG_SCHEMA, "solver_config": validate_solver_config(config["solver_config"]),
            "claims_allowed": False}


def _feature_join(join: Any, ids: set[str]) -> dict:
    if not isinstance(join, dict) or "status" not in join:
        raise ValueError("T05 feature_join: missing status")
    if join["status"] == "UNAVAILABLE":
        _exact(join, {"status"}, "T05 feature_join")
        return {"status": "UNAVAILABLE", "mapped_candidates": 0}
    if join["status"] != "EXACT_ID_JOIN_DECLARED_UNVERIFIED":
        raise ValueError("T05 feature_join: unsupported status")
    _exact(join, {"status", "feature_source_sha256", "candidate_map"}, "T05 feature_join")
    _sha(join["feature_source_sha256"], "T05 feature source")
    mp = join["candidate_map"]
    if not isinstance(mp, dict) or set(mp) != ids:
        raise ValueError("T05 feature join must match exactly all candidate IDs")
    node_keys = set()
    for cid, value in mp.items():
        _exact(value, {"node_graph_id", "raw_query_index"}, f"T05 candidate {cid}")
        key = (value["node_graph_id"], value["raw_query_index"])
        if any(type(v) is not int or v < 0 for v in key):
            raise ValueError("T05 exact keys must be nonnegative integers")
        if key in node_keys:
            raise ValueError("T05 duplicate node/raw-query mapping")
        node_keys.add(key)
    return {"status": join["status"], "mapped_candidates": len(mp),
            "feature_source_sha256": join["feature_source_sha256"],
            "candidate_map_sha256": sha_obj(mp)}


def validate_row(row: dict) -> dict:
    _no_gt(row)
    _exact(row, {"schema", "manifest", "observed_events", "feature_join", "provenance"}, "T06 row")
    if row["schema"] != ROW_SCHEMA:
        raise ValueError("unsupported T06 input schema")
    manifest = row["manifest"]
    audit = validate_candidate_pool(manifest)
    ids = {p["candidate_id"] for p in manifest["baseline"] + manifest["rescue"]}
    feature = _feature_join(row["feature_join"], ids)
    validate_event_contract(row["observed_events"])
    provenance = row["provenance"]
    if not isinstance(provenance, dict) or "mode" not in provenance:
        raise ValueError("T06 provenance must declare input scope")
    mode = provenance["mode"]
    if mode == "SYNTHETIC_CONTRACT":
        _exact(provenance, {"mode"}, "T06 synthetic provenance")
    elif mode == "EXTERNAL_HANDOFF_UNVERIFIED":
        _exact(provenance, {"mode", "source_image_sha256", "t04_pool_sha256", "t07_events_sha256"},
               "T06 external provenance")
        for k in ("source_image_sha256", "t04_pool_sha256", "t07_events_sha256"):
            _sha(provenance[k], k)
        metadata = manifest.get("metadata", {})
        if "image_sha256" in metadata and metadata["image_sha256"] != provenance["source_image_sha256"]:
            raise ValueError("T06 external source-image SHA mismatch")
        thu = metadata.get("thu_frozen_features")
        if isinstance(thu, dict) and "feature_sha256" in thu and feature.get("feature_source_sha256") is not None:
            if thu["feature_sha256"] != feature["feature_source_sha256"]:
                raise ValueError("T06 T05 feature SHA mismatch")
    else:
        raise ValueError("T06 scope must be SYNTHETIC_CONTRACT or EXTERNAL_HANDOFF_UNVERIFIED")
    return {"mode": mode, "feature_join": feature, "input_audit": audit}


def replay_row(row: dict, *, solver_config: dict) -> tuple[dict, dict]:
    validated = validate_row(row)
    m = row["manifest"]
    events = row["observed_events"]
    fallback = m["baseline_count"]
    decision = run_t09_delivery(m, events, fallback_count=fallback, config=solver_config)
    nodelist = []
    trace_by_id = {x["candidate_id"]: x for x in decision["candidate_trace"]}
    for role in ("baseline", "rescue"):
        for p in m[role]:
            cid = p["candidate_id"]
            nodelist.append({"candidate_id": cid, "role": role,
                             "view_id": p.get("view_id", "full"),
                             "box_xyxy": p["box_xyxy"],
                             "observed": trace_by_id[cid]})
    if len(nodelist) != len(trace_by_id):
        raise AssertionError("T06 node count mismatch")
    graph = {"schema": TRACE_SCHEMA, "image_id": m["image_id"], "query_id": m["query_id"],
             "input_row_sha256": sha_obj(row), "source_mode": validated["mode"],
             "manifest_sha256": sha_obj(m), "observed_events_sha256": sha_obj(events),
             "feature_join": validated["feature_join"],
             "source_provenance": row["provenance"],
             "nodes": nodelist, "observed_events": events,
             "physical_instances": decision["strict_result"]["physical_instances"],
             "blocking_issues": decision["blocking_issues"],
             "source_visible_sha256": decision["source_visible_sha256"],
             "solver_config_sha256": decision["solver_config_sha256"],
             "scene_zero_certified": False, "gt_used": False}
    if {p["candidate_id"] for p in nodelist} != set(trace_by_id):
        raise AssertionError("T06 missing trace node")
    observed_refs = {e["evidence_ref"] for e in events}
    for n in nodelist:
        if not set(n["observed"]["observed_evidence_refs"]).issubset(observed_refs):
            raise AssertionError("T06 trace references unobserved evidence")
    pred = {"schema": PRED_SCHEMA, "image_id": m["image_id"], "query_id": m["query_id"],
            "source_mode": validated["mode"], "input_row_sha256": sha_obj(row),
            "graph_trace_sha256": sha_obj(graph),
            "decision": decision["decision"], "count": decision["count"],
            "evaluation_fallback_count": decision["evaluation_fallback_count"],
            "observed_pool_decision": decision["observed_pool_decision"],
            "observed_pool_count": decision["observed_pool_count"],
            "n_baseline_candidates": len(m["baseline"]), "n_rescue_candidates": len(m["rescue"]),
            "n_observed_events": len(events), "n_candidate_trace_nodes": len(nodelist),
            "n_strict_instances": len(graph["physical_instances"]),
            "blocking_issues": decision["blocking_issues"],
            "source_visible_sha256": decision["source_visible_sha256"],
            "scene_zero_certified": False, "gt_used": False,
            "model_calls_by_T06": 0, "operational_count_correction_authorized": False,
            "handoff_status": "SYNTHETIC_CPU_CONTRACT_ONLY" if validated["mode"] == "SYNTHETIC_CONTRACT"
                             else "EXTERNAL_UNVERIFIED_PENDING_C1_C2_SIGNOFF"}
    if pred["count"] is None and pred["evaluation_fallback_count"] != fallback:
        raise AssertionError("T06 fallback cannot drift on ABSTAIN")
    if pred["decision"] == "K" and pred["count"] != pred["n_strict_instances"]:
        raise AssertionError("T06 K must be witnessed")
    return pred, graph


def replay_batch(rows: list[dict], config: dict) -> tuple[list[dict], list[dict], dict]:
    cfg = validate_config(config)
    if not isinstance(rows, list) or not rows:
        raise ValueError("T06 requires nonempty roster")
    seen = set()
    predictions, traces = [], []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("T06 roster rows must be JSON objects")
        m = row.get("manifest")
        key = (m.get("image_id"), m.get("query_id")) if isinstance(m, dict) else (None, None)
        if key in seen:
            raise ValueError("T06 duplicate image/query pair")
        seen.add(key)
        pred, trace = replay_row(row, solver_config=cfg["solver_config"])
        predictions.append(pred)
        traces.append(trace)
    count = len(predictions)
    assert len(traces) == len(rows) == count
    report = {"schema": "countlab_t06_pilot_replay_report_v1",
              "roster_pairs": count,
              "synthetic_pairs": sum(p["source_mode"] == "SYNTHETIC_CONTRACT" for p in predictions),
              "external_unverified_pairs": sum(p["source_mode"] == "EXTERNAL_HANDOFF_UNVERIFIED" for p in predictions),
              "counts": {s: sum(p["decision"] == s for p in predictions) for s in ("K", "ABSTAIN")},
              "unknown_events_not_inferred_negative": True,
              "real_handoff_accepted": False,
              "external_integration_status": "HOLD_PENDING_C1_T05_C2_T04_T07_SIGNOFF",
              "scientific_claims_allowed": False, "gt_used": False,
              "model_calls_by_T06": 0}
    return predictions, traces, report


def parse_jsonl(raw: bytes) -> list[dict]:
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError("T06 input JSONL must be UTF-8") from e
    records = []
    for line_number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"T06 blank row at line {line_number}")
        def unique_keys(pairs):
            result = {}
            for k, v in pairs:
                if k in result:
                    raise ValueError("duplicate JSON key: " + k)
                result[k] = v
            return result
        try:
            records.append(json.loads(line, object_pairs_hook=unique_keys,
                                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))))
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"T06 malformed JSON at line {line_number}") from e
    return records


def jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def report_markdown(summary: dict, *, input_sha: str, config_sha: str,
                    pred_sha: str, graph_sha: str) -> bytes:
    text = ("# T06 G1 — Pilot graph replay (CPU contract audit)\n\n"
            "**REAL PILOT: HOLD.** No signed T04/T05/T07 handoff or actual verifier call is authenticated here.\n"
            "Synthetic fixtures do not represent benchmark predictions.\n\n"
            f"- Input roster SHA256: `{input_sha}`\n"
            f"- Frozen T06 config SHA256: `{config_sha}`\n"
            f"- Predictions SHA256: `{pred_sha}`\n"
            f"- Graph trace SHA256: `{graph_sha}`\n"
            f"- Pairs: {summary['roster_pairs']} (synthetic {summary['synthetic_pairs']}; external unverified {summary['external_unverified_pairs']})\n"
            f"- Strict decisions: K={summary['counts']['K']}; ABSTAIN={summary['counts']['ABSTAIN']}\n"
            "- All candidate/event-only outputs are GT-blind and include original-view trace.\n"
            "- Empty candidate pool does not certify scene ZERO; fallback counts are evaluation-only.\n"
            "- `UNKNOWN` is not `NO`; coordinate IoU is not verified physical identity.\n"
            "- Operational count correction: not authorized.\n"
            "- Official integration: HOLD pending signed canonical candidate pool T04, feature join T05, and observed event logs T07.\n"
            "- No paired MAE, real GPU cost, source-disjoint or A* contribution claims.\n")
    return text.encode("utf-8")


def run_pilot_file(inputs: Path, config_path: Path, out_dir: Path) -> dict:
    if out_dir.exists():
        raise FileExistsError("T06 create-only output directory already exists")
    raw = inputs.read_bytes()
    config_raw = config_path.read_bytes()
    cfg = json.loads(config_raw.decode("utf-8-sig"))
    rows = parse_jsonl(raw)
    predictions, graphs, summary = replay_batch(rows, cfg)
    prediction_raw = jsonl_bytes(predictions)
    trace_raw = jsonl_bytes(graphs)
    report_raw = report_markdown(summary, input_sha=sha_bytes(raw), config_sha=sha_bytes(config_raw),
                                  pred_sha=sha_bytes(prediction_raw), graph_sha=sha_bytes(trace_raw))
    lock = {"schema": "countlab_t06_pilot_replay_lock_v1",
            "input_sha256": sha_bytes(raw), "config_sha256": sha_bytes(config_raw),
            "predictions_sha256": sha_bytes(prediction_raw), "graph_trace_sha256": sha_bytes(trace_raw),
            "report_sha256": sha_bytes(report_raw), "roster_pairs": summary["roster_pairs"],
            "gt_used": False, "actual_model_calls": 0,
            "real_integration_signed_off": False,
            "research_status": "CPU_CONTRACT_ONLY_REAL_PILOT_HOLD"}
    # Complete validation happens before creating the output directory; never drop a pair.
    out_dir.mkdir(parents=True, exist_ok=False)
    outputs = {"pilot_predictions.jsonl": prediction_raw, "graph_trace.jsonl": trace_raw,
               "pilot_replay_report.md": report_raw,
               "pilot_replay_lock.json": canonical_bytes(lock) + b"\n"}
    for name, data in outputs.items():
        with (out_dir / name).open("xb") as stream:
            stream.write(data)
    return lock
