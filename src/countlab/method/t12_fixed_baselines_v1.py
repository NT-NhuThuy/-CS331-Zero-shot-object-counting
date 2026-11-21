"""T12/B5-B6 matched frozen-evidence CPU replay; no GPU calls or GT.

The schedule is built *solely* from a T03 candidate pool and a fixed B6 config.
The event bank is inspected only AFTER the complete schedule is frozen. A missed
observation never triggers replacement actions. B5 and B6 see the identical
scheduled observed evidence; extra bank observations are never consumed.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .baselines import BaselinePolicy, run_baseline_arm
from .joint_solver_v2 import validate_event_contract, validate_solver_config
from .schema import box_iou, parse_input
from .t03_contract_v1 import validate_candidate_pool
from .t09_delivery_v1 import run_t09_delivery
from .joint_solver_v2 import project_observed_evidence


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _fixed_keys(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} must have exactly {sorted(keys)}")


def validate_b5_config(config: dict) -> dict:
    _fixed_keys(config, {"schema", "policy"}, "B5_config")
    if config["schema"] != "countlab_t12_b5_config_v1":
        raise ValueError("invalid B5_config schema")
    policy = BaselinePolicy.parse(config["policy"]).to_dict()
    return {"schema": config["schema"], "policy": policy}


def validate_b6_config(config: dict) -> dict:
    _fixed_keys(config, {"schema", "verification_budget_calls", "max_pair_actions",
                         "min_pair_iou", "solver_config"}, "B6_config")
    if config["schema"] != "countlab_t12_b6_fixed_schedule_config_v1":
        raise ValueError("invalid B6_config schema")
    for key in ("verification_budget_calls", "max_pair_actions"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be nonnegative integer")
    threshold = config["min_pair_iou"]
    if type(threshold) not in (float, int) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("min_pair_iou must be finite in [0,1]")
    solver = validate_solver_config(config["solver_config"])
    return {"schema": config["schema"],
            "verification_budget_calls": config["verification_budget_calls"],
            "max_pair_actions": config["max_pair_actions"],
            "min_pair_iou": float(threshold), "solver_config": solver}


def _action_key(action: dict) -> tuple[str, ...]:
    if action["op"] == "PAIR_IDENTITY":
        a, b = sorted((action["a"], action["b"]))
        return ("PAIR_IDENTITY", a, b)
    return (action["op"], action["candidate_id"])


def _event_action(event: dict) -> dict:
    if event["op"] == "PAIR_IDENTITY":
        a, b = sorted((event["a"], event["b"]))
        return {"op": "PAIR_IDENTITY", "a": a, "b": b}
    return {"op": event["op"], "candidate_id": event["candidate_id"]}


def fixed_schedule(manifest: dict, b6_config: dict) -> list[dict]:
    """Freeze ordered action keys before receiving any observation/verdict.

    Unary actions use the same static risk/confidence ranking as C1 router;
    identity slots use geometry only for PRIORITIZATION, never as identity truth.
    """
    validate_candidate_pool(manifest)
    cfg = validate_b6_config(b6_config)
    baseline, rescue, _ = parse_input(manifest)
    candidates = baseline + rescue
    def priority(item):
        risk = item.risk_hint if item.risk_hint is not None else (
            1 - item.confidence if item.confidence is not None else 0.5)
        return (-risk, item.candidate_id)
    ordered = sorted(candidates, key=priority)
    unary = [{"op": op, "candidate_id": p.candidate_id}
             for p in ordered for op in ("REGION_SEMANTIC", "REGION_WHOLE")]
    pairs = []
    for i, a in enumerate(candidates):
        for b in candidates[i+1:]:
            overlap = box_iou(a.box, b.box)
            if overlap >= cfg["min_pair_iou"]:
                first, second = sorted((a.candidate_id, b.candidate_id))
                pairs.append((-overlap, first, second))
    pairs.sort()
    pair_actions = [{"op": "PAIR_IDENTITY", "a": a, "b": b}
                    for _, a, b in pairs[:cfg["max_pair_actions"]]]
    return (unary + pair_actions)[:cfg["verification_budget_calls"]]


def _validate_bank(manifest: dict, events: list[dict]) -> dict[tuple[str, ...], dict]:
    validate_event_contract(events)
    base, rescue, _ = parse_input(manifest)
    ids = {p.candidate_id for p in base + rescue}
    by_action: dict[tuple[str, ...], dict] = {}
    seen_refs: set[str] = set()
    for index, event in enumerate(events):
        ref = event["evidence_ref"]
        if not isinstance(ref, str) or not ref.strip() or ref in seen_refs:
            raise ValueError(f"event {index}: missing/repeated evidence_ref")
        seen_refs.add(ref)
        if event["op"] == "PAIR_IDENTITY":
            if event["a"] not in ids or event["b"] not in ids or event["a"] == event["b"]:
                raise ValueError(f"event {index}: unknown/self pair")
            if event["relation"] not in ("SAME", "DIFFERENT", "UNKNOWN"):
                raise ValueError(f"event {index}: invalid relation")
        else:
            if event["candidate_id"] not in ids:
                raise ValueError(f"event {index}: unknown candidate")
            if event["verdict"] not in ("YES", "NO", "UNKNOWN"):
                raise ValueError(f"event {index}: invalid unary verdict")
        key = _action_key(_event_action(event))
        if key in by_action:
            raise ValueError(f"event {index}: multiple observations for one fixed action; retries require a separate protocol")
        by_action[key] = event
    return by_action


def run_t12_matched(manifest: dict, observed_event_bank: list[dict], *,
                    b5_config: dict, b6_config: dict) -> dict:
    """Three GT-blind frozen-evidence arms, one candidate roster, one action set.

    Inference cannot certify actual call cost using an offline event bank.
    """
    validate_candidate_pool(manifest)
    cfg5 = validate_b5_config(b5_config)
    cfg6 = validate_b6_config(b6_config)
    schedule = fixed_schedule(manifest, cfg6)
    schedule_sha = _sha(schedule)
    bank = _validate_bank(manifest, observed_event_bank)
    observed: list[dict] = []
    action_trace = []
    for i, action in enumerate(schedule):
        event = bank.get(_action_key(action))
        action_trace.append({"slot": i, "action": action,
                             "observation_status": "OBSERVED" if event else "MISSING_UNKNOWN",
                             "evidence_ref": event["evidence_ref"] if event else None})
        if event is not None:
            observed.append(event)
    projected, conflicts = project_observed_evidence(manifest, observed)
    if conflicts:
        raise AssertionError("single-observation slots cannot produce conflicts")
    nms = run_baseline_arm(projected, "B5", policy=cfg5["policy"])
    complete = run_baseline_arm(projected, "B8", policy=cfg5["policy"])
    strict = run_t09_delivery(manifest, observed, fallback_count=manifest["baseline_count"],
                              config=cfg6["solver_config"])
    observed_refs = {event["evidence_ref"] for event in observed}
    if not all(t["evidence_ref"] is None or t["evidence_ref"] in observed_refs for t in action_trace):
        raise AssertionError("trace contains unobserved ref")

    def b5_arm(kind, result):
        return {"arm": kind, "decision_scope": "HEURISTIC_OBSERVED_POOL_NOT_SCENE_CERTIFIED",
                "count": result["count"], "evaluation_fallback_count": result["count"],
                "query_abstained": False, "predicted_components": result["predicted_components"],
                "accepted_candidate_ids": result["accepted_candidate_ids"],
                "suppressed_candidate_ids": result["suppressed_candidate_ids"],
                "unresolved_candidate_ids": result["unresolved_candidate_ids"],
                "used_observed_events_sha256": _sha(observed),
                "limitations": ["geometry does not prove physical-instance identity",
                                "zero over observed proposals is not certified scene absence"]}

    return {"schema": "countlab_t12_b5_b6_matched_v1",
            "image_id": manifest["image_id"], "query_id": manifest["query_id"],
            "source_visible_sha256": strict["source_visible_sha256"],
            "manifest_sha256": _sha(manifest),
            "b5_config_sha256": _sha(cfg5), "b6_config_sha256": _sha(cfg6),
            "fixed_schedule_sha256": schedule_sha,
            "observed_events_sha256": _sha(observed),
            "action_trace": action_trace,
            "cost_scope": {"budget_unit": "planned_evidence_action_slot",
                           "scheduled_slots": len(schedule), "observed_slots": len(observed),
                           "missing_slots": len(schedule) - len(observed),
                           "actual_model_calls": None, "measured_gpu_seconds": None,
                           "measured_tokens": None, "matched_to_b7_measured_cost": False},
            "arms": {"B5_NMS": b5_arm("B5_NMS", nms),
                     "B5_COMPLETE_LINK": b5_arm("B5_COMPLETE_LINK", complete),
                     "B6_FIXED": {"arm": "B6_FIXED", "count": strict["count"],
                                  "decision": strict["decision"],
                                  "evaluation_fallback_count": strict["evaluation_fallback_count"],
                                  "query_abstained": strict["count"] is None,
                                  "strict_result": strict,
                                  "used_observed_events_sha256": _sha(observed)}},
            "gt_used": False, "online_verifier_calls_executed": 0,
            "count_correction_authorized": False,
            "research_status": "CPU_CONTRACT_REPLAY_ONLY_PENDING_C2_COST_AND_B7"}
