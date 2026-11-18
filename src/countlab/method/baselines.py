"""Deterministic T1 strong-control replay arms (Thuy / P05).

These controls consume the same GT-blind, frozen candidate manifest as the
unified method.  They are deliberately CPU-only evidence replays: an
``UNKNOWN`` verdict means that a verification was unavailable, not NO.  The
returned count is consequently an observed-candidate decision, never a scene
ground-truth estimate or a calibrated experiment result.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from .identity_graph import arbitrate_identity_graph
from .router import select_for_verification
from .schema import Proposal, box_iou, parse_input
from .verifier import decision


ARM_IDS = frozenset(f"B{i}" for i in range(10))


@dataclass(frozen=True)
class BaselinePolicy:
    """Frozen replay settings; calibration belongs to an approved CAL split."""

    score_threshold: float = 0.5
    nms_iou: float = 0.5
    identity_min_same_iou: float = 0.10
    identity_ambiguity_iou: float = 0.30

    @classmethod
    def parse(cls, policy: dict[str, Any] | None) -> "BaselinePolicy":
        if policy is None:
            return cls()
        expected = {"score_threshold", "nms_iou", "identity_min_same_iou", "identity_ambiguity_iou"}
        if set(policy) != expected:
            raise ValueError("baseline policy must have EXACTLY the frozen threshold keys")
        values = {key: float(policy[key]) for key in expected}
        if any(not math.isfinite(value) or not 0 < value <= 1 for value in values.values()):
            raise ValueError("baseline policy thresholds must be finite values in (0, 1]")
        if values["identity_ambiguity_iou"] < values["identity_min_same_iou"]:
            raise ValueError("identity ambiguity IoU must be >= minimum SAME IoU")
        return cls(**values)

    def to_dict(self) -> dict[str, float]:
        return {key: getattr(self, key) for key in (
            "score_threshold", "nms_iou", "identity_min_same_iou", "identity_ambiguity_iou")}


def _nms(items: list[Proposal], threshold: float) -> tuple[list[Proposal], list[str]]:
    """Standard score NMS, with stable ID tie-breaks and explicit suppression."""
    ranked = sorted(items, key=lambda item: (-(item.confidence if item.confidence is not None else -1.0), item.candidate_id))
    kept: list[Proposal] = []
    suppressed: list[str] = []
    for item in ranked:
        if any(box_iou(item.box, chosen.box) >= threshold for chosen in kept):
            suppressed.append(item.candidate_id)
        else:
            kept.append(item)
    return kept, suppressed


def _eligible(items: list[Proposal], arm: str) -> list[Proposal]:
    if arm == "B2":
        return [item for item in items if item.semantic != "NO"]
    if arm == "B3":
        return [item for item in items if item.whole != "NO"]
    if arm in {"B4", "B5", "B7", "B8", "B9"}:
        return [item for item in items if decision(item.semantic, item.whole) != "REJECTED"]
    return items


def run_baseline_arm(data: dict[str, Any], arm: str, *, budget: int | None = None,
                     policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Replay one registered strong-control arm without changing v1 defaults.

    B0 is the frozen CountGD++ baseline proposal count.  B1--B5 are ordinary
    score/unary/NMS controls. B6 delegates to the incumbent sequential method.
    B7 uses intentionally naive transitive SAME union; B8 is the constrained
    complete-link counterpart. B9 is a fixed-slot routing replay (the router's
    deterministic risk/confidence order), not an online cost measurement.
    """
    if arm not in ARM_IDS:
        raise ValueError(f"unknown baseline arm: {arm}")
    frozen_policy = BaselinePolicy.parse(policy)
    baseline, rescues, edges = parse_input(data)
    input_sha = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if arm == "B6":
        # Kept as an isolated import to avoid a cycle at module import time.
        from .pipeline import run_method
        incumbent = run_method(data, budget=budget, policy={
            "baseline_rescue_match_iou": frozen_policy.nms_iou,
            "identity_min_same_iou": frozen_policy.identity_min_same_iou,
            "identity_overlap_ambiguity_iou": frozen_policy.identity_ambiguity_iou,
        })
        return {"schema": "countlab_t1_baseline_replay_v1", "arm": arm,
                "input_sha256": input_sha, "policy": frozen_policy.to_dict(),
                "selected_candidate_ids": sorted(x["candidate_id"] for x in incumbent["accepted_instances"]
                                         if "candidate_id" in x),
                "predicted_components": [[x["candidate_id"]] for x in incumbent["accepted_instances"]
                                         if "candidate_id" in x],
                "count": incumbent["count"], "query_abstained": incumbent["query_abstained"],
                "trace": {"delegated_to": "countlab_unified_method_output_v1"}, "incumbent": incumbent}

    all_items = baseline + rescues
    selected_ids = set(item.candidate_id for item in all_items)
    # B4 is the budgeted dual-NO control; B5 is its fully observed oracle
    # control.  Keeping this distinction explicit prevents an ablation from
    # silently comparing the same arm twice.
    evidence_mode = "all_precomputed"
    if arm in {"B4", "B9"} and budget is not None:
        selected_ids = select_for_verification(all_items, budget)
        all_items = [item if item.candidate_id in selected_ids else Proposal(
            item.candidate_id, item.role, item.view_id, item.box, item.confidence, item.risk_hint,
            "UNKNOWN", "UNKNOWN", item.evidence_refs) for item in all_items]
        evidence_mode = "budget_masked"
    if arm == "B0":
        kept = list(baseline)
        suppressed: list[str] = []
        unresolved: list[str] = []
        components = [[item.candidate_id] for item in kept]
    else:
        eligible = _eligible(all_items, arm)
        if arm == "B1":
            eligible = [item for item in eligible if item.confidence is not None and item.confidence >= frozen_policy.score_threshold]
        if arm == "B7":
            # Deliberately weak comparator: transitive SAME is allowed here.
            parent = {item.candidate_id: item.candidate_id for item in eligible}
            def find(candidate_id: str) -> str:
                while parent[candidate_id] != candidate_id:
                    parent[candidate_id] = parent[parent[candidate_id]]
                    candidate_id = parent[candidate_id]
                return candidate_id
            for (left, right), relation in sorted(edges.items()):
                if relation == "SAME" and left in parent and right in parent:
                    a, b = find(left), find(right)
                    if a != b:
                        parent[max(a, b)] = min(a, b)
            groups: dict[str, list[Proposal]] = {}
            for item in eligible:
                groups.setdefault(find(item.candidate_id), []).append(item)
            kept = [sorted(group, key=lambda item: item.candidate_id)[0] for _, group in sorted(groups.items())]
            components = [sorted(item.candidate_id for item in group) for _, group in sorted(groups.items())]
            suppressed = sorted(item.candidate_id for group in groups.values() for item in group[1:])
            unresolved = []
        elif arm == "B8":
            graph = arbitrate_identity_graph(eligible, edges,
                                              min_same_iou=frozen_policy.identity_min_same_iou,
                                              ambiguity_iou=frozen_policy.identity_ambiguity_iou)
            by_id = {item.candidate_id: item for item in eligible}
            kept = [by_id[group[0]] for group in graph.accepted_components]
            components = [list(group) for group in graph.accepted_components]
            unresolved = sorted(candidate_id for group in graph.unresolved_components for candidate_id in group)
            suppressed = sorted(candidate_id for group in graph.accepted_components for candidate_id in group[1:])
        else:
            kept, suppressed = _nms(eligible, frozen_policy.nms_iou)
            unresolved = []
            components = [[item.candidate_id] for item in kept]
    return {
        "schema": "countlab_t1_baseline_replay_v1", "arm": arm,
        "input_sha256": input_sha, "policy": frozen_policy.to_dict(), "budget": budget,
        "count": len(kept), "query_abstained": False,
        "selected_candidate_ids": sorted(selected_ids),
        "accepted_candidate_ids": sorted(item.candidate_id for item in kept),
        "predicted_components": components,
        "suppressed_candidate_ids": suppressed, "unresolved_candidate_ids": unresolved,
        "trace": {"evidence_mode": evidence_mode, "cost_scope": "not_online_billed_cost",
                  "control_pair": "B4_budgeted_vs_B5_all_verify"},
        "limitations": ["Not calibrated unless policy was selected on an approved CAL split",
                        "UNKNOWN evidence is never treated as NO", "Not a real-data effectiveness claim"],
    }
