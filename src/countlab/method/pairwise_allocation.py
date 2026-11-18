"""Relation-blind allocation of pairwise identity verification budget.

The planner ranks *which pair to ask about* using only proposal role, unary
decision, geometry, confidence/risk and the requested budget.  It never reads
the answer stored in ``identity_edges``.  A caller may reveal only the answers
for selected pairs, which makes fixed-budget replay auditable.
"""
from __future__ import annotations

from typing import Iterable, Literal

from .schema import Proposal, box_iou


def select_pairwise_queries(baseline: Iterable[Proposal], rescues: Iterable[Proposal], states: dict[str, str], *,
                            budget: int | None, collision_iou: float,
                            identity_min_same_iou: float,
                            strategy: Literal["impact", "geometry", "risk"] = "impact") -> list[dict[str, object]]:
    """Select high-decision-value pairs without looking at relation outcomes."""
    if budget is not None and budget < 0:
        raise ValueError("pairwise verification budget must be non-negative")
    if strategy not in {"impact", "geometry", "risk"}:
        raise ValueError("unknown pairwise allocation strategy")
    baseline, rescues = list(baseline), list(rescues)
    proposals: list[tuple[int, float, float, tuple[str, str], str]] = []
    # A baseline/rescue collision can change the final count (ADD, COALESCE,
    # or UNRESOLVED), so it ranks above rescue-only duplicate resolution.
    for base in baseline:
        for rescue in rescues:
            iou = box_iou(base.box, rescue.box)
            if iou < collision_iou or states.get(rescue.candidate_id) != "ACCEPTED":
                continue
            utility = 4 if states.get(base.candidate_id) != "REJECTED" else 3
            risk = max(base.risk_hint if base.risk_hint is not None else 1 - (base.confidence or .5),
                       rescue.risk_hint if rescue.risk_hint is not None else 1 - (rescue.confidence or .5))
            proposals.append((utility, iou, risk, tuple(sorted((base.candidate_id, rescue.candidate_id))),
                              "baseline_rescue_collision"))
    # Rescue-rescue evidence can prevent duplicate additions. It is lower
    # value because it cannot by itself overturn a baseline-rescue collision.
    for index, left in enumerate(rescues):
        for right in rescues[index + 1:]:
            iou = box_iou(left.box, right.box)
            if (iou < identity_min_same_iou or states.get(left.candidate_id) != "ACCEPTED" or
                    states.get(right.candidate_id) != "ACCEPTED" or left.view_id == right.view_id):
                continue
            risk = max(left.risk_hint if left.risk_hint is not None else 1 - (left.confidence or .5),
                       right.risk_hint if right.risk_hint is not None else 1 - (right.confidence or .5))
            proposals.append((2, iou, risk, tuple(sorted((left.candidate_id, right.candidate_id))),
                              "rescue_duplicate_resolution"))
    if strategy == "impact":
        ranked = sorted(proposals, key=lambda row: (-row[0], -row[1], row[3]))
    elif strategy == "geometry":
        ranked = sorted(proposals, key=lambda row: (-row[1], row[3]))
    else:
        ranked = sorted(proposals, key=lambda row: (-row[2], row[3]))
    if budget is not None:
        ranked = ranked[:budget]
    return [{"pair": list(pair), "priority": priority, "iou": iou, "risk": risk,
             "purpose": purpose, "strategy": strategy}
            for priority, iou, risk, pair, purpose in ranked]
