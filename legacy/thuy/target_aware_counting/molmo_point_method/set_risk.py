"""Finite-bank point-set decisions in source-image pixel coordinates."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Hypothesis:
    points: tuple[tuple[float, float], ...]
    log_probability: float
    trace_id: str


def _validated(hypotheses: Sequence[Hypothesis]) -> None:
    if not hypotheses:
        raise ValueError("empty hypothesis bank")
    for h in hypotheses:
        if not math.isfinite(h.log_probability):
            raise ValueError("log probabilities must be finite, including stopping")
        if any(len(p) != 2 or not all(math.isfinite(x) for x in p) for p in h.points):
            raise ValueError("invalid source-image point")


def unique_sets(hypotheses: Sequence[Hypothesis]) -> list[Hypothesis]:
    """Deduplicate identical sets; retain their best complete sequence only.

    Search paths that rediscover one set must not multiply its posterior weight.
    Point order is irrelevant to set risk, while trace IDs preserve provenance.
    """
    _validated(hypotheses)
    best: dict[tuple[tuple[float, float], ...], Hypothesis] = {}
    for h in hypotheses:
        if len(set(h.points)) != len(h.points):
            raise ValueError("duplicate coordinates within a point set")
        key = tuple(sorted(h.points))
        if key not in best or h.log_probability > best[key].log_probability:
            best[key] = h
    return list(best.values())


def gospa(a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]],
          cutoff_pixels: float, order: float = 2.0) -> float:
    """GOSPA with alpha=2: missed/extra cost c^p/2 per point."""
    if not math.isfinite(cutoff_pixels) or cutoff_pixels <= 0 or order <= 0:
        raise ValueError("positive finite cutoff and order required")
    n, m = len(a), len(b)
    if not n or not m:
        return abs(n-m) * cutoff_pixels**order / 2
    costs = [[min(math.dist(x, y), cutoff_pixels)**order for y in b] for x in a]
    rows, cols = linear_sum_assignment(costs)
    return float(sum(costs[i][j] for i, j in zip(rows, cols)) +
                 abs(n-m) * cutoff_pixels**order / 2)


def decide(hypotheses: Sequence[Hypothesis], cutoff_pixels: float) -> dict:
    bank = unique_sets(hypotheses)
    pivot = max(h.log_probability for h in bank)
    unnormalized = [math.exp(h.log_probability-pivot) for h in bank]
    total = sum(unnormalized)
    weights = [w/total for w in unnormalized]
    # GOSPA is symmetric. Reuse each assignment for both expected-risk rows;
    # dense images can make a single assignment expensive.
    risks = [0.0] * len(bank)
    for i, candidate in enumerate(bank):
        for j in range(i+1, len(bank)):
            distance = gospa(candidate.points, bank[j].points, cutoff_pixels)
            risks[i] += weights[j] * distance
            risks[j] += weights[i] * distance
    map_index = max(range(len(bank)), key=lambda i: bank[i].log_probability)
    risk_index = min(range(len(bank)), key=lambda i: (risks[i], -bank[i].log_probability))
    return {
        "map_trace_id": bank[map_index].trace_id,
        "mbr_trace_id": bank[risk_index].trace_id,
        "map_points_xy": bank[map_index].points,
        "mbr_points_xy": bank[risk_index].points,
        "bank_size_unique": len(bank),
        "weights": {h.trace_id: w for h, w in zip(bank, weights)},
        "risks": {h.trace_id: r for h, r in zip(bank, risks)},
        "cutoff_pixels": cutoff_pixels,
        "weight_scope": "normalized_complete_sequences_in_truncated_bank",
    }
