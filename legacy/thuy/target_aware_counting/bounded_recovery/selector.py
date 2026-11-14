"""Frozen, GT-free selector for conservative recovery of empty CountGD outputs.

The default configuration exactly represents the rule transferred on FSC-147 VAL
and LVIS cohort C.  It consumes only prediction geometry and scores.  It does not
inspect dataset annotations or image/query identifiers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


Box = list[float]


@dataclass(frozen=True)
class RecoveryConfig:
    """Fixed global thresholds used by the validated bounded-recovery rule."""

    owl_score_floor: float = 0.10
    owl_match_iou: float = 0.50
    nms_iou: float = 0.50
    siglip_score_floor: float = 0.05
    max_siglip_fallback: int = 1

    def __post_init__(self) -> None:
        for name in ("owl_score_floor", "owl_match_iou", "nms_iou", "siglip_score_floor"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
        if self.max_siglip_fallback != 1:
            raise ValueError("validated bounded recovery requires exactly one SigLIP fallback slot")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _box(value: Sequence[float], *, field: str = "box") -> Box:
    if len(value) != 4:
        raise ValueError(f"{field} must contain four xyxy coordinates")
    result = [float(x) for x in value]
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{field} contains a non-finite coordinate")
    if result[2] < result[0] or result[3] < result[1]:
        raise ValueError(f"{field} must use xyxy order")
    return result


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Return IoU for two xyxy boxes, matching the locked audit implementation."""

    a, b = _box(first, field="first"), _box(second, field="second")
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    ranks: set[int] = set()
    for index, row in enumerate(rows):
        if "rank" not in row or "box" not in row:
            raise ValueError(f"candidate {index} requires rank and box")
        rank = int(row["rank"])
        if rank < 1 or rank in ranks:
            raise ValueError(f"candidate ranks must be unique positive integers, got {rank}")
        ranks.add(rank)
        item = {"rank": rank, "box": _box(row["box"], field=f"candidate[{rank}].box")}
        if "native_score" in row:
            native_score = float(row["native_score"])
            if not math.isfinite(native_score):
                raise ValueError(f"candidate {rank} has a non-finite native_score")
            item["native_score"] = native_score
        result.append(item)
    return result


def _owl_detections(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if "score" not in row or "box" not in row:
            raise ValueError(f"OWLv2 detection {index} requires score and box")
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError(f"OWLv2 detection {index} has a non-finite score")
        result.append({"score": score, "box": _box(row["box"], field=f"owl[{index}].box")})
    return result


def _siglip_score(scores: Mapping[int | str, float], rank: int) -> float:
    if rank in scores:
        value = scores[rank]
    elif str(rank) in scores:
        value = scores[str(rank)]
    else:
        raise ValueError(f"missing SigLIP score for candidate rank {rank}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"candidate rank {rank} has a non-finite SigLIP score")
    return result


def _owl_support(
    candidate_box: Box,
    detections: Sequence[Mapping[str, Any]],
    config: RecoveryConfig,
) -> dict[str, Any] | None:
    supported: list[dict[str, Any]] = []
    for detection in detections:
        score = float(detection["score"])
        overlap = box_iou(candidate_box, detection["box"])
        if score >= config.owl_score_floor and overlap >= config.owl_match_iou:
            supported.append({"score": score, "box": list(detection["box"]), "iou": overlap})
    if not supported:
        return None
    # The locked rule ranks candidates by maximum OWLv2 score.  Deterministic
    # tie-breaking below only enriches the explanation and does not change it.
    return sorted(supported, key=lambda row: (-row["score"], -row["iou"], row["box"]))[0]


def _nms(rows: Sequence[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (-item["score"], item["rank"])):
        if all(box_iou(row["box"], old["box"]) <= threshold for old in kept):
            kept.append(row)
    return kept


def select_owl_candidates(
    candidates: Sequence[Mapping[str, Any]],
    owl_detections: Sequence[Mapping[str, Any]],
    config: RecoveryConfig = RecoveryConfig(),
) -> list[dict[str, Any]]:
    """Select OWLv2-supported candidates and remove geometric duplicates."""

    candidate_rows = _candidates(candidates)
    detections = _owl_detections(owl_detections)
    pool: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        support = _owl_support(candidate["box"], detections, config)
        if support is None:
            continue
        item = {
            "box": candidate["box"],
            "rank": candidate["rank"],
            "score": support["score"],
            "reason": "owl_supported",
            "evidence": {
                "model": "owlv2",
                "score": support["score"],
                "matched_box": support["box"],
                "match_iou": support["iou"],
            },
        }
        if "native_score" in candidate:
            item["native_score"] = candidate["native_score"]
        pool.append(item)
    return _nms(pool, config.nms_iou)


def select_bounded_candidates(
    candidates: Sequence[Mapping[str, Any]],
    owl_detections: Sequence[Mapping[str, Any]],
    siglip_scores: Mapping[int | str, float],
    config: RecoveryConfig = RecoveryConfig(),
) -> list[dict[str, Any]]:
    """Apply OWLv2 primary selection, then at most one SigLIP fallback."""

    owl_selected = select_owl_candidates(candidates, owl_detections, config)
    if owl_selected:
        return owl_selected

    candidate_rows = _candidates(candidates)
    supported: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        score = _siglip_score(siglip_scores, candidate["rank"])
        if score < config.siglip_score_floor:
            continue
        item = {
            "box": candidate["box"],
            "rank": candidate["rank"],
            "score": score,
            "reason": "siglip_single_fallback",
            "evidence": {"model": "siglip2", "score": score},
        }
        if "native_score" in candidate:
            item["native_score"] = candidate["native_score"]
        supported.append(item)
    return sorted(supported, key=lambda item: (-item["score"], item["rank"]))[:1]


def apply_empty_recovery(
    baseline_boxes: Sequence[Sequence[float]],
    candidates: Sequence[Mapping[str, Any]] | None = None,
    owl_detections: Sequence[Mapping[str, Any]] | None = None,
    siglip_scores: Mapping[int | str, float] | None = None,
    config: RecoveryConfig = RecoveryConfig(),
) -> dict[str, Any]:
    """Return final regions while preserving every nonempty baseline verbatim.

    Auxiliary inputs are only required when the baseline is empty.  The returned
    count is always derived from the number of final regions.
    """

    baseline = [_box(box, field=f"baseline[{index}]") for index, box in enumerate(baseline_boxes)]
    if baseline:
        regions = [{"box": box, "reason": "countgd_baseline"} for box in baseline]
        return {"mode": "baseline", "count": len(regions), "regions": regions, "config": config.to_dict()}

    if candidates is None or owl_detections is None or siglip_scores is None:
        raise ValueError("empty baseline requires candidates, OWLv2 detections, and SigLIP scores")
    regions = select_bounded_candidates(candidates, owl_detections, siglip_scores, config)
    mode = regions[0]["reason"] if regions else "abstain"
    return {"mode": mode, "count": len(regions), "regions": regions, "config": config.to_dict()}
