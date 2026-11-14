"""Strict semantic verification and optional SAM region attachment.

This layer is applied only to recovered regions.  Baseline CountGD regions are
preserved.  The fixed score floor was developed on LVIS-D and transferred to
FSC-147 without retuning; it still requires confirmation on a fresh cohort.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class StrictVerifierConfig:
    siglip_tight_score_floor: float = 0.003

    def __post_init__(self) -> None:
        value = float(self.siglip_tight_score_floor)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("siglip_tight_score_floor must be finite and in [0, 1]")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _lookup(mapping: Mapping[int | str, Any], rank: int, field: str) -> Any:
    if rank in mapping:
        return mapping[rank]
    if str(rank) in mapping:
        return mapping[str(rank)]
    raise ValueError(f"missing {field} for recovered rank {rank}")


def _mask_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "mask_bbox_xyxy" in value:
        box = [float(number) for number in value["mask_bbox_xyxy"]]
        if len(box) != 4 or not all(math.isfinite(number) for number in box):
            raise ValueError("mask_bbox_xyxy must contain four finite coordinates")
        if box[2] < box[0] or box[3] < box[1]:
            raise ValueError("mask_bbox_xyxy must use xyxy order")
        result["box"] = box
    if "mask_path" in value:
        result["mask_path"] = str(value["mask_path"])
    if "mask_area" in value:
        area = int(value["mask_area"])
        if area < 0:
            raise ValueError("mask_area must be nonnegative")
        result["mask_area"] = area
    if "sam_score" in value:
        score = float(value["sam_score"])
        if not math.isfinite(score):
            raise ValueError("sam_score must be finite")
        result["sam_score"] = score
    return result


def apply_strict_verification(
    decision: Mapping[str, Any],
    tight_siglip_scores: Mapping[int | str, float] | None = None,
    masks_by_rank: Mapping[int | str, Mapping[str, Any]] | None = None,
    config: StrictVerifierConfig = StrictVerifierConfig(),
) -> dict[str, Any]:
    """Filter recovered regions and optionally replace boxes with SAM tight boxes.

    ``masks_by_rank`` only changes the reported geometry and explanation after a
    region passes the semantic score.  It never changes which rank is accepted.
    """

    result = deepcopy(dict(decision))
    regions = result.get("regions")
    if not isinstance(regions, list) or int(result.get("count", -1)) != len(regions):
        raise ValueError("decision must contain regions with count parity")
    if result.get("mode") == "baseline":
        result["profile"] = "strict"
        result["strict_verifier"] = config.to_dict()
        return result
    if not regions:
        result["mode"] = "abstain"
        result["profile"] = "strict"
        result["strict_verifier"] = config.to_dict()
        return result
    if tight_siglip_scores is None:
        raise ValueError("recovered regions require SAM-tight SigLIP scores")

    accepted: list[dict[str, Any]] = []
    for region in regions:
        if "rank" not in region:
            raise ValueError("a recovered region requires rank")
        rank = int(region["rank"])
        score = float(_lookup(tight_siglip_scores, rank, "SAM-tight SigLIP score"))
        if not math.isfinite(score):
            raise ValueError(f"rank {rank} has a non-finite SAM-tight SigLIP score")
        if score < config.siglip_tight_score_floor:
            continue
        kept = deepcopy(region)
        kept["verification"] = {
            "model": "siglip2",
            "view": "sam_tight_raw_rgb",
            "score": score,
            "floor": config.siglip_tight_score_floor,
        }
        if masks_by_rank is not None:
            metadata = _mask_metadata(_lookup(masks_by_rank, rank, "SAM mask metadata"))
            if "box" in metadata:
                kept["proposal_box"] = kept["box"]
                kept["box"] = metadata.pop("box")
            kept["region"] = {"type": "sam2_mask", **metadata}
        accepted.append(kept)

    result["regions"] = accepted
    result["count"] = len(accepted)
    result["mode"] = "strict_recovery" if accepted else "abstain"
    result["profile"] = "strict"
    result["strict_verifier"] = config.to_dict()
    return result
