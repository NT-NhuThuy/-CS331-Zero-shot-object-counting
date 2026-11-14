"""GT-free CountGD candidate lineage; raw F1 and final F2 remain distinct."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    box_xyxy_original: tuple[float, float, float, float]
    source_image_id: str
    query_id: str
    forward_id: str
    view_id: str
    slot_id: int | None
    native_score: float | None
    is_m0: bool

    def serializable(self) -> dict:
        return asdict(self)


def valid_box(box: tuple[float, float, float, float], image_hw: tuple[int, int]) -> bool:
    h, w = image_hw
    x0, y0, x1, y1 = box
    # Native CountGD rectangles can protrude by a pixel after integer rounding.
    # Preserve their coordinates for output; atom selection uses image overlap.
    return x0 < x1 and y0 < y1 and x1 > 0 and y1 > 0 and x0 < w and y0 < h


def tile_box_to_source(box: tuple[float, float, float, float],
                       source_rect: tuple[float, float, float, float],
                       tile_hw: tuple[int, int]) -> tuple[float, float, float, float]:
    """Invert a fixed view crop/resize; source_rect is in original image pixels."""
    x0, y0, x1, y1 = source_rect
    th, tw = tile_hw
    if th <= 0 or tw <= 0 or x1 <= x0 or y1 <= y0:
        raise ValueError("invalid tile geometry")
    sx, sy = (x1 - x0) / tw, (y1 - y0) / th
    return (x0 + box[0] * sx, y0 + box[1] * sy,
            x0 + box[2] * sx, y0 + box[3] * sy)


def action_pool(action_row: dict[str, Any], image_hw: tuple[int, int]) -> list[Candidate]:
    """Read the locked LVIS top-10/view plus every M0 F2 box, without GT."""
    q = action_row["query"]
    m0_ids = set(q["m0_ids"])
    records = q["candidates"]
    by_id = {}
    for record in records:
        candidate_id = str(record["candidate_id"])
        if candidate_id in by_id:
            if by_id[candidate_id] != record:
                raise ValueError(f"conflicting candidate ID {candidate_id}")
            continue
        by_id[candidate_id] = record
    if not m0_ids <= set(by_id):
        raise ValueError("M0 F2 IDs missing from candidate pool")
    if len(m0_ids) != int(q["m0_count"]):
        raise ValueError("M0 count and final box IDs disagree")
    result = []
    for candidate_id, record in by_id.items():
        box = tuple(float(v) for v in record["box_xyxy"])
        if len(box) != 4 or not valid_box(box, image_hw):
            raise ValueError(f"invalid original-image box {candidate_id}")
        is_m0 = candidate_id in m0_ids
        forward = str(record["forward"])
        if is_m0 and forward != "F2_M0":
            raise ValueError("M0 final box mislabeled as raw F1")
        if not is_m0 and forward != "F1_raw":
            raise ValueError("raw candidate mislabeled as M0 F2")
        result.append(Candidate(candidate_id, box, str(q["source_image_id"]),
                                str(q["query_id"]), forward, str(record["view"]),
                                int(record["slot_index"]) if record.get("slot_index") is not None else None,
                                float(record["native_score"]) if record.get("native_score") is not None else None,
                                is_m0))
    return sorted(result, key=lambda c: c.candidate_id)


def m0_prediction_pool(prediction: dict[str, Any], image_hw: tuple[int, int]) -> list[Candidate]:
    """Bind native F2 output when no raw F1 cache exists; caller must report missing raw coverage."""
    if prediction["variant"] != "M0" or prediction["count"] != len(prediction["final_boxes"]):
        raise ValueError("invalid frozen M0 prediction")
    result = []
    for record in prediction["final_boxes"]:
        box = tuple(float(v) for v in record["bbox_xyxy_float"])
        if not valid_box(box, image_hw):
            raise ValueError("invalid native F2 box")
        result.append(Candidate(str(record["source_id"]), box,
                                str(prediction["source_image_id"]), str(prediction["query_id"]),
                                "F2_M0", "full", int(record["slot"]), None, True))
    if len({c.candidate_id for c in result}) != len(result):
        raise ValueError("duplicate native F2 source ID")
    return result
