"""Strict GT-blind schema for a unified baseline/recovery candidate pool."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math

VALID = {"YES", "NO", "UNKNOWN"}
RELATIONS = {"SAME", "DIFFERENT", "UNKNOWN"}
FORBIDDEN = {"gt", "ground_truth", "ground_truth_count", "gt_count", "true_count", "target_count", "label_count"}


def _validate_no_gt(obj: Any) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in FORBIDDEN:
                raise ValueError(f"GT/label field forbidden in inference input: {key}")
            _validate_no_gt(value)
    elif isinstance(obj, list):
        for x in obj:
            _validate_no_gt(x)


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x1 - x0) * max(0., y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.


@dataclass(frozen=True)
class Proposal:
    candidate_id: str
    role: str
    view_id: str
    box: tuple[float, float, float, float]
    confidence: float | None
    risk_hint: float | None
    semantic: str
    whole: str
    evidence_refs: tuple[str, ...]

    @classmethod
    def parse(cls, d: dict, role: str) -> "Proposal":
        if not isinstance(d, dict):
            raise ValueError("proposal must be object")
        candidate_id = str(d.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError("candidate_id is required")
        if role not in {"baseline", "rescue"}:
            raise ValueError("invalid role")
        box = d.get("box_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"{candidate_id}: box_xyxy must have four coordinates")
        box = tuple(float(v) for v in box)
        if not all(map(math.isfinite, box)) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"{candidate_id}: invalid original-coordinate box")
        evidence = d.get("evidence", {})
        if not isinstance(evidence, dict):
            raise ValueError("evidence must be object")
        semantic, whole = evidence.get("semantic", "UNKNOWN"), evidence.get("whole_instance", "UNKNOWN")
        if semantic not in VALID or whole not in VALID:
            raise ValueError(f"{candidate_id}: invalid unary evidence")
        confidence = d.get("confidence")
        risk = d.get("risk_hint")
        for name, v in (("confidence", confidence), ("risk_hint", risk)):
            if v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1):
                raise ValueError(f"{candidate_id}: invalid {name}")
        refs = d.get("evidence_refs", [])
        if not isinstance(refs, list) or not all(isinstance(x, str) for x in refs):
            raise ValueError("evidence_refs must be string list")
        return cls(candidate_id, role, str(d.get("view_id", "full")), box,
                   None if confidence is None else float(confidence),
                   None if risk is None else float(risk), semantic, whole, tuple(refs))

    def to_dict(self, decision: str) -> dict:
        return {"candidate_id": self.candidate_id, "role": self.role,
                "view_id": self.view_id, "box_xyxy": list(self.box),
                "semantic": self.semantic, "whole_instance": self.whole,
                "decision": decision, "evidence_refs": list(self.evidence_refs)}


def parse_input(data: dict) -> tuple[list[Proposal], list[Proposal], dict[tuple[str, str], str]]:
    if not isinstance(data, dict):
        raise ValueError("inference input must be JSON object")
    _validate_no_gt(data)
    if data.get("schema") != "countlab_candidate_pool_v1":
        raise ValueError("schema must be countlab_candidate_pool_v1")
    if not data.get("image_id") or not data.get("query_id") or not data.get("query_text"):
        raise ValueError("image_id, query_id, query_text required")
    base = [Proposal.parse(x, "baseline") for x in data.get("baseline", [])]
    rescue = [Proposal.parse(x, "rescue") for x in data.get("rescue", [])]
    ids = [x.candidate_id for x in base + rescue]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id must be unique across baseline and rescue")
    if data.get("baseline_count") != len(base):
        raise ValueError("v1 requires baseline_count == number of frozen baseline proposals; use fallback outside v1 for count-only modes")
    edges = {}
    for edge in data.get("identity_edges", []):
        a, b, rel = edge["a"], edge["b"], edge["relation"]
        if a not in ids or b not in ids or a == b or rel not in RELATIONS:
            raise ValueError("identity edge invalid")
        key = tuple(sorted((a, b)))
        if key in edges and edges[key] != rel:
            raise ValueError("conflicting identity edges")
        edges[key] = rel
    return base, rescue, edges
