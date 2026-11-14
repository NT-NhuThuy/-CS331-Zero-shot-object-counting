"""Actual native CountGD++ single-pass adapter from Thúy's preserved backend.

CRITICAL: legacy backend.py uses zero exemplars, one forward pass and NO
pseudo-exemplar feedback. It is not interchangeable with FQIE V14 or with the
official recommended CountGD++ pseudo-exemplar configuration.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROTOCOL_ID = "CGD_NATIVE_SINGLE_PASS_NO_PSEUDO_DIAGNOSTIC_V1"


def _tolist(value: Any) -> list:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def normalized_box_to_xyxy(box: list[float], width: int, height: int,
                           offset: tuple[float, float] = (0., 0.)) -> list[float] | None:
    """cxcywh normalized -> original-image xyxy, clip at image boundary."""
    cx, cy, bw, bh = map(float, box)
    if any(not __import__('math').isfinite(x) for x in (cx, cy, bw, bh)):
        raise ValueError("nonfinite model box")
    x0, y0 = max(0., (cx - bw/2) * width), max(0., (cy - bh/2) * height)
    x1, y1 = min(float(width), (cx + bw/2) * width), min(float(height), (cy + bh/2) * height)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0 + offset[0], y0 + offset[1], x1 + offset[0], y1 + offset[1]]


def proposals_from_prediction(prediction: dict, *, threshold: float, role: str,
                              view_id: str, offset: tuple[float, float]=(0.,0.),
                              image_id: str = "", source: str = PROTOCOL_ID) -> list[dict]:
    """Convert the real native backend's pred_boxes and native text mask.

    Preserves the model's score+content-token filter. The score threshold is
    specified at the frozen experimental operating point, never tuned on test.
    """
    if not 0 <= threshold <= 1 or role not in ("baseline", "rescue"):
        raise ValueError("invalid threshold/role")
    w, h = map(int, prediction["original_size"])
    boxes = _tolist(prediction["boxes_cxcywh_normalized"])
    scores = _tolist(prediction["native_scores"])
    passes = _tolist(prediction["native_text_pass"])
    if not len(boxes) == len(scores) == len(passes):
        raise ValueError("unaligned CountGD++ model outputs")
    out = []
    for i, (box, score, passed) in enumerate(zip(boxes, scores, passes)):
        score = float(score)
        if not 0 <= score <= 1:
            raise ValueError("score outside [0,1]")
        if not passed or score <= threshold:
            continue
        region = normalized_box_to_xyxy(box, w, h, offset)
        if region is None:
            continue
        out.append({"candidate_id": f"{role}:{view_id}:{i}", "view_id": view_id,
                    "box_xyxy": region, "confidence": score,
                    "evidence": {"semantic": "UNKNOWN", "whole_instance": "UNKNOWN"},
                    "evidence_refs": [f"{source}:{image_id}:{view_id}:{i}"]})
    return out


class NativeCountGD:
    """Wrap real preserved CountGDBackend; lazy-load torch and upstream source.

    Use dependency injection of a fake backend only for *software* tests.
    """
    def __init__(self, config: dict, *, backend: Any = None):
        self.config = dict(config)
        self.threshold = float(config.get("threshold", .23))
        if not 0 <= self.threshold <= 1:
            raise ValueError("invalid threshold")
        if backend is None:
            root = Path(__file__).resolve().parents[3]
            original = root / "legacy" / "thuy" / "src"
            if not (original / "zoc" / "backends" / "countgd_grounding" / "backend.py").exists():
                raise FileNotFoundError("Preserved Thuy backend missing; use repository checkout")
            sys.path.insert(0, str(original))
            from zoc.backends.countgd_grounding.backend import CountGDBackend
            backend = CountGDBackend({k: v for k, v in config.items() if k != "threshold"})
        self.backend = backend
        self.protocol_id = PROTOCOL_ID
        self.provenance = {"adapter": PROTOCOL_ID, "threshold": self.threshold,
                           "upstream": getattr(backend, "spec", {"status": "injected_test_double"})}

    def infer(self, image_path: Path, query: str, *, role: str = "baseline",
              view_id: str = "full", offset: tuple[float,float]=(0.,0.),
              image_id: str = "") -> list[dict]:
        raw = self.backend.predict(str(image_path), query)
        return proposals_from_prediction(raw, threshold=self.threshold, role=role,
                                         view_id=view_id, offset=offset, image_id=image_id)
