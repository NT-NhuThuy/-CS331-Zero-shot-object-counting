"""Pure score helpers used after synchronized Alpha-CLIP feature extraction."""
from __future__ import annotations

import math


def margin(descriptor_cosine: float, query_cosine: float) -> float:
    if not all(math.isfinite(x) for x in (descriptor_cosine, query_cosine)):
        raise ValueError("nonfinite cosine")
    return descriptor_cosine - query_cosine


def conservative_margin(views: dict) -> float:
    return min(margin(views[v]["descriptor_cosine"], views[v]["query_cosine"])
               for v in ("full", "context"))
