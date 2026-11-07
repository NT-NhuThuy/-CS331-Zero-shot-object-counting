"""Conservative remove-only decision on frozen, query-free region evidence."""
from __future__ import annotations

import math


def decision(evidence: dict, tau: float) -> tuple[str, str]:
    if not math.isfinite(tau):
        raise ValueError("tau must be finite")
    if evidence.get("caption_status") != "VALID":
        return "KEEP_UNVERIFIED", "caption_invalid_or_missing"
    if evidence.get("compatibility") != "CONTRADICTS":
        return "KEEP_UNVERIFIED", "no_verified_contradiction"
    views = evidence.get("views", {})
    try:
        margins = [float(views[v]["descriptor_cosine"]) - float(views[v]["query_cosine"])
                   for v in ("full", "context")]
        pixels = [int(views[v]["alpha_pixels"]) for v in ("full", "context")]
    except (KeyError, TypeError, ValueError, OverflowError):
        return "KEEP_UNVERIFIED", "view_missing_or_invalid"
    if not all(math.isfinite(x) for x in margins) or min(pixels) <= 0:
        return "KEEP_UNVERIFIED", "view_missing_or_invalid"
    return ("REMOVE", "descriptor_wins_both_views") if min(margins) > tau else ("KEEP_UNVERIFIED", "margin_below_tau")


def filter_candidates(candidates: list[dict], evidence_by_id: dict, tau: float) -> tuple[list[dict], list[dict]]:
    ids = [x["candidate_id"] for x in candidates]
    if len(ids) != len(set(ids)) or set(evidence_by_id) != set(ids):
        raise ValueError("candidate/evidence binding mismatch")
    kept, ledger = [], []
    for item in candidates:
        cid = item["candidate_id"]
        action, reason = decision(evidence_by_id[cid], tau)
        ledger.append({"candidate_id": cid, "decision": action, "reason": reason})
        if action != "REMOVE":
            kept.append(item)
    return kept, ledger

