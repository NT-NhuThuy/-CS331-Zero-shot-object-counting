"""Shared evidence schemas and validation: conservative, not truth-certifying.

The intervals are OBSERVED-CANDIDATE intervals, not mathematically certified
bounds on all ground-truth instances in an image. This distinction is essential.
"""
from dataclasses import dataclass
from typing import Any, Mapping

STATES = frozenset({"YES", "NO", "UNKNOWN"})
DECISIONS = frozenset({"ACCEPTED", "REJECTED", "UNKNOWN"})
STATUSES = frozenset({"exact", "partial", "abstained"})


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_id: str
    image_id: str
    query_id: str
    role: str
    semantic: str
    whole_instance: str
    decision: str
    box_xyxy: tuple[float, float, float, float] | None = None
    identity_component_id: str | None = None
    source_view: str | None = None
    evidence_refs: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.candidate_id or not self.image_id or not self.query_id:
            raise ValueError("candidate_id, image_id, query_id required")
        if self.semantic not in STATES or self.whole_instance not in STATES:
            raise ValueError("invalid semantic / whole-instance verdict")
        if self.decision not in DECISIONS:
            raise ValueError("invalid decision")
        if self.decision == "ACCEPTED" and (self.semantic, self.whole_instance) != ("YES", "YES"):
            raise ValueError("ACCEPTED requires semantic=YES AND whole_instance=YES")
        if self.role not in {"baseline", "rescue"}:
            raise ValueError("candidate role must be baseline or rescue")
        if self.box_xyxy is not None:
            if len(self.box_xyxy) != 4 or not (self.box_xyxy[2] > self.box_xyxy[0] and self.box_xyxy[3] > self.box_xyxy[1]):
                raise ValueError("invalid original-coordinate box")


def validate_instance_output(o: Mapping[str, Any]) -> list[str]:
    """Reject impossible schema; warn on epistemic gaps not inferable from schema.

    Does NOT establish that instance labels, count intervals or absence are true.
    """
    required = {"count", "count_lower_bound", "count_upper_bound", "count_status",
                "accepted_instances", "rejected_regions", "unknown_regions", "evidence_trace", "query_abstained"}
    missing = required - set(o)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    s = str(o["count_status"])
    if s not in STATUSES:
        raise ValueError("invalid count_status")
    lo, hi = o["count_lower_bound"], o["count_upper_bound"]
    if not (isinstance(lo, int) and isinstance(hi, int) and 0 <= lo <= hi):
        raise ValueError("count interval must have integer 0 <= L <= U")
    count = o["count"]
    if count is not None and (not isinstance(count, int) or not lo <= count <= hi):
        raise ValueError("count must lie in interval, unless withheld (None)")
    if s == "exact" and (lo != hi or o["unknown_regions"]):
        raise ValueError("exact conflicts with interval or UNKNOWN regions")
    if s == "abstained" and not o["query_abstained"]:
        raise ValueError("abstained status requires query_abstained")
    if not all(isinstance(o[k], list) for k in ("accepted_instances", "rejected_regions", "unknown_regions", "evidence_trace")):
        raise ValueError("evidence fields must be lists")
    warnings = []
    if s == "exact" and lo == 0:
        warnings.append("exact-zero is an output-state label, not certified target absence")
    if not o["evidence_trace"]:
        warnings.append("empty evidence trace: output is not externally auditable")
    return warnings


def decision_from_unary(semantic: str, whole_instance: str) -> str:
    """The V14 conservative two-NO rule; contract utility, not a new verifier."""
    if semantic not in STATES or whole_instance not in STATES:
        raise ValueError("verdict must be YES / NO / UNKNOWN")
    if semantic == whole_instance == "YES":
        return "ACCEPTED"
    if semantic == whole_instance == "NO":
        return "REJECTED"
    return "UNKNOWN"
