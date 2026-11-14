"""Owner: Lộc. Shared unary verification contract (precomputed CPU replay v1).

Replace PrecomputedUnaryVerifier with a frozen Qwen/other backend only after
model revision, prompts, crops, coordinate transforms and GPU parity are locked.
Unlike Lộc V7.6, this is an INSTANCE-LEVEL contract, not global class veto.
"""
from __future__ import annotations
from typing import Protocol
from .schema import Proposal


class UnaryVerifier(Protocol):
    def verdict(self, proposal: Proposal, *, selected: bool) -> tuple[str, str]: ...


class PrecomputedUnaryVerifier:
    def verdict(self, proposal: Proposal, *, selected: bool) -> tuple[str, str]:
        if not selected:
            return "UNKNOWN", "UNKNOWN"
        return proposal.semantic, proposal.whole


def decision(semantic: str, whole: str) -> str:
    if semantic == whole == "YES":
        return "ACCEPTED"
    if semantic == whole == "NO":
        return "REJECTED"
    return "UNKNOWN"
