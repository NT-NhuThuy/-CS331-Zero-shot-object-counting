"""Small, intentionally conservative descriptor parser for frozen captions."""
from __future__ import annotations

import re


AMBIGUOUS = re.compile(r"\b(multiple|several|many|part of|portion of|unclear|uncertain|possibly|maybe|appears to be|not sure)\b", re.I)
NOUN = re.compile(r"^(?:a|an|the)\s+([a-z][a-z -]{0,60})[.!]?$", re.I)


def parse(raw: str, *, truncated: bool = False) -> tuple[str | None, str]:
    if truncated or not isinstance(raw, str) or not raw.strip():
        return None, "UNKNOWN"
    text = raw.strip()
    if AMBIGUOUS.search(text) or len(re.findall(r"[.!?]", text)) > 1:
        return None, "UNKNOWN"
    match = NOUN.fullmatch(text)
    if not match:
        return None, "UNKNOWN"
    noun = match.group(1).strip().lower()
    if noun in {"object", "thing", "item", "something", "region", "area"}:
        return None, "UNKNOWN"
    return noun, "VALID"


def compatibility(descriptor: str | None, query: str, *, known_disjoint: set[tuple[str, str]] | None = None) -> str:
    """Only an explicitly reviewed disjoint pair can contradict a query."""
    if not descriptor or not query.strip():
        return "UNKNOWN"
    a, b = descriptor.casefold().strip(), query.casefold().strip()
    if a == b or re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", b) or re.search(r"(?<!\w)" + re.escape(b) + r"(?!\w)", a):
        return "COMPATIBLE"
    return "CONTRADICTS" if tuple(sorted((a, b))) in (known_disjoint or set()) else "UNKNOWN"

