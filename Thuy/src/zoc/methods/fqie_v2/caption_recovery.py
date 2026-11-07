"""Query-free recovery of leading descriptors from preserved caption text.

An extracted phrase is a syntactic observation, NOT a verified semantic label.
This does not modify v1 captions or supply contradiction labels to the gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .run import write_once

VERSION = "leading_descriptor_v3_20261105"
REJECT = re.compile(
    r"\b(?:multiple|several|many|two|three|four|five|six|group|cluster|pair|"
    r"part of|portion of|unclear|uncertain|possibly|maybe|appears to be|seems to be|not sure)\b", re.I)
PREFIX = re.compile(
    r"^(?:(?:the |this )?(?:main visible object|main object|object|indicated region|region) "
    r"(?:is|shows|contains|depicts)\s+)?(?:a|an|the)\s+", re.I)
BOUNDARY = re.compile(
    r"[;.!?]|\s+(?:with|that|which|featuring|standing|sitting|lying|wearing|holding|"
    r"positioned|located|shown|resting|flying|facing|perched|covered|topped|surrounded|"
    r"placed|parked|displayed|in|on|under|near|at|beside|behind|against|next to)\b", re.I)
MODIFIERS = set("red green blue white black brown pink purple yellow orange gray grey dark light pale bright "
                "circular spherical round square rectangular triangular metallic wooden plastic textured smooth "
                "glossy shiny fluffy curved straight small large long short tall thin thick deep vibrant".split())


def extract(raw: str, output_tokens: int) -> dict:
    def unknown(reason):
        return {"descriptor": None, "status": "UNKNOWN", "reason": reason}
    if not isinstance(raw, str) or not raw.strip():
        return unknown("empty")
    if output_tokens >= 96:
        return unknown("token_ceiling")
    if "<think>" in raw and "</think>" not in raw:
        return unknown("unfinished_think")
    text = raw.split("</think>")[-1].strip()
    if REJECT.search(text):
        return unknown("ambiguous_multiple_or_part")
    match = PREFIX.match(text)
    if not match:
        return unknown("unsupported_sentence_form")
    phrase = BOUNDARY.split(text[match.end():], maxsplit=1)[0].strip().lower()
    # Commas between adjectives belong to the noun phrase: 'circular, metallic cap'.
    phrase = re.sub(r"\s+", " ", phrase.replace(",", " ")).strip()
    if not re.fullmatch(r"[a-z]+(?:[- ][a-z]+){0,11}", phrase):
        return unknown("invalid_phrase")
    words = phrase.split()
    if any(w in {"a", "an", "the", "is", "are", "was", "were", "and", "or", "not", "no", "of"} for w in words):
        return unknown("clause_or_ambiguous_phrase")
    if words[-1] in MODIFIERS:
        return unknown("adjective_without_identified_noun")
    if words[-1] in {"object", "objects", "thing", "things", "item", "items", "region", "area", "image"}:
        return unknown("generic")
    return {"descriptor": phrase, "status": "DESCRIPTOR_EXTRACTED",
            "reason": "leading_phrase_only_semantics_unverified"}


def audit(caption_dir: Path, destination: Path) -> dict:
    # Snapshot exactly the files seen at entry; a running producer may add files later.
    paths = sorted(caption_dir.glob("*.json"))
    rows, source_hashes, statuses, examples = [], {}, Counter(), []
    for path in paths:
        blob = path.read_bytes()
        data = json.loads(blob)
        parsed = extract(data["raw_response"], data["output_tokens"])
        cid = data["candidate_id"]
        if cid in source_hashes:
            raise ValueError("duplicate candidate")
        source_hashes[cid] = hashlib.sha256(blob).hexdigest()
        rows.append({"candidate_id": cid, "source_caption_sha256": source_hashes[cid], **parsed})
        statuses[parsed["status"]] += 1
        if len(examples) < 20 and parsed["descriptor"]:
            examples.append({"raw": data["raw_response"], **parsed})
    report = {"parser_version": VERSION, "n": len(rows), "counts": dict(statuses),
              "old_valid": sum(json.loads(p.read_bytes()).get("caption_status") == "VALID" for p in paths),
              "interpretation": "syntactic coverage only; no GT read, semantic accuracy not measured",
              "examples": examples}
    write_once(destination / "source_snapshot.json", {"caption_hashes": source_hashes})
    write_once(destination / "descriptors.json", {"parser_version": VERSION, "items": rows})
    write_once(destination / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("captions", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = audit(args.captions, args.destination)
    print(json.dumps({k: result[k] for k in ("n", "counts", "old_valid", "interpretation")}))
