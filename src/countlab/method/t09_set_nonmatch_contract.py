"""Experimental C3 set-level nonmatch evidence contract (NOT a counting method).

This module NEVER changes an operational count. A 'supportable' decision means
*conditional on the correctness of the independently gated visual observations*.
Use separate T06 validation before adding refs to `gated_evidence_refs`.

The query-level universe digest fixes identity and box geometry but DOES NOT
by itself attest the historical detector or authenticate an evidence ref.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

_HASH_CHARS = frozenset("0123456789abcdef")
_UNARY = frozenset({"REGION_SEMANTIC", "REGION_WHOLE"})


def _hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HASH_CHARS for c in value)


def _candidate_ids(items: Any, label: str) -> list[str]:
    if not isinstance(items, list):
        raise ValueError(f"{label}_NOT_LIST")
    ids = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("candidate_id"), str) or not item["candidate_id"]:
            raise ValueError(f"{label}_BAD_CANDIDATE")
        ids.append(item["candidate_id"])
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}_DUPLICATED_IDS")
    return sorted(ids)


def universe_sha256(manifest: dict, source_image_sha256: str) -> str:
    """Bind the full candidate universe, image and prompt; no GT fields read."""
    if not _hex64(source_image_sha256):
        raise ValueError("BAD_IMAGE_HASH")
    if not isinstance(manifest, dict):
        raise ValueError("BAD_MANIFEST")
    if not all(isinstance(manifest.get(x), str) and manifest[x] for x in ("image_id", "query_id", "query_text")):
        raise ValueError("MISSING_QUERY_IDENTITY")
    baseline = manifest.get("baseline")
    rescue = manifest.get("rescue", [])
    b_ids = _candidate_ids(baseline, "BASELINE")
    r_ids = _candidate_ids(rescue, "RESCUE")
    if set(b_ids) & set(r_ids):
        raise ValueError("CROSS_POOL_COLLIDING_IDS")
    if manifest.get("baseline_count") is not None and (type(manifest["baseline_count"]) is not int or manifest["baseline_count"] != len(b_ids)):
        raise ValueError("BASELINE_COUNT_MISMATCH")
    def view(items: list[dict]) -> list[dict]:
        return sorted(({
            "candidate_id": x["candidate_id"],
            "box_xyxy": x.get("box_xyxy"),
            "view_id": x.get("view_id"),
        } for x in items), key=lambda x: x["candidate_id"])
    return _hash({
        "schema": "c3_set_nonmatch_universe_v0",
        "source_image_sha256": source_image_sha256,
        "image_id": manifest["image_id"],
        "query_id": manifest["query_id"],
        "query_text": manifest["query_text"],
        "baseline": view(baseline),
        "rescue": view(rescue),
    })


def audit_set_nonmatch(
    manifest: dict,
    events: list[dict],
    set_observations: list[dict],
    *,
    source_image_sha256: str,
    gated_evidence_refs: set[str] | frozenset[str],
    max_group_size: int = 8,
) -> dict:
    """Audit one or more rescues; NEVER emit a forced/operational count.

    New externally observed SET_IDENTITY event schema:
      op=SET_IDENTITY, rescue_candidate_id, baseline_candidate_ids (exact list),
      verdict=NONE_MATCH|SOME_MATCH|UNKNOWN, evidence_ref, universe_sha256,
      source_image_sha256, query_id.

    Source/raw-response/prompt hashes and gate decisions must be checked by T06,
    which passes independently verified evidence refs into this function.
    A NONE_MATCH verdict is a set-level model assertion, NOT automatically a
    mathematically authenticated visual fact.
    """
    if type(max_group_size) is not int or max_group_size < 1:
        raise ValueError("INVALID_MAX_GROUP_SIZE")
    if not isinstance(gated_evidence_refs, (set, frozenset)):
        raise ValueError("GATE_REFS_REQUIRED")
    if not isinstance(events, list) or not isinstance(set_observations, list):
        raise ValueError("EVENTS_NOT_LIST")
    b_ids = _candidate_ids(manifest.get("baseline"), "BASELINE")
    r_ids = _candidate_ids(manifest.get("rescue", []), "RESCUE")
    if set(b_ids) & set(r_ids):
        raise ValueError("CROSS_POOL_COLLIDING_IDS")
    if manifest.get("baseline_count") is not None and (type(manifest["baseline_count"]) is not int or manifest["baseline_count"] != len(b_ids)):
        raise ValueError("BASELINE_COUNT_MISMATCH")
    universe = universe_sha256(manifest, source_image_sha256)
    all_ids = set(b_ids) | set(r_ids)
    unary = defaultdict(lambda: defaultdict(set))
    unary_refs = defaultdict(set)
    same_with_baseline = set()
    linked_rescues = set()
    direct_pair_ref = defaultdict(set)
    seen_refs = set()
    errors = []

    def checked_ref(item, idx):
        ref = item.get("evidence_ref")
        if not isinstance(ref, str) or not ref.strip() or ref in seen_refs:
            raise ValueError(f"INVALID_OR_DUPLICATE_EVIDENCE_REF:{idx}")
        seen_refs.add(ref)
        return ref

    for i, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError("EVENT_NOT_OBJECT")
        ref = checked_ref(event, f"event:{i}")
        if ref not in gated_evidence_refs:
            errors.append("UNGATED_EVENT:" + ref)
            continue
        op = event.get("op")
        if op in _UNARY:
            cid = event.get("candidate_id")
            val = event.get("verdict")
            if cid not in all_ids or val not in {"YES", "NO", "UNKNOWN"}:
                raise ValueError("INVALID_UNARY_EVENT")
            unary[cid][op].add(val)
            unary_refs[cid].add(ref)
        elif op == "PAIR_IDENTITY":
            a, b, rel = event.get("a"), event.get("b"), event.get("relation")
            if a not in all_ids or b not in all_ids or a == b or rel not in {"SAME", "DIFFERENT", "UNKNOWN"}:
                raise ValueError("INVALID_PAIR_EVENT")
            direct_pair_ref[tuple(sorted((a, b)))].add((rel, ref))
            if rel == "SAME":
                if {a, b} & set(r_ids) and {a, b} & set(b_ids):
                    same_with_baseline.add(next(x for x in (a, b) if x in r_ids))
                elif a in r_ids and b in r_ids:
                    linked_rescues.update((a, b))
        else:
            raise ValueError("UNSUPPORTED_EXISTING_EVENT_OP")

    none_sets = defaultdict(list)
    blocked_by_some = set()
    refs_by_rescue = defaultdict(set)
    for i, event in enumerate(set_observations):
        if not isinstance(event, dict):
            raise ValueError("SET_EVENT_NOT_OBJECT")
        ref = checked_ref(event, f"set:{i}")
        if event.get("op") != "SET_IDENTITY":
            raise ValueError("INVALID_SET_EVENT_OP")
        rid = event.get("rescue_candidate_id")
        members = event.get("baseline_candidate_ids")
        val = event.get("verdict")
        if rid not in r_ids or not isinstance(members, list) or not members or not all(isinstance(x, str) for x in members):
            raise ValueError("INVALID_SET_MEMBERS")
        if members != sorted(set(members)) or len(members) > max_group_size or not set(members) <= set(b_ids):
            raise ValueError("UNBOUND_SET_MEMBERS")
        if val not in {"NONE_MATCH", "SOME_MATCH", "UNKNOWN"}:
            raise ValueError("INVALID_SET_VERDICT")
        if (event.get("universe_sha256") != universe or
                event.get("source_image_sha256") != source_image_sha256 or
                event.get("query_id") != manifest["query_id"]):
            raise ValueError("SET_EVIDENCE_UNIVERSE_MISMATCH")
        if ref not in gated_evidence_refs:
            errors.append("UNGATED_SET_EVENT:" + ref)
            continue
        refs_by_rescue[rid].add(ref)
        if val == "NONE_MATCH":
            none_sets[rid].append(set(members))
        elif val == "SOME_MATCH":
            blocked_by_some.add(rid)

    result = []
    for rid in r_ids:
        problems = []
        if errors:
            problems.append("UNGATED_INPUT_OBSERVATIONS")
        for op in sorted(_UNARY):
            observed = unary[rid][op]
            informative = observed - {"UNKNOWN"}
            if informative != {"YES"}:
                problems.append("MISSING_OR_CONFLICTING_" + op)
        if rid in same_with_baseline:
            problems.append("OBSERVED_SAME_BASELINE")
        if rid in linked_rescues:
            problems.append("RESCUE_SAME_RELATION_REQUIRES_GROUP_REASONING")
        if rid in blocked_by_some:
            problems.append("SET_SOME_MATCH_BLOCKS_ADMISSION")
        for bid in b_ids:
            pair = direct_pair_ref.get(tuple(sorted((rid, bid))), set())
            if any(rel == "SAME" for rel, _ in pair):
                problems.append("DIRECT_SAME_BASELINE_CONFLICT")
                break
        covered = set().union(*none_sets[rid]) if none_sets[rid] else set()
        missing = sorted(set(b_ids) - covered)
        if missing:
            problems.append("BASELINE_MEMBERS_NOT_COVERED")
        if not none_sets[rid]:
            problems.append("NO_GATED_SET_NONMATCH")
        result.append({
            "rescue_candidate_id": rid,
            "conditional_set_separation_supported": not problems,
            "uncovered_baseline_candidate_ids": missing,
            "blocking_issues": sorted(set(problems)),
            "gated_set_evidence_refs": sorted(refs_by_rescue[rid]),
            "gated_unary_evidence_refs": sorted(unary_refs[rid]),
        })
    return {
        "schema": "c3_set_nonmatch_audit_experimental_v0",
        "source_universe_sha256": universe,
        "baseline_candidate_count": len(b_ids),
        "rescue_candidate_count": len(r_ids),
        "identity_set_calls_observed_and_gated": sum(len(v) for v in refs_by_rescue.values()),
        "rescue_audits": result,
        "evidence_provenance_issues": sorted(errors),
        "operational_count_changed": False,
        "simultaneous_rescue_admissions_certified": False,
        "forced_count": None,
        "scope": "candidate_universe_only_conditioned_on_trustworthy_set_level_verdicts",
        "limitations": [
            "T06 must independently bind each gated ref to an actual raw model response",
            "Set-level NONE_MATCH is subject to correlated VLM errors and prompt complexity",
            "No physical-instance validity or whole-scene completeness is proven by this sidecar",
            "Rescue-to-rescue counting and A4 integration are NOT implemented",
            "No count promotion or accuracy claim is authorized by this audit",
        ],
    }
