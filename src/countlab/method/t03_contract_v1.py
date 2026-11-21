"""Opt-in, strict T03 boundary for candidate/evidence manifests and method outputs.

GT-blind software contract only. Does not perform inference, change solver policy,
or certify the physical truth of supplied evidence. Additive to the legacy v1
parser so teams can migrate at an explicit integration boundary.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

from .schema import parse_input

SCHEMA_ID = "countlab_candidate_pool_v1"
OUTPUT_SCHEMAS = frozenset({"countlab_unified_method_output_v1", "countlab_joint_arbitration_experimental_v1"})
ROOT_KEYS = frozenset({
    "schema", "image_id", "query_id", "query_text", "baseline_count", "baseline",
    "rescue", "identity_edges", "metadata",
})
REQUIRED_ROOT = ROOT_KEYS - {"metadata"}
CANDIDATE_KEYS = frozenset({
    "candidate_id", "view_id", "box_xyxy", "confidence", "risk_hint", "evidence",
    "evidence_refs", "provenance",
})
EDGE_KEYS = frozenset({"a", "b", "relation", "evidence_refs", "provenance"})
UNARY_KEYS = frozenset({"semantic", "whole_instance"})
UNARY_VALUES = frozenset({"YES", "NO", "UNKNOWN"})
PAIR_VALUES = frozenset({"SAME", "DIFFERENT", "UNKNOWN"})
# Normalize separators and casing to catch key variants inside provenance too.
_GT_KEYS = frozenset({
    "gt", "gtcount", "gtboxes", "gtbbox", "gtmask", "gtlabels", "gtpoints",
    "groundtruth", "groundtruthcount", "groundtruthboxes", "groundtruthannotations",
    "truecount", "targetcount", "labelcount", "goldcount", "numgt",
    "annotation", "annotations", "label", "labels", "presencelabel",
})


class T03ContractError(ValueError):
    """A contract error, not an inference result of UNKNOWN."""


def _fail(location: str, explanation: str) -> None:
    raise T03ContractError(f"{location}: {explanation}")


def _no_gt(value: Any, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                _fail(location, "object keys must be strings")
            normalized = re.sub(r"[^a-z0-9]", "", raw_key.casefold())
            if normalized in _GT_KEYS or normalized.startswith("groundtruth"):
                _fail(f"{location}.{raw_key}", "GT/annotation field forbidden in inference")
            _no_gt(child, f"{location}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _no_gt(child, f"{location}[{index}]")


def _keys(value: Any, location: str, allowed: frozenset[str], required: frozenset[str]) -> None:
    if not isinstance(value, dict):
        _fail(location, "expected JSON object")
    extra = set(value) - allowed
    missing = required - set(value)
    if extra:
        _fail(location, "unexpected key(s): " + ", ".join(sorted(map(str, extra))))
    if missing:
        _fail(location, "missing key(s): " + ", ".join(sorted(missing)))


def _text(value: Any, location: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(location, "must be a nonempty string")


def _float(value: Any, location: str, *, probability: bool = False) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        _fail(location, "must be a finite JSON number (not bool or numeric string)")
    if probability and not 0 <= value <= 1:
        _fail(location, "must lie in [0, 1]")


def _refs(value: Any, location: str) -> None:
    if not isinstance(value, list):
        _fail(location, "must be a list of evidence reference strings")
    for index, ref in enumerate(value):
        _text(ref, f"{location}[{index}]")
    if len(set(value)) != len(value):
        _fail(location, "duplicate evidence reference")


def validate_candidate_pool(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate strict T03 v1 syntax and return lightweight audit counts.

    Calls the existing authoritative parse_input after enforcing closed keys and
    numeric/ID types. Does not rewrite or normalize the input.
    """
    _no_gt(manifest)
    _keys(manifest, "$", ROOT_KEYS, REQUIRED_ROOT)
    if manifest["schema"] != SCHEMA_ID:
        _fail("$.schema", f"expected {SCHEMA_ID}")
    for key in ("image_id", "query_id", "query_text"):
        _text(manifest[key], "$." + key)
    if type(manifest["baseline_count"]) is not int or manifest["baseline_count"] < 0:
        _fail("$.baseline_count", "must be nonnegative integer")
    if "metadata" in manifest and not isinstance(manifest["metadata"], dict):
        _fail("$.metadata", "expected provenance object")

    ids: set[str] = set()
    for role in ("baseline", "rescue"):
        items = manifest[role]
        if not isinstance(items, list):
            _fail("$." + role, "must be a list")
        for index, proposal in enumerate(items):
            loc = f"$.{role}[{index}]"
            _keys(proposal, loc, CANDIDATE_KEYS, frozenset({"candidate_id", "box_xyxy"}))
            cid = proposal["candidate_id"]
            _text(cid, loc + ".candidate_id")
            if cid in ids:
                _fail(loc + ".candidate_id", "duplicate candidate ID across baseline/rescue")
            ids.add(cid)
            if "view_id" in proposal:
                _text(proposal["view_id"], loc + ".view_id")
            b = proposal["box_xyxy"]
            if not isinstance(b, list) or len(b) != 4:
                _fail(loc + ".box_xyxy", "expected [x1,y1,x2,y2] in original pixels")
            for k, value in enumerate(b):
                _float(value, f"{loc}.box_xyxy[{k}]")
            if b[2] <= b[0] or b[3] <= b[1]:
                _fail(loc + ".box_xyxy", "must have strictly positive area")
            for key in ("confidence", "risk_hint"):
                if key in proposal and proposal[key] is not None:
                    _float(proposal[key], loc + "." + key, probability=True)
            if "evidence_refs" in proposal:
                _refs(proposal["evidence_refs"], loc + ".evidence_refs")
            if "provenance" in proposal and not isinstance(proposal["provenance"], dict):
                _fail(loc + ".provenance", "expected provenance object")
            if "evidence" in proposal:
                ev = proposal["evidence"]
                _keys(ev, loc + ".evidence", UNARY_KEYS, frozenset())
                for name, verdict in ev.items():
                    if not isinstance(verdict, str) or verdict not in UNARY_VALUES:
                        _fail(loc + ".evidence." + name, "expected YES/NO/UNKNOWN")
    if manifest["baseline_count"] != len(manifest["baseline"]):
        _fail("$.baseline_count", "must equal number of baseline proposal IDs")

    edges = manifest["identity_edges"]
    if not isinstance(edges, list):
        _fail("$.identity_edges", "must be a list")
    seen: set[tuple[str, str]] = set()
    for index, edge in enumerate(edges):
        loc = f"$.identity_edges[{index}]"
        _keys(edge, loc, EDGE_KEYS, frozenset({"a", "b", "relation"}))
        for key in ("a", "b"):
            _text(edge[key], loc + "." + key)
        a, b = edge["a"], edge["b"]
        if a not in ids or b not in ids or a == b:
            _fail(loc, "identity edge must connect two distinct known candidates")
        if not isinstance(edge["relation"], str) or edge["relation"] not in PAIR_VALUES:
            _fail(loc + ".relation", "expected SAME/DIFFERENT/UNKNOWN")
        pair = tuple(sorted((a, b)))
        if pair in seen:
            _fail(loc, "repeated identity pair (including reverse order)")
        seen.add(pair)
        if "evidence_refs" in edge:
            _refs(edge["evidence_refs"], loc + ".evidence_refs")
        if "provenance" in edge and not isinstance(edge["provenance"], dict):
            _fail(loc + ".provenance", "expected provenance object")
    try:
        base, rescue, observed_edges = parse_input(manifest)
    except (TypeError, ValueError, KeyError) as exc:
        raise T03ContractError("legacy parse_input rejected manifest: " + str(exc)) from exc
    return {
        "schema": "countlab_t03_validation_v1",
        "valid": True,
        "image_id": manifest["image_id"],
        "query_id": manifest["query_id"],
        "baseline_count": len(base),
        "rescue_count": len(rescue),
        "identity_edge_count": len(observed_edges),
        "input_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "guarantee": "STRUCTURE_AND_GT_KEY_SCAN_ONLY_NOT_PHYSICAL_TRUTH",
    }


def validate_method_output(manifest: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    """Validate common ZERO/K/ABSTAIN and fallback semantics for v1 and joint_t1.

    This is a *boundary* validation, not a re-execution or a proof that unary/
    pairwise evidence is visually true. It does not normalize COUNT.
    """
    inp = validate_candidate_pool(manifest)
    _no_gt(prediction)
    if not isinstance(prediction, dict):
        _fail("$prediction", "expected object")
    schema = prediction.get("schema")
    if schema not in OUTPUT_SCHEMAS:
        _fail("$prediction.schema", "unknown output schema")
    for key in ("count", "evaluation_fallback_count", "count_lower_bound", "count_upper_bound",
                "count_interval_scope", "query_abstained", "count_status", "input_sha256"):
        if key not in prediction:
            _fail("$prediction", "missing field: " + key)
    if prediction["input_sha256"] != inp["input_sha256"]:
        _fail("$prediction.input_sha256", "does not match frozen input")
    if type(prediction["query_abstained"]) is not bool:
        _fail("$prediction.query_abstained", "must be boolean")
    fallback = prediction["evaluation_fallback_count"]
    lower, upper = prediction["count_lower_bound"], prediction["count_upper_bound"]
    for key, value in (("evaluation_fallback_count", fallback),
                       ("count_lower_bound", lower), ("count_upper_bound", upper)):
        if type(value) is not int or value < 0:
            _fail("$prediction." + key, "must be nonnegative integer")
    if not lower <= fallback <= upper:
        _fail("$prediction", "observed-candidate bounds do not contain fallback")
    if prediction["count_interval_scope"] != "observed_candidates_only_not_scene_ground_truth_bounds":
        _fail("$prediction.count_interval_scope", "must not claim scene-wide bounds")
    abstained = prediction["query_abstained"]
    if prediction["count_status"] != ("abstained" if abstained else "partial"):
        _fail("$prediction.count_status", "inconsistent with abstention")
    if (prediction["count"] is not None if abstained else
            (type(prediction["count"]) is not int or prediction["count"] != fallback)):
        _fail("$prediction.count", "must be null when abstained, else an integer equal to fallback")
    if schema == "countlab_unified_method_output_v1":
        for key in ("image_id", "query_id"):
            if prediction.get(key) != manifest[key]:
                _fail("$prediction." + key, "does not match input")
        for key in ("accepted_instances", "rejected_regions", "unknown_regions", "evidence_trace"):
            if not isinstance(prediction.get(key), list):
                _fail("$prediction." + key, "must be list")
    else:
        for key in ("fallback_baseline_candidate_ids", "predicted_components",
                    "unresolved_candidates", "joint_trace"):
            if not isinstance(prediction.get(key), list):
                _fail("$prediction." + key, "must be list")
    return {
        "schema": "countlab_t03_output_validation_v1",
        "valid": True,
        "input_sha256": inp["input_sha256"],
        "output_schema": schema,
        "decision": "ABSTAIN" if abstained else ("ZERO" if fallback == 0 else "K"),
        "fallback_count": fallback,
        "guarantee": "OUTPUT_CONTRACT_ONLY_NOT_VISUAL_OR_SCENE_TRUTH",
    }
