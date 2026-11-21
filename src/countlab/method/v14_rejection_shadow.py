"""GT-blind V14-style rejection-only operational comparator.

Not full V14 parity, an identity solver, or scene certification.
"""

import hashlib
import json

_UNARY = {
    "REGION_SEMANTIC": "semantic",
    "REGION_WHOLE": "whole_instance",
}
_VALID = {"YES", "NO", "UNKNOWN"}


def _digest(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_v14_rejection_shadow(manifest, observed_events, *, fallback_count):
    if not isinstance(manifest, dict) or not isinstance(observed_events, list):
        raise ValueError("invalid input types")

    if (
        not isinstance(fallback_count, int)
        or isinstance(fallback_count, bool)
        or fallback_count < 0
    ):
        raise ValueError("invalid fallback_count")

    if any(k in manifest for k in ("gt_count", "ground_truth", "target_count")):
        raise ValueError("GT fields forbidden in inference manifest")

    baseline = manifest.get("baseline")
    rescue = manifest.get("rescue", [])

    if not isinstance(baseline, list) or not isinstance(rescue, list):
        raise ValueError("baseline/rescue must be lists")

    if fallback_count != len(baseline):
        raise ValueError("fallback_count must equal baseline proposal count")

    ids = [
        x.get("candidate_id") if isinstance(x, dict) else None
        for x in baseline
    ]
    rescue_ids = [
        x.get("candidate_id") if isinstance(x, dict) else None
        for x in rescue
    ]

    if any(not isinstance(cid, str) or not cid for cid in ids + rescue_ids):
        raise ValueError("missing candidate_id")

    if len(set(ids + rescue_ids)) != len(ids + rescue_ids):
        raise ValueError("duplicate candidate_id")

    baseline_set = set(ids)
    allowed = set(ids + rescue_ids)
    by_id = {
        cid: {"semantic": [], "whole_instance": []}
        for cid in ids
    }

    evidence_slots = set()
    query_id = str(manifest.get("query_id"))
    observed_unary = 0
    ignored_rescue_unary = 0

    for event in observed_events:
        if not isinstance(event, dict):
            raise ValueError("malformed observed event")

        op = event.get("op")
        if op not in _UNARY:
            continue

        cid = event.get("candidate_id")

        if cid not in allowed:
            raise ValueError(f"unknown candidate_id: {cid!r}")

        if "query_id" in event and str(event["query_id"]) != query_id:
            raise ValueError("cross-query unary evidence")

        verdict = event.get("verdict")
        ref = event.get("evidence_ref")

        if verdict not in _VALID or not isinstance(ref, str) or not ref:
            raise ValueError("invalid verdict or evidence_ref")

        slot = (cid, op, ref)

        if slot in evidence_slots:
            raise ValueError("duplicate unary observation slot")

        evidence_slots.add(slot)
        observed_unary += 1

        if cid not in baseline_set:
            ignored_rescue_unary += 1
            continue

        by_id[cid][_UNARY[op]].append((verdict, ref))

    def state(items):
        values = {v for v, _ in items}

        if not values:
            return "MISSING"

        if len(values) == 1:
            return next(iter(values))

        return "CONFLICT"

    rejected = []
    retained = []
    unresolved = []

    for cid in ids:
        semantic = state(by_id[cid]["semantic"])
        whole = state(by_id[cid]["whole_instance"])

        if semantic == "NO" and whole == "NO":
            rejected.append({
                "candidate_id": cid,
                "rule_id": "V14_DUAL_NO_REJECTION",
                "evidence_refs": sorted(
                    ref
                    for items in by_id[cid].values()
                    for _, ref in items
                ),
            })
        else:
            retained.append(cid)

            if not (semantic == "YES" and whole == "YES"):
                unresolved.append({
                    "candidate_id": cid,
                    "semantic": semantic,
                    "whole_instance": whole,
                })

    return {
        "schema": "countlab_v14_rejection_shadow_v1",
        "policy": "V14_DUAL_NO_REJECTION_ONLY_NO_RESCUE_NO_IDENTITY",
        "query_id": query_id,
        "operational_count": fallback_count - len(rejected),
        "original_fallback_count": fallback_count,
        "count_delta": -len(rejected),
        "rejected": rejected,
        "retained_candidate_ids": retained,
        "unresolved": unresolved,
        "observed_unary_events": observed_unary,
        "ignored_rescue_unary_events": ignored_rescue_unary,
        "scene_zero_certified": False,
        "global_count_certified": False,
        "certified_status": "NOT_ASSESSED_SHADOW_ONLY",
        "manifest_sha256": _digest(manifest),
        "observed_events_sha256": _digest(observed_events),
    }
