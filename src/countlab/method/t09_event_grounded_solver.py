"""C3/T09 experimental evidence-event-grounded physical-instance counting.

No GT, detector, Qwen, C1 policy, T1 pairwise plan or model calls here.
Observed events, not hidden precomputed manifest answers, determine the graph.
All count assertions are conditional on truthful supplied verifier observations
and concern the observed candidate pool, never completeness of the scene.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Any

from .schema import box_iou, parse_input
from .t09_constraint_bounds import _chromatic_bounds

_OPERATIONS = frozenset({"REGION_SEMANTIC", "REGION_WHOLE", "PAIR_IDENTITY"})
_UNARY_VALUES = frozenset({"YES", "NO", "UNKNOWN"})
_PAIR_VALUES = frozenset({"SAME", "DIFFERENT", "UNKNOWN"})


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _check_cap(value: int, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _canonical_source(manifest: dict, candidates: list[Any]) -> dict:
    """Strip all model answers so hidden precomputed evidence cannot leak."""
    metadata = manifest.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    return {
        "schema": "countlab_c3_visible_source_v1",
        "image_sha256": metadata.get("image_sha256"),
        "backend_protocol": metadata.get("backend_protocol"),
        "image_id": manifest["image_id"],
        "query_id": manifest["query_id"],
        "query_text": manifest["query_text"],
        "baseline_count": manifest["baseline_count"],
        "candidates": [
            {"candidate_id": p.candidate_id, "role": p.role, "view_id": p.view_id,
             "box_xyxy": list(p.box), "confidence": p.confidence}
            for p in sorted(candidates, key=lambda x: x.candidate_id)
        ],
    }


def _decode_events(events: list[dict], candidate_ids: set[str]) -> tuple[dict, dict, dict, list[str]]:
    """Fold repeated UNKNOWNs/retries but refuse conflicting informative answers.

    An UNKNOWN followed by YES/NO is a valid evidence-acquisition retry.
    Two incompatible informative outcomes are an explicit contradiction, not
    last-write-wins. Unique evidence references preserve the observed history.
    """
    if not isinstance(events, list):
        raise ValueError("events must be a list")
    semantic: dict[str, tuple[str, str]] = {}
    whole: dict[str, tuple[str, str]] = {}
    pairs: dict[tuple[str, str], tuple[str, str]] = {}
    all_refs: set[str] = set()
    conflicts: list[str] = []

    def update(target: dict, key: Any, value: str, ref: str, label: str) -> None:
        if key not in target:
            target[key] = (value, ref)
            return
        previous, _ = target[key]
        if previous == "UNKNOWN" and value != "UNKNOWN":
            target[key] = (value, ref)
        elif previous != "UNKNOWN" and value not in {"UNKNOWN", previous}:
            conflicts.append("verifier_conflict:" + label)

    for index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("op") not in _OPERATIONS:
            raise ValueError(f"event {index}: unsupported evidence operation")
        operation = event["op"]
        evidence_ref = event.get("evidence_ref")
        if not isinstance(evidence_ref, str) or not evidence_ref.strip() or evidence_ref in all_refs:
            raise ValueError(f"event {index}: missing or duplicated evidence_ref")
        all_refs.add(evidence_ref)
        if operation == "PAIR_IDENTITY":
            a, b = event.get("a"), event.get("b")
            relation = event.get("relation")
            if (not isinstance(a, str) or not isinstance(b, str) or a == b or
                    a not in candidate_ids or b not in candidate_ids or relation not in _PAIR_VALUES):
                raise ValueError(f"event {index}: invalid pairwise observation")
            pair = tuple(sorted((a, b)))
            update(pairs, pair, relation, evidence_ref,
                   "PAIR_IDENTITY:" + ",".join(pair))
        else:
            cid, verdict = event.get("candidate_id"), event.get("verdict")
            if not isinstance(cid, str) or cid not in candidate_ids or verdict not in _UNARY_VALUES:
                raise ValueError(f"event {index}: invalid region observation")
            target = semantic if operation == "REGION_SEMANTIC" else whole
            update(target, cid, verdict, evidence_ref, operation + ":" + cid)
    return semantic, whole, pairs, sorted(set(conflicts))


def infer_event_grounded_count(manifest: dict, events: list[dict], *,
                               fallback_count: int,
                               max_search_states: int = 20000,
                               max_candidates: int = 512,
                               max_hints: int = 32,
                               min_same_iou: float = 0.05) -> dict[str, Any]:
    """Reason from observed C2 events; return a conservative provisional K or abstain.

    A decision-changing K requires exact POSITIVE observed-pool count, verified
    query membership, complete direct SAME support within each multi-proposal
    component, and explicit DIFFERENT separation of all counted components.
    This is a logical *conditional* certificate, not a calibrated guarantee.

    The evaluation forced count changes only on this strict positive branch.
    No candidate pool can by itself prove global absence; empty/NO-only pool
    stays ABSTAIN with the caller's fixed fallback count.
    """
    _check_cap(fallback_count, "fallback_count")
    _check_cap(max_search_states, "max_search_states")
    _check_cap(max_candidates, "max_candidates", minimum=1)
    _check_cap(max_hints, "max_hints")
    if isinstance(min_same_iou, bool) or not isinstance(min_same_iou, (int, float)) or not (0 <= min_same_iou <= 1):
        raise ValueError("min_same_iou must be in [0,1]")
    baseline, rescue, _ = parse_input(manifest)
    candidates = baseline + rescue
    by_id = {p.candidate_id: p for p in candidates}
    semantic, whole, pairs, verifier_conflicts = _decode_events(events, set(by_id))
    visible_event_data = [dict(e) for e in events]
    source_digest = _sha(_canonical_source(manifest, candidates))
    reasoning_policy = {"min_same_iou": float(min_same_iou),
                        "max_search_states": max_search_states,
                        "max_candidates": max_candidates, "max_hints": max_hints}
    event_digest = _sha(sorted(visible_event_data, key=lambda e: e["evidence_ref"]))

    result: dict[str, Any] = {
        "schema": "countlab_c3_event_grounded_decision_experimental_v1",
        "scope": "observed_candidate_pool_conditioned_on_verifier_observations",
        "status": "ABSTAIN",
        "count": None,
        "evaluation_fallback_count": fallback_count,
        "original_fallback_count": fallback_count,
        "decision_changes_fallback": False,
        "observed_count_lower_bound": None,
        "observed_count_upper_bound": None,
        "bounds_exact": False,
        "physical_instances": [],
        "unresolved_candidate_ids": [],
        "excluded_candidate_ids": [],
        "blocking_issues": [],
        "missing_evidence_hints": [],
        "source_visible_sha256": source_digest,
        "reasoning_policy": reasoning_policy,
        "reasoning_policy_sha256": _sha(reasoning_policy),
        "source_has_image_sha256": isinstance(manifest.get("metadata"), dict)
        and isinstance(manifest["metadata"].get("image_sha256"), str)
        and len(manifest["metadata"]["image_sha256"]) == 64,
        "source_has_backend_protocol": isinstance(manifest.get("metadata"), dict)
        and bool(manifest["metadata"].get("backend_protocol")),
        "observed_events_sha256": event_digest,
        "observed_event_count": len(events),
        "observed_operation_counts": {op: sum(e["op"] == op for e in events)
                                      for op in sorted(_OPERATIONS)},
        "executed_model_calls_by_C3": 0,
        "limitations": [
            "Verifier observations can be wrong: no visual-truth or calibration guarantee",
            "Conditional counts cover observed proposals only, not missed scene objects",
            "Positive forced count is a provisional candidate-pool decision, not scene certification",
            "No zero/absence certification from an empty or rejected candidate pool",
            "C3 does not select or execute C1/C2 evidence actions",
            "Action cost must be obtained from measured C2 execution logs",
            "Evidence refs are source-supplied identifiers, not independently authenticated signatures",
            "No result here establishes correctness of target class or visual identity",
        ],
    }
    if len(candidates) > max_candidates:
        result["blocking_issues"].append("resource_limit:candidate_cap")
        return result

    state_by_candidate = {}
    for cid in sorted(by_id):
        s = semantic.get(cid, ("UNKNOWN", None))[0]
        w = whole.get(cid, ("UNKNOWN", None))[0]
        state_by_candidate[cid] = (
            "YES" if s == w == "YES" else "NO" if s == w == "NO" else "UNKNOWN"
        )
    parent = {cid: cid for cid in by_id}

    def root(cid: str) -> str:
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid

    for (a, b), (relation, _) in sorted(pairs.items()):
        if relation == "SAME":
            left, right = root(a), root(b)
            if left != right:
                parent[max(left, right)] = min(left, right)

    groups_by_root: dict[str, list[str]] = {}
    for cid in sorted(by_id):
        groups_by_root.setdefault(root(cid), []).append(cid)
    groups = [tuple(ids) for _, ids in sorted(groups_by_root.items())]
    lookup = {cid: index for index, group in enumerate(groups) for cid in group}
    states = []
    issues = list(verifier_conflicts)
    for (a, b), (relation, _) in pairs.items():
        if relation == "SAME" and box_iou(by_id[a].box, by_id[b].box) < min_same_iou:
            issues.append("low_overlap_SAME_review:" + a + "," + b)
    for group in groups:
        member_states = {state_by_candidate[cid] for cid in group}
        if "YES" in member_states and "NO" in member_states:
            issues.append("unary_SAME_contradiction:" + ",".join(group))
        states.append("YES" if "YES" in member_states else
                      "NO" if "NO" in member_states else "UNKNOWN")
        for a, b in combinations(group, 2):
            if pairs.get((a, b), (None, None))[0] != "SAME":
                issues.append("incomplete_SAME_clique:" + a + "," + b)

    different = set()
    pair_ref = {}
    for (a, b), (relation, ref) in sorted(pairs.items()):
        if relation == "DIFFERENT":
            i, j = lookup[a], lookup[b]
            if i == j:
                issues.append("SAME_DIFFERENT_contradiction:" + a + "," + b)
            else:
                pair = tuple(sorted((i, j)))
                different.add(pair)
                pair_ref.setdefault(pair, set()).add(ref)

    result["unverified_unary_candidate_ids"] = [cid for cid in sorted(by_id)
                                                if state_by_candidate[cid] == "UNKNOWN"]
    result["unresolved_candidate_ids"] = [cid for cid in sorted(by_id)
                                          if states[lookup[cid]] == "UNKNOWN"]
    result["excluded_candidate_ids"] = [cid for cid in sorted(by_id)
                                         if states[lookup[cid]] == "NO"]
    active = [i for i, state in enumerate(states) if state == "YES"]
    optional = sum(state == "UNKNOWN" for state in states)
    fatal = any(s.startswith(("verifier_conflict:", "unary_SAME_contradiction:",
                               "SAME_DIFFERENT_contradiction:")) for s in issues)
    lower = upper = None
    exact = False
    if not fatal:
        active_index = {group_index: i for i, group_index in enumerate(active)}
        adj = [set() for _ in active]
        for a, b in different:
            if a in active_index and b in active_index:
                i, j = active_index[a], active_index[b]
                adj[i].add(j)
                adj[j].add(i)
        lower, _, exact, _ = _chromatic_bounds(tuple(frozenset(neighbors)
                                                  for neighbors in adj), max_search_states)
        upper = len(active) + optional
        if not exact:
            issues.append("resource_limit:chromatic_search")
    result["observed_count_lower_bound"] = lower
    result["observed_count_upper_bound"] = upper
    result["bounds_exact"] = exact

    # C3 supplies missing-evidence *hints*. This is not C1 action selection.
    hints = []
    if not candidates:
        hints.append({"op": "RECOVER_VIEW", "reason": "empty_pool_is_not_absence"})
    for cid in sorted(by_id):
        if cid not in semantic:
            hints.append({"op": "REGION_SEMANTIC", "candidate_id": cid})
        if cid not in whole:
            hints.append({"op": "REGION_WHOLE", "candidate_id": cid})
    for reason in sorted(set(issues)):
        if reason.startswith("incomplete_SAME_clique:"):
            a, b = reason.split(":", 1)[1].split(",", 1)
            hints.append({"op": "PAIR_IDENTITY", "a": a, "b": b,
                          "reason": "confirm_transitive_identity"})
    for i, j in combinations(active, 2):
        pair = tuple(sorted((i, j)))
        if pair not in different:
            hints.append({"op": "PAIR_IDENTITY", "a": groups[i][0], "b": groups[j][0],
                          "reason": "distinct_instance_ambiguity"})
            if len(hints) >= max_hints + 1:
                break
    result["missing_evidence_hints"] = hints[:max_hints]
    result["hints_truncated"] = len(hints) > max_hints
    result["blocking_issues"] = sorted(set(issues))

    if fatal or issues or not exact or lower is None or lower != upper or lower == 0:
        return result

    # Positive singleton implies all potentially counted components are forced
    # present and mutually DIFFERENT; verify this witness explicitly.
    if optional:
        raise AssertionError("singleton positive interval cannot contain optional groups")
    if len(active) != lower:
        raise AssertionError("singleton positive interval requires complete differentiation")
    for i, j in combinations(active, 2):
        if tuple(sorted((i, j))) not in different:
            raise AssertionError("positive uniqueness without observed DIFFERENT evidence")

    certified_instances = []
    for i in active:
        members = groups[i]
        anchors = [cid for cid in members if state_by_candidate[cid] == "YES"]
        if not anchors:
            raise AssertionError("present group needs verified unary evidence")
        unary_refs = sorted({semantic[cid][1] for cid in anchors} |
                            {whole[cid][1] for cid in anchors})
        same_refs = sorted(pairs[(a, b)][1] for a, b in combinations(members, 2))
        separation_refs = sorted({ref for j in active if j != i
                                 for ref in pair_ref[tuple(sorted((i, j)))]})
        certified_instances.append({
            "instance_id": "obs:" + _sha(members)[:16],
            "member_candidate_ids": list(members),
            "query_grounded_anchor_candidate_ids": anchors,
            "unary_evidence_refs": unary_refs,
            "same_identity_evidence_refs": same_refs,
            "different_identity_evidence_refs": separation_refs,
            "decision": "INCLUDE_CONDITIONAL_ON_EVIDENCE",
            "decision_rule": "DUAL_YES_COMPLETE_SAME_AND_EXPLICIT_DIFFERENT",
        })
    result.update({
        "status": "PROVISIONAL_POSITIVE",
        "count": lower,
        "evaluation_fallback_count": lower,
        "decision_changes_fallback": lower != fallback_count,
        "physical_instances": certified_instances,
    })
    return result


def replay_evidence_prefixes(manifest: dict, events: list[dict], *, fallback_count: int,
                             max_events: int = 256,
                             max_search_states: int = 20000) -> list[dict[str, Any]]:
    """Audit how each actually observed event causally changes count bounds.

    No hypothetical outcomes. A consistent exact feasible count interval may
    only shrink as observed constraints accumulate, conditional on evidence
    truth and fixed observed candidate set.
    """
    _check_cap(max_events, "max_events")
    if not isinstance(events, list) or len(events) > max_events:
        raise ValueError("event sequence exceeds replay cap")
    rows = []
    previous = None
    for k in range(len(events) + 1):
        out = infer_event_grounded_count(manifest, events[:k], fallback_count=fallback_count,
                                         max_search_states=max_search_states)
        bounds = (out["observed_count_lower_bound"], out["observed_count_upper_bound"])
        if previous is not None and previous["bounds_exact"] and out["bounds_exact"] and (
                not previous["blocking_issues"] and not out["blocking_issues"]):
            if not (previous["observed_count_lower_bound"] <= bounds[0] <= bounds[1] <=
                    previous["observed_count_upper_bound"]):
                raise AssertionError("compatible evidence cannot widen exact count interval")
        rows.append({"event_index": k,
                     "event_ref": None if k == 0 else events[k - 1]["evidence_ref"],
                     "lower": bounds[0], "upper": bounds[1],
                     "bounds_exact": out["bounds_exact"],
                     "status": out["status"],
                     "forced_count": out["evaluation_fallback_count"],
                     "blocking_issues": out["blocking_issues"]})
        previous = out
    return rows
