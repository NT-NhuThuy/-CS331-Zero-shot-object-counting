"""C3/T09: GT-blind conditional count bounds on a frozen candidate pool.

This computes logically admissible counts under *observed* unary/pairwise answers,
NOT truth of those answers, scene completeness, or calibrated counting accuracy.
No detector, verifier, router or baseline code is called except frozen T1 replay.

For each SAME-connected group, verified YES forces presence, verified NO forces
absence, and otherwise presence is optional. Explicit DIFFERENT is inequality.
The minimal number of present physical instances is the chromatic number of
DIFFERENT constraints among forced-present groups. The maximum is the number
of all potentially present groups. If chromatic search is cut short, the lower
bound is deliberately conservative and is NOT called an exact feasible set.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

from .schema import parse_input
from .t03_integrity import validate_joint_t1_integrity
from .t09_admissible_worlds import _candidate_state, _visible


def _greedy_clique_lower(adj: tuple[frozenset[int], ...]) -> int:
    """A genuine clique gives a proof of a lower bound on chromatic number."""
    n = len(adj)
    if n == 0:
        return 0
    result = 1
    for start in sorted(range(n), key=lambda i: (-len(adj[i]), i)):
        clique = {start}
        for u in sorted(adj[start], key=lambda i: (-len(adj[i]), i)):
            if all(u in adj[v] for v in clique):
                clique.add(u)
        result = max(result, len(clique))
    return result


def _greedy_coloring_upper(adj: tuple[frozenset[int], ...]) -> int:
    colors: dict[int, int] = {}
    for node in sorted(range(len(adj)), key=lambda i: (-len(adj[i]), i)):
        taken = {colors[neighbor] for neighbor in adj[node] if neighbor in colors}
        color = 0
        while color in taken:
            color += 1
        colors[node] = color
    return max(colors.values(), default=-1) + 1


def _chromatic_bounds(adj: tuple[frozenset[int], ...], max_search_states: int) -> tuple[int, int, bool, int]:
    """(proven lower, proven upper, exact, explored states).

    Search checks k-colorability. A completed UNSAT search certifies chi > k.
    A SAT coloring certifies chi <= k. On state-limit exit, neither is inferred.
    """
    n = len(adj)
    if n == 0:
        return 0, 0, True, 0
    if all(not row for row in adj):
        return 1, 1, True, 0
    if all(len(row) == n - 1 for row in adj):
        return n, n, True, 0
    lb = _greedy_clique_lower(adj)
    ub = _greedy_coloring_upper(adj)
    if lb == ub:
        return lb, ub, True, 0

    explored = 0
    for k in range(lb, ub):
        colors = [-1] * n
        exhausted = False

        def solve(colored: int, used: int) -> bool:
            nonlocal explored, exhausted
            if explored >= max_search_states:
                exhausted = True
                return False
            explored += 1
            if colored == n:
                return True
            uncolored = [v for v in range(n) if colors[v] < 0]
            pick = min(uncolored, key=lambda v: (
                -len({colors[w] for w in adj[v] if colors[w] >= 0}),
                -len(adj[v]), v))
            prohibited = {colors[w] for w in adj[pick] if colors[w] >= 0}
            for color in range(min(k, used + 1)):
                if color in prohibited:
                    continue
                colors[pick] = color
                if solve(colored + 1, max(used, color + 1)):
                    return True
                colors[pick] = -1
                if exhausted:
                    return False
            return False

        found = solve(0, 0)
        if found:
            return k, k, True, explored
        if exhausted:
            return lb, ub, False, explored
        lb = k + 1
    return ub, ub, True, explored


def reason_count_bounds(manifest: dict, t1_prediction: dict, *,
                        max_search_states: int = 20000) -> dict[str, Any]:
    """Return evidence-conditional count bounds without exponential partition replay.

    Fail closed for contradictory unary/identity evidence. Missing pair evidence
    is UNKNOWN, not DIFFERENT. Incomplete SAME cliques are flagged and cannot
    justify an unqualified positive decision. A zero observed count never
    certifies absence of target instances outside the candidate pool.
    """
    if type(max_search_states) is not int or max_search_states < 0:
        raise ValueError("max_search_states must be a nonnegative integer")
    integrity = validate_joint_t1_integrity(manifest, t1_prediction)
    baseline, rescues, _ = parse_input(manifest)
    if not t1_prediction["joint_policy"]["enable_recovery"]:
        rescues = []
    candidates = baseline + rescues
    by_id = {p.candidate_id: p for p in candidates}
    selected = set(t1_prediction["selected_candidate_ids"])
    edges = {pair: relation for pair, relation in _visible(manifest, t1_prediction).items()
             if pair[0] in by_id and pair[1] in by_id}

    parent = {cid: cid for cid in by_id}

    def root(cid: str) -> str:
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid

    for (a, b), relation in sorted(edges.items()):
        if relation == "SAME":
            left, right = root(a), root(b)
            if left != right:
                parent[max(left, right)] = min(left, right)

    components: dict[str, list[str]] = {}
    for cid in sorted(by_id):
        components.setdefault(root(cid), []).append(cid)
    groups = [tuple(group) for _, group in sorted(components.items())]
    index = {cid: i for i, group in enumerate(groups) for cid in group}
    violations: list[str] = []
    states: list[str] = []
    for group in groups:
        observed = {_candidate_state(by_id[cid], selected) for cid in group}
        if "YES" in observed and "NO" in observed:
            violations.append("semantic_identity_conflict:" + ",".join(group))
        states.append("YES" if "YES" in observed else "NO" if "NO" in observed else "UNKNOWN")
        for a, b in combinations(group, 2):
            if edges.get(tuple(sorted((a, b)))) != "SAME":
                violations.append("unconfirmed_transitive_same:" + a + "," + b)

    different_pairs: set[tuple[int, int]] = set()
    for (a, b), relation in sorted(edges.items()):
        if relation == "DIFFERENT":
            u, v = index[a], index[b]
            if u == v:
                violations.append("same_different_conflict:" + a + "," + b)
            else:
                different_pairs.add(tuple(sorted((u, v))))

    violations = sorted(set(violations))
    fatal = any(reason.startswith(("semantic_identity_conflict:", "same_different_conflict:"))
                for reason in violations)
    active = [i for i, state in enumerate(states) if state == "YES"]
    optional = sum(state == "UNKNOWN" for state in states)
    lower: int | None = None
    upper: int | None = None
    exact = False
    steps = 0
    certificate: dict[str, Any] = {"reason": "contradictory_constraints"} if fatal else {}
    if not fatal:
        active_index = {node: j for j, node in enumerate(active)}
        adjacency = [set() for _ in active]
        for a, b in different_pairs:
            if a in active_index and b in active_index:
                i, j = active_index[a], active_index[b]
                adjacency[i].add(j)
                adjacency[j].add(i)
        adj = tuple(frozenset(row) for row in adjacency)
        lower, chi_upper, exact, steps = _chromatic_bounds(adj, max_search_states)
        upper = len(active) + optional
        certificate = {
            "chromatic_lower_bound": lower,
            "chromatic_upper_bound": chi_upper,
            "chromatic_exact": exact,
            "chromatic_search_states": steps,
            "forced_present_groups": len(active),
            "optional_groups": optional,
            "ignored_absent_groups": sum(state == "NO" for state in states),
            "assumptions": ["supplied_visible_verdicts_are_correct",
                            "same_is_identity_equality", "different_is_identity_inequality"],
        }
    # A coarse lower bound is never advertised as a list of exact possibilities.
    feasible = list(range(lower, upper + 1)) if exact and lower is not None else None
    positive = bool(lower is not None and lower == upper and lower > 0 and not violations)
    return {
        "schema": "countlab_t09_conditional_count_bounds_v1",
        "scope": "observed_candidates_not_full_scene_conditional_on_verifier_evidence",
        "status": "OBSERVED_POSITIVE" if positive else "ABSTAIN",
        "count": lower if positive else None,
        "evaluation_fallback_count": t1_prediction["evaluation_fallback_count"],
        "observed_count_lower_bound": lower,
        "observed_count_upper_bound": upper,
        "feasible_observed_counts": feasible,
        "bounds_exact": exact,
        "constraints_consistent": not fatal,
        "unresolved_identity_issues": violations,
        "same_components": [list(group) for group in groups],
        "component_membership_states": states,
        "chromatic_certificate": certificate,
        "input_sha256": integrity["input_sha256"],
        "t1_prediction_sha256": integrity["prediction_sha256"],
        "budget_visibility": "all_supplied_edges_unmetered" if t1_prediction["joint_policy"]["pairwise_budget"] is None else "selected_pairwise_plan_only",
        "limitations": [
            "A positive observed count is not a full-scene recall certificate",
            "A zero observed count never certifies global target absence",
            "Verifier evidence may be wrong; no calibration or empirical improvement is implied",
            "Conservative bounds are not an exact feasible-count enumeration when bounds_exact is false",
            "This is a C3 analysis module; it does not execute C1/C2 evidence acquisition",
        ],
    }
