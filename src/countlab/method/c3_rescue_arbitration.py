"""GT-blind direct-evidence rescue arbitration for the C3 operational channel."""

from collections import defaultdict
from itertools import combinations

_UNARY = {"REGION_SEMANTIC": "semantic", "REGION_WHOLE": "whole_instance"}


def reconcile_rescues(manifest, events, shadow, *, max_candidates=512,
                      max_search_states=20000, max_hints=32):
    baseline = {item["candidate_id"] for item in manifest["baseline"]}
    rescues = {item["candidate_id"] for item in manifest.get("rescue", [])}
    ids = baseline | rescues
    rejected = {item["candidate_id"] for item in shadow["rejected"]}
    unary = defaultdict(lambda: defaultdict(list))
    raw_pairs = defaultdict(list)

    for event in events:
        if event["op"] in _UNARY:
            unary[event["candidate_id"]][_UNARY[event["op"]]].append(
                (event["verdict"], event["evidence_ref"])
            )
        elif event["op"] == "PAIR_IDENTITY":
            raw_pairs[tuple(sorted((event["a"], event["b"])))].append(
                (event["relation"], event["evidence_ref"])
            )

    def unary_state(cid, field):
        values = {value for value, _ in unary[cid][field]}
        observed = values - {"UNKNOWN"}
        if len(observed) == 1:
            return next(iter(observed))
        if len(observed) > 1:
            return "CONFLICT"
        return "UNKNOWN" if "UNKNOWN" in values else "MISSING"

    positive = {
        cid for cid in ids
        if unary_state(cid, "semantic") == "YES"
        and unary_state(cid, "whole_instance") == "YES"
    }
    negative = {
        cid for cid in ids
        if unary_state(cid, "semantic") == "NO"
        and unary_state(cid, "whole_instance") == "NO"
    }

    def pair_state(a, b):
        if a == b:
            return "SELF"
        values = {
            value for value, _ in raw_pairs.get(tuple(sorted((a, b))), [])
        }
        observed = values - {"UNKNOWN"}
        if len(observed) == 1:
            return next(iter(observed))
        if len(observed) > 1:
            return "CONFLICT"
        return "UNKNOWN" if "UNKNOWN" in values else "MISSING"

    def pair_refs(a, b):
        return {
            ref for value, ref in raw_pairs.get(tuple(sorted((a, b))), [])
            if value != "UNKNOWN"
        }

    def unary_refs(cid):
        return {
            ref for field in _UNARY.values()
            for value, ref in unary[cid][field]
            if value != "UNKNOWN"
        }

    if len(ids) > max_candidates:
        return {
            "operational_count": len(baseline),
            "baseline_identity_merge_reduction": 0,
            "coalesced_baseline_groups": [],
            "effective_rejected": [],
            "protected_rejections": [
                {"candidate_id": cid, "reason": "RESOURCE_LIMIT", "evidence_refs": []}
                for cid in sorted(rejected)
            ],
            "admitted_rescue_groups": [],
            "pending_rescues": [
                {"candidate_id": cid, "reason": "RESOURCE_LIMIT"}
                for cid in sorted(rescues)
            ],
            "matched_rescues": [],
            "conflicts": ["resource_limit:rescue_candidate_cap"],
            "pair_hints": [],
            "hints_truncated": False,
            "search_exact": False,
            "search_states": 0,
            "policy": "DIRECT_DISTINCT_MAX_CLIQUE_GUARDED",
        }

    parent = {cid: cid for cid in ids}

    def find(cid):
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for a, b in sorted(raw_pairs):
        if pair_state(a, b) == "SAME":
            union(a, b)

    groups_by_root = defaultdict(set)
    for cid in sorted(ids):
        groups_by_root[find(cid)].add(cid)

    groups = [tuple(sorted(v)) for _, v in sorted(groups_by_root.items())]
    member_group = {
        cid: group for group in groups for cid in group
    }
    bad = set()
    conflicts = []
    hint_pairs = set()

    for group in groups:
        for a, b in combinations(group, 2):
            relation = pair_state(a, b)
            if relation != "SAME":
                bad.add(group)
                conflicts.append(
                    f"unverified_SAME_clique:{a},{b}:{relation}"
                )
                if relation in {"MISSING", "UNKNOWN"}:
                    hint_pairs.add((a, b, "confirm_same_clique"))
        if any(cid in positive for cid in group) and any(
            cid in negative for cid in group
        ):
            bad.add(group)
            conflicts.append(
                "unary_SAME_contradiction:" + ",".join(group)
            )

    for a, b in sorted(raw_pairs):
        if pair_state(a, b) in {"CONFLICT", "SELF"}:
            bad.add(member_group[a])
            bad.add(member_group[b])
            conflicts.append(f"pair_verifier_conflict:{a},{b}")

    protected = {
        cid for cid in rejected
        if member_group[cid] in bad
        or any(member in positive for member in member_group[cid])
    }
    effective_rejected = rejected - protected
    protected_ledger = []

    for cid in sorted(protected):
        group = member_group[cid]
        refs = set()
        for member in group:
            refs.update(unary_refs(member))
        for a, b in combinations(group, 2):
            refs.update(pair_refs(a, b))
        for a, b in raw_pairs:
            if (a in group or b in group) and pair_state(a, b) == "CONFLICT":
                refs.update(pair_refs(a, b))
        protected_ledger.append({
            "candidate_id": cid,
            "reason": "IDENTITY_OR_UNARY_CONFLICT",
            "evidence_refs": sorted(refs),
        })

    retained = baseline - effective_rejected
    coalesced_baseline_groups = []
    baseline_identity_merge_reduction = 0

    for group in groups:
        members = sorted(set(group) & retained)
        if len(members) < 2 or group in bad:
            continue

        refs = sorted({
            ref
            for a, b in combinations(members, 2)
            for ref in pair_refs(a, b)
        })

        if not refs:
            raise AssertionError("SAME merge lacks direct evidence")

        reduction = len(members) - 1
        baseline_identity_merge_reduction += reduction
        coalesced_baseline_groups.append({
            "member_candidate_ids": members,
            "reduction": reduction,
            "same_evidence_refs": refs,
            "rule_id": "DIRECT_COMPLETE_SAME_BASELINE_COALESCING",
        })

    retained_groups = [
        group for group in groups
        if set(group) & retained
    ]

    candidates = []
    pending = []
    matched = []

    for group in groups:
        anchors = sorted(set(group) & rescues & positive)
        if not anchors:
            continue
        if group in bad:
            for cid in anchors:
                pending.append({
                    "candidate_id": cid,
                    "reason": "IDENTITY_CONFLICT_OR_INCOMPLETE_CLIQUE",
                    "group": list(group),
                })
            continue
        if set(group) & retained:
            for cid in anchors:
                matched.append({
                    "candidate_id": cid,
                    "reason": "OBSERVED_SAME_RETAINED_BASELINE",
                    "group": list(group),
                })
            continue

        unseparated_groups = []

        for other in retained_groups:
            witnesses = [
                (a, b) for a in group for b in other
                if pair_state(a, b) == "DIFFERENT"
            ]
            if witnesses:
                continue

            unseparated_groups.append(other)
            a = anchors[0]
            b = sorted(set(other) & retained)[0]
            hint_pairs.add((
                *sorted((a, b)), "prove_distinct_rescue"
            ))

        if unseparated_groups:
            for cid in anchors:
                pending.append({
                    "candidate_id": cid,
                    "reason": "NOT_DISTINCT_FROM_ALL_RETAINED_BASELINE_GROUPS",
                    "unsatisfied_count": len(unseparated_groups),
                })
            continue

        candidates.append((group, anchors))

    def cross_distinct(a, b):
        return any(
            pair_state(x, y) == "DIFFERENT"
            for x in a for y in b
        )

    compatibility = [set() for _ in candidates]
    for i, j in combinations(range(len(candidates)), 2):
        if cross_distinct(candidates[i][0], candidates[j][0]):
            compatibility[i].add(j)
            compatibility[j].add(i)

    selected = []
    search_states = 0
    search_exact = True

    def search(chosen, remaining):
        nonlocal selected, search_states
        search_states += 1
        if search_states > max_search_states:
            raise OverflowError("bounded rescue clique search exhausted")
        if len(chosen) + len(remaining) <= len(selected):
            return
        if not remaining:
            selected = list(chosen)
            return
        head, tail = remaining[0], remaining[1:]
        search(
            chosen + [head],
            [i for i in tail if i in compatibility[head]]
        )
        search(chosen, tail)

    try:
        search([], list(range(len(candidates))))
    except OverflowError:
        search_exact = False
        selected = []
        for i in range(len(candidates)):
            if all(i in compatibility[j] for j in selected):
                selected.append(i)
        conflicts.append(
            "resource_limit:rescue_clique_search_used_greedy"
        )

    chosen_groups = [candidates[i][0] for i in selected]
    accepted = []

    for i in selected:
        group, anchors = candidates[i]
        refs = set()
        for cid in anchors:
            refs.update(unary_refs(cid))
        for a, b in combinations(group, 2):
            refs.update(pair_refs(a, b))
        for a in group:
            for b in retained:
                refs.update(pair_refs(a, b))
            for other in chosen_groups:
                if other != group:
                    for b in other:
                        refs.update(pair_refs(a, b))
        accepted.append({
            "member_candidate_ids": list(group),
            "positive_anchor_candidate_ids": anchors,
            "evidence_refs": sorted(refs),
            "rule_id": "DUAL_YES_DIRECT_DIFFERENT_CLIQUE",
        })

    selected_set = set(selected)
    for i, (group, anchors) in enumerate(candidates):
        if i in selected_set:
            continue
        for chosen in chosen_groups:
            if cross_distinct(group, chosen):
                continue

            a = anchors[0]
            b = chosen[0]
            hint_pairs.add((
                *sorted((a, b)), "prove_distinct_rescue"
            ))

        for cid in anchors:
            pending.append({
                "candidate_id": cid,
                "reason": "NOT_PROVEN_DISTINCT_FROM_SELECTED_RESCUES",
                "group": list(group),
            })

    hints = [
        {"op": "PAIR_IDENTITY", "a": a, "b": b, "reason": reason}
        for a, b, reason in sorted(hint_pairs)
    ]

    count = (
        len(baseline)
        - len(effective_rejected)
        - baseline_identity_merge_reduction
        + len(accepted)
    )
    if count < 0:
        raise AssertionError("negative operational count")

    return {
        "operational_count": count,
        "baseline_identity_merge_reduction": baseline_identity_merge_reduction,
        "coalesced_baseline_groups": coalesced_baseline_groups,
        "effective_rejected": sorted(effective_rejected),
        "protected_rejections": protected_ledger,
        "admitted_rescue_groups": accepted,
        "pending_rescues": pending,
        "matched_rescues": matched,
        "conflicts": sorted(set(conflicts)),
        "pair_hints": hints[:max_hints],
        "hints_truncated": len(hints) > max_hints,
        "search_exact": search_exact,
        "search_states": search_states,
        "policy": "DIRECT_DISTINCT_MAX_CLIQUE_GUARDED",
    }
