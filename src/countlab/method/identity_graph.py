"""Deterministic constrained identity graph arbitration (Thúy / P05).

Only explicit verifier edges create or separate identity components. Geometry
routes ambiguity but is never promoted to identity evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
from .schema import Proposal, box_iou


@dataclass(frozen=True)
class IdentityGraphResult:
    accepted_components: tuple[tuple[str, ...], ...]
    unresolved_components: tuple[tuple[str, ...], ...]
    rejected_same_edges: tuple[tuple[str, str, str], ...]
    reasons: tuple[str, ...]


def arbitrate_identity_graph(candidates: list[Proposal], edges: dict[tuple[str, str], str], *,
                             min_same_iou: float = .05,
                             ambiguity_iou: float = .30) -> IdentityGraphResult:
    if not 0 < min_same_iou <= ambiguity_iou <= 1:
        raise ValueError("identity graph IoU policy is invalid")
    by_id = {x.candidate_id: x for x in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("identity graph candidate IDs must be unique")
    for (a, b), relation in edges.items():
        if (a not in by_id or b not in by_id or a == b or a > b or
                relation not in {"SAME", "DIFFERENT", "UNKNOWN"}):
            raise ValueError("identity graph edge is invalid")

    parent = {x: x for x in by_id}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a, b)] = min(a, b)

    rejected=[]
    for (a, b), relation in sorted(edges.items()):
        if relation != "SAME": continue
        pa, pb = by_id[a], by_id[b]
        if pa.view_id == pb.view_id:
            rejected.append((a, b, "same_view_same_edge")); continue
        if box_iou(pa.box, pb.box) < min_same_iou:
            rejected.append((a, b, "same_edge_below_min_iou")); continue
        # Complete-link only: a SAME-b edge cannot make an unsupported
        # transitive bridge. Two components merge only when every cross-pair
        # has independently observed SAME evidence.
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            continue
        group_a = sorted(cid for cid in by_id if find(cid) == root_a)
        group_b = sorted(cid for cid in by_id if find(cid) == root_b)
        missing = [(left, right) for left in group_a for right in group_b
                   if edges.get(tuple(sorted((left, right)))) != "SAME"]
        if missing:
            rejected.append((a, b, "incomplete_same_clique"))
            continue
        union(a, b)

    groups={}
    for cid in sorted(by_id): groups.setdefault(find(cid), []).append(cid)
    components=[tuple(v) for _,v in sorted(groups.items())]
    blocked=set(); reasons=[]
    component_of={cid:i for i,c in enumerate(components) for cid in c}
    for (a,b),relation in sorted(edges.items()):
        if relation == "DIFFERENT" and component_of[a] == component_of[b]:
            i=component_of[a];blocked.add(i);reasons.append(f"internal_same_different_conflict:{a},{b}")

    # Two components with concerning overlap can both count only when every
    # concerning cross-pair has explicit DIFFERENT evidence.
    for i,a_group in enumerate(components):
        for j,b_group in enumerate(components[i+1:],i+1):
            concerning=[]
            for a in a_group:
                for b in b_group:
                    if box_iou(by_id[a].box,by_id[b].box) >= ambiguity_iou:
                        relation=edges.get(tuple(sorted((a,b))),"UNKNOWN")
                        if relation != "DIFFERENT": concerning.append((a,b))
            if concerning:
                blocked.update((i,j));ids=sorted({x for pair in concerning for x in pair})
                reasons.append("unresolved_cross_component_overlap:"+",".join(ids))
    accepted=tuple(c for i,c in enumerate(components) if i not in blocked)
    unresolved=tuple(c for i,c in enumerate(components) if i in blocked)
    return IdentityGraphResult(accepted,unresolved,tuple(rejected),tuple(sorted(set(reasons))))
