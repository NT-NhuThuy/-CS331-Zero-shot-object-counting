"""Owner: Thúy. Complete-link SAME clustering with overlap ambiguity abstention.

Pair edges are precomputed GT-blind verifier outputs, not oracle instance IDs.
An UNKNOWN pair is never promoted into a merge. Results are conservative
candidate-graph decisions, not a guarantee of complete scene recall.
"""
from __future__ import annotations
from .schema import Proposal, box_iou


def decide_rescue_components(
    eligible: list[Proposal], edges: dict[tuple[str, str], str], *,
    min_same_iou: float = 0.10, ambiguous_iou: float = 0.30,
    use_identity: bool = True,
) -> tuple[list[list[Proposal]], list[Proposal], list[str]]:
    """Return independently admissible components, unresolved candidates, reasons.

    A complete-link cluster only forms if every pair has SAME, different view
    and nonzero reasonable geometric overlap in original image coordinates.
    Unresolved overlaps across clusters prevent *both* clusters from counting.
    """
    def relation(a: Proposal, b: Proposal) -> str:
        return edges.get(tuple(sorted((a.candidate_id, b.candidate_id))), "UNKNOWN")

    groups: list[list[Proposal]] = []
    for p in sorted(eligible, key=lambda z: z.candidate_id):
        placed = False
        if use_identity:
            for cluster in groups:
                if all(relation(p, q) == "SAME" and p.view_id != q.view_id
                       and box_iou(p.box, q.box) >= min_same_iou for q in cluster):
                    cluster.append(p)
                    placed = True
                    break
        if not placed:
            groups.append([p])

    blocked: set[int] = set()
    reasons: list[str] = []
    # If cross-component proposals overlap, counting both may double count.
    # Only explicit all-pairs DIFFERENT resolves this conflict.
    if use_identity:
        for i, group_a in enumerate(groups):
            for j, group_b in enumerate(groups[i+1:], i+1):
                concerning = [(a, b) for a in group_a for b in group_b
                              if box_iou(a.box, b.box) >= ambiguous_iou
                              and relation(a, b) != "DIFFERENT"]
                if concerning:
                    blocked.update((i, j))
                    reasons.append("unresolved_overlap:" + ",".join(sorted({x.candidate_id for pair in concerning for x in pair})))
    else:
        # Identity OFF is an explicit ablation: it can double count.
        pass
    accepted = [group for i, group in enumerate(groups) if i not in blocked]
    unknown = [p for i, group in enumerate(groups) if i in blocked for p in group]
    return accepted, unknown, reasons
