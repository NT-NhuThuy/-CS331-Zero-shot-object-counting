"""Owner: Thư. Frozen-evidence verification routing; no learned model is implied.

Future adapter: supply *GT-blind* risk_hint computed from held-out-trained
structural/semantic signals, never from GT counts or post-hoc errors.
"""
from .schema import Proposal


def select_for_verification(items: list[Proposal], budget: int | None = None) -> set[str]:
    if budget is not None and budget < 0:
        raise ValueError("verification budget must be non-negative")
    # Candidate uncertainty is a screening hint, not a calibrated risk score.
    def priority(p: Proposal) -> tuple:
        hint = p.risk_hint if p.risk_hint is not None else (1 - p.confidence if p.confidence is not None else 0.5)
        return (-hint, p.candidate_id)
    ranked = sorted(items, key=priority)
    selected = ranked if budget is None else ranked[:budget]
    return {x.candidate_id for x in selected}

