#!/usr/bin/env python3
"""Pre-registered candidate quality gates; no GT-dependent decisions."""
def admissible(candidate, *, min_support_views=3):
    """Quality gate for a new validation run.

    Support count is provenance, not semantic proof. The gate only rejects
    candidates with insufficient independent views; semantic/whole checks stay
    separate and UNKNOWN remains UNKNOWN.
    """
    support=int(candidate.get('support_count',len(candidate.get('support_views',[]))))
    if support < min_support_views:return False,'INSUFFICIENT_VIEW_SUPPORT'
    if not all(float(v)==float(v) for v in candidate.get('box_xyxy',[])):return False,'NONFINITE_GEOMETRY'
    return True,None
