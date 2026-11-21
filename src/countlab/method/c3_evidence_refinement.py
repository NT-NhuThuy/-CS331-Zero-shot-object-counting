"""UNKNOWN-neutral projection for C3 operational inference.

The raw verifier event log is never mutated or discarded.
"""

def project_decisive_events(events):
    decisive = []
    ignored_refs = []
    for event in events:
        op = event["op"]
        unknown = (
            op in ("REGION_SEMANTIC", "REGION_WHOLE")
            and event["verdict"] == "UNKNOWN"
        ) or (
            op == "PAIR_IDENTITY"
            and event["relation"] == "UNKNOWN"
        )
        if unknown:
            ignored_refs.append(event["evidence_ref"])
        else:
            decisive.append(event)
    return decisive, sorted(ignored_refs)
