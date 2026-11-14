#!/usr/bin/env python3
"""GT-blind, monotone correction of frozen CountGD++ detections.

Every baseline proposal begins UNKNOWN and is retained in the point estimate.
Only explicit REJECTED evidence may remove it; only ACCEPTED, unmatched rescue
evidence may add an instance.  Thus missing evidence can never lower a count.
"""
from __future__ import annotations

DECISIONS={'ACCEPTED','REJECTED','UNKNOWN'}

def _decision(x):
    d=x.get('decision','UNKNOWN')
    if d not in DECISIONS: raise ValueError(f'invalid decision: {d}')
    return d

def reconcile(query_id:int, baseline:list[dict], rescue:list[dict]) -> dict:
    """Return an auditable interval and conservative corrected point count.

    Baseline ids are required. Rescue items may refer to a baseline id through
    ``matched_baseline_id``; matched rescues are evidence about that proposal,
    never a second instance.
    """
    ids=[str(x['instance_id']) for x in baseline]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate baseline instance_id')
    by_id={str(x['instance_id']):x for x in baseline}
    state={k:_decision(v) for k,v in by_id.items()}
    additions=[];unknown=[]
    for x in rescue:
        d=_decision(x);match=x.get('matched_baseline_id')
        if match is not None:
            match=str(match)
            if match not in by_id: raise ValueError(f'unknown matched_baseline_id: {match}')
            # Conflicting positive/negative proof is unresolved, not a vote.
            if state[match] != d and state[match] != 'UNKNOWN' and d != 'UNKNOWN': state[match]='UNKNOWN'
            elif d!='UNKNOWN': state[match]=d
        elif d=='ACCEPTED': additions.append(x)
        elif d=='UNKNOWN': unknown.append(x)
    accepted_base=[by_id[k] for k,d in state.items() if d=='ACCEPTED']
    rejected_base=[by_id[k] for k,d in state.items() if d=='REJECTED']
    unknown_base=[by_id[k] for k,d in state.items() if d=='UNKNOWN']
    lower=len(accepted_base)+len(additions)
    upper=lower+len(unknown_base)+len(unknown)
    # Retain unknown baseline detections in the estimate; reject only with proof.
    count=len(baseline)-len(rejected_base)+len(additions)
    return {'query_id':int(query_id),'count':count,'count_lower_bound':lower,
            'count_upper_bound':upper,'count_status':'exact' if not unknown_base and not unknown else 'partial',
            'accepted_instances':accepted_base+additions,'rejected_regions':rejected_base,
            'unknown_regions':unknown_base+unknown,'evidence_trace':[
                {'type':'bidirectional_controller','baseline_total':len(baseline),'rejected_baseline':len(rejected_base),
                 'accepted_rescues':len(additions),'unknown_baseline':len(unknown_base),'unknown_rescues':len(unknown)}],
            'query_abstained':False}
