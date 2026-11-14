#!/usr/bin/env python3
"""Conservative, non-transitive identity clustering.

This module is GT-free and consumes frozen candidate/pair evidence. It never
turns an UNKNOWN edge into a merge and never lets a single bridge edge connect
two otherwise incompatible clusters.
"""
from __future__ import annotations
from collections import defaultdict

def _views(x):
    p=x.get('view_provenance')
    if isinstance(p,(list,tuple,set)):return set(p)
    return {x.get('view_id')}

def safe_cluster(candidates: list[dict], pair_edges: list[dict], large_component_threshold: int = 20) -> dict:
    eligible=[x for x in candidates if x.get('semantic')=='YES' and x.get('whole_instance')=='YES']
    by_id={x['candidate_id']:x for x in eligible}
    yes=set(); different=set()
    for e in pair_edges:
        a,b=e.get('a'),e.get('b');
        if a not in by_id or b not in by_id or a==b:continue
        key=tuple(sorted((a,b)))
        if e.get('decision')=='YES':yes.add(key)
        elif e.get('decision')=='NO':different.add(key)
    groups=defaultdict(list)
    for x in eligible:groups[x['query_id']].append(x)
    components=[];unresolved=[]
    for qid,items in sorted(groups.items()):
        # Deterministic high-support order, then lexical tie-break.
        support={x['candidate_id']:sum(tuple(sorted((x['candidate_id'],y['candidate_id']))) in yes for y in items if y['candidate_id']!=x['candidate_id']) for x in items}
        clusters=[]
        for x in sorted(items,key=lambda z:(-support[z['candidate_id']],z['candidate_id'])):
            placed=False
            for c in clusters:
                ids=[z['candidate_id'] for z in c]
                if any(_views(z)&_views(x) for z in c):continue
                edges=[tuple(sorted((x['candidate_id'],cid))) for cid in ids]
                if all(e in yes and e not in different for e in edges):
                    c.append(x);placed=True;break
            if not placed:clusters.append([x])
        for c in clusters:
            components.append({'query_id':qid,'representative':sorted(z['candidate_id'] for z in c)[0],'members':sorted(z['candidate_id'] for z in c),'support_edges':sum(tuple(sorted((a,b))) in yes for i,a in enumerate([z['candidate_id'] for z in c]) for b in [z['candidate_id'] for z in c][i+1:]),'count_status':'exact' if len(c)==1 or all(tuple(sorted((a,b))) in yes for i,a in enumerate([z['candidate_id'] for z in c]) for b in [z['candidate_id'] for z in c][i+1:]) else 'partial'})
        # Candidate pairs not resolved into the same clique remain explicitly ambiguous.
        for i,a in enumerate(items):
            for b in items[i+1:]:
                key=tuple(sorted((a['candidate_id'],b['candidate_id'])))
                if key not in yes and key not in different:unresolved.append({'query_id':qid,'a':key[0],'b':key[1],'status':'UNKNOWN'})
    large=[{'component_id':x['representative'],'size':len(x['members']),'status':'LARGE_COMPONENT'}
           for x in components if len(x['members'])>=large_component_threshold]
    return {'components':components,'unresolved':unresolved,'eligible':len(eligible),'yes_edges':len(yes),'different_edges':len(different),
            'large_components':large,'large_component_threshold':large_component_threshold}
