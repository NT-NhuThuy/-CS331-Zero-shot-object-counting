#!/usr/bin/env python3
"""Pure CPU contract validators for FQIE evidence records."""
from __future__ import annotations
import math

SEMANTIC={'YES','NO','UNKNOWN'}
WHOLE=SEMANTIC
IDENTITY={'SAME','DIFFERENT','UNKNOWN'}
DECISION={'ACCEPTED','REJECTED','UNKNOWN'}
COUNT_STATUS={'exact','partial','abstained'}

def _finite_box(box):
    return isinstance(box,list) and len(box)==4 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in box) and 0<=box[0]<box[2] and 0<=box[1]<box[3]

def validate_candidate(x):
    required=('candidate_id','query_id','box_xyxy','source_image_id','view_id','view_provenance','proposal_score','source_model','semantic','whole_instance','identity','evidence_refs','artifact_hashes','decision')
    missing=[k for k in required if k not in x]
    if missing: return False,f'missing:{missing}'
    if not isinstance(x['candidate_id'],str) or not x['candidate_id']:return False,'candidate_id'
    if not isinstance(x['query_id'],int) or not isinstance(x['source_image_id'],int):return False,'ids'
    if not _finite_box(x['box_xyxy']):return False,'box_xyxy'
    if not isinstance(x['view_id'],str) or not x['view_id']:return False,'view_id'
    if not isinstance(x['view_provenance'],list) or not x['view_provenance'] or not all(isinstance(v,str) and v for v in x['view_provenance']):return False,'view_provenance'
    if not isinstance(x['proposal_score'],(int,float)) or not math.isfinite(x['proposal_score']):return False,'proposal_score'
    if not isinstance(x['source_model'],str) or not x['source_model']:return False,'source_model'
    if x['semantic'] not in SEMANTIC:return False,'semantic'
    if x['whole_instance'] not in WHOLE:return False,'whole_instance'
    if x['identity'] not in IDENTITY:return False,'identity'
    if not isinstance(x['evidence_refs'],list) or not all(isinstance(v,str) and v for v in x['evidence_refs']):return False,'evidence_refs'
    if not isinstance(x['artifact_hashes'],dict) or not x['artifact_hashes'] or not all(isinstance(k,str) and isinstance(v,str) and v for k,v in x['artifact_hashes'].items()):return False,'artifact_hashes'
    if x['decision'] not in DECISION:return False,'decision'
    if x['decision']=='ACCEPTED' and not (x['semantic']=='YES' and x['whole_instance']=='YES'):
        return False,'accepted_requires_semantic_and_whole_yes'
    if x['identity']=='SAME' and x['decision']=='REJECTED':return False,'rejected_same_conflict'
    return True,None

def validate_count(x):
    required=('count','count_lower_bound','count_upper_bound','count_status','accepted_instances','rejected_regions','unknown_regions','evidence_trace','query_abstained')
    missing=[k for k in required if k not in x]
    if missing:return False,f'missing:{missing}'
    ints=('count','count_lower_bound','count_upper_bound')
    if any(not isinstance(x[k],int) or x[k]<0 for k in ints):return False,'counts'
    if not (x['count_lower_bound']<=x['count']<=x['count_upper_bound']):return False,'bounds'
    if x['count_status'] not in COUNT_STATUS:return False,'count_status'
    if not isinstance(x['query_abstained'],bool):return False,'query_abstained'
    if not all(isinstance(x[k],list) for k in ('accepted_instances','rejected_regions','unknown_regions','evidence_trace')):return False,'lists'
    if x['count_status']=='exact' and x['count_lower_bound']!=x['count_upper_bound']:return False,'exact_requires_equal_bounds'
    if x['count_status']=='abstained' and not x['query_abstained']:return False,'abstained_requires_query_abstained'
    return True,None

def require_valid_candidate(x):
    ok,reason=validate_candidate(x)
    if not ok:raise ValueError(reason)

def require_valid_count(x):
    ok,reason=validate_count(x)
    if not ok:raise ValueError(reason)
