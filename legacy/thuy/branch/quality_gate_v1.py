#!/usr/bin/env python3
"""Frozen GT-blind admission gate for the next replication."""
CONFIG={'version':'quality_gate_v1','min_support_views':3,'require_semantic':'YES','require_whole_instance':'YES','unknown_policy':'reject'}
def admit(candidate, config=CONFIG):
 support=int(candidate.get('support_count',len(candidate.get('support_views',[]))))
 if support<int(config['min_support_views']):return False,'INSUFFICIENT_VIEW_SUPPORT'
 box=candidate.get('box_xyxy',[])
 if len(box)!=4 or not all(isinstance(v,(int,float)) for v in box) or not 0<=box[0]<box[2] or not 0<=box[1]<box[3]:return False,'INVALID_GEOMETRY'
 if candidate.get('semantic')!=config['require_semantic']:return False,'SEMANTIC_NOT_YES'
 if candidate.get('whole_instance',candidate.get('whole'))!=config['require_whole_instance']:return False,'WHOLE_INSTANCE_NOT_YES'
 return True,None
