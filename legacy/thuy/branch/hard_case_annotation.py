#!/usr/bin/env python3
"""Validator for human-annotated FQIE hard cases."""
from __future__ import annotations
import math
TARGET_TYPES={'whole_target'};NEG_TYPES={'confuser','part','texture','background','multiple_objects','other'};AMB_TYPES={'ambiguous','occluded','truncated'}
def box(b):return isinstance(b,list) and len(b)==4 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in b) and 0<=b[0]<b[2] and 0<=b[1]<b[3]
def validate_case(x):
 for k in ('query_id','image_width','image_height','query','target_instances','negative_regions','ambiguous_regions','annotator_status'):
  if k not in x:return False,f'missing:{k}'
 if not isinstance(x['query_id'],int) or not isinstance(x['image_width'],int) or not isinstance(x['image_height'],int) or x['image_width']<=0 or x['image_height']<=0:return False,'metadata'
 if x['annotator_status'] not in {'complete','needs_review'}:return False,'annotator_status'
 ids=set()
 for t in x['target_instances']:
  if not isinstance(t,dict) or not isinstance(t.get('instance_id'),str) or t['instance_id'] in ids or not box(t.get('box_xyxy')) or t.get('region_type') not in TARGET_TYPES:return False,'target_instance'
  ids.add(t['instance_id'])
 for n in x['negative_regions']:
  if not isinstance(n,dict) or not isinstance(n.get('region_id'),str) or not box(n.get('box_xyxy')) or n.get('region_type') not in NEG_TYPES:return False,'negative_region'
 for a in x['ambiguous_regions']:
  if not isinstance(a,dict) or not isinstance(a.get('region_id'),str) or not box(a.get('box_xyxy')) or a.get('region_type') not in AMB_TYPES:return False,'ambiguous_region'
 if x.get('gt_count')!=len(x['target_instances']):return False,'gt_count_mismatch'
 return True,None
def require_valid_case(x):
 ok,reason=validate_case(x)
 if not ok:raise ValueError(reason)
