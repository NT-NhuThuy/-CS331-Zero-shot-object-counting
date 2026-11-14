#!/usr/bin/env python3
"""Deterministic GT-free evidence routing for dense proposal sets."""
import argparse,json
from collections import defaultdict
from pathlib import Path
def iou(a,b):
 x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3]);z=max(0,x1-x0)*max(0,y1-y0);u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z;return z/u if u else 0.
def route(tasks,budget):
 if budget<1:raise ValueError('budget must be positive')
 groups=defaultdict(list)
 for x in tasks:
  if x['role']=='baseline':groups[int(x['query_id'])].append(x)
 selected=set(x['task_id'] for x in tasks if x['role']=='rescue')
 reasons={x['task_id']:'rescue_always' for x in tasks if x['role']=='rescue'}
 for _,items in groups.items():
  if len(items)<=budget:
   for x in items:selected.add(x['task_id']);reasons[x['task_id']]='sparse_verify_all'
   continue
  risk={x['task_id']:max((iou(x['box_xyxy'],y['box_xyxy']) for y in items if y is not x),default=0.) for x in items}
  ranked=sorted(items,key=lambda x:(-risk[x['task_id']],float(x.get('proposal_score',0.0)),x['task_id']))
  for x in ranked[:budget]:selected.add(x['task_id']);reasons[x['task_id']]='dense_overlap_low_score_budget'
 out=[]
 for x in tasks:out.append({**x,'route_selected':x['task_id'] in selected,'route_reason':reasons.get(x['task_id'],'budget_deferred_unknown')})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--baseline-budget',type=int,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 x=json.loads(a.input.read_text());rows=route(x['tasks'],a.baseline_budget);payload={**x,'schema':'fqie_bidirectional_routed_manifest_v1','routing':{'uses_gt':False,'baseline_budget':a.baseline_budget,'dense_rank':['max_pairwise_iou_desc','proposal_score_asc','task_id_asc'],'deferred_policy':'UNKNOWN_RETAIN_BASELINE'},'tasks':rows};a.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps({'all_tasks':len(rows),'selected':sum(z['route_selected'] for z in rows),'deferred':sum(not z['route_selected'] for z in rows),'out':str(a.out)},indent=2))
if __name__=='__main__':main()
