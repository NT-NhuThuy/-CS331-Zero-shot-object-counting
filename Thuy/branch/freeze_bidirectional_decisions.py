#!/usr/bin/env python3
"""Freeze unary evidence and reconcile baseline plus rescue proposals, GT-blind."""
from __future__ import annotations
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
from bidirectional_evidence_controller import reconcile

def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def iou(a,b):
 x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3]);inter=max(0,x1-x0)*max(0,y1-y0);aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1]);return inter/(aa+bb-inter) if aa+bb-inter else 0.
def decision(d):
 if d['semantic']=='YES' and d['whole']=='YES':return 'ACCEPTED'
 if d['semantic']=='NO' and d['whole']=='NO':return 'REJECTED'
 return 'UNKNOWN'
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();h=a.run
 if a.out.exists():raise FileExistsError(a.out)
 idx=json.loads((h/'bidirectional_task_index.json').read_text())['tasks'];raw={x['candidate_id']:x for x in map(json.loads,(h/'identity_unary_decisions.jsonl').read_text().splitlines()) if x};rows=[]
 for x in idx:
  d=raw.get(x['task_id'],{'semantic':'UNKNOWN','whole':'UNKNOWN'});rows.append({**x,'semantic':d['semantic'],'whole':d['whole'],'decision':decision(d)})
 byq=defaultdict(list)
 for x in rows:byq[int(x['query_id'])].append(x)
 outputs={}
 for q,items in byq.items():
  base=[x for x in items if x['role']=='baseline'];rescue=[x.copy() for x in items if x['role']=='rescue']
  for r in rescue:
   scores=[(iou(r['box_xyxy'],b['box_xyxy']),b['instance_id']) for b in base];best=max(scores,default=(0.,None))
   if best[0]>=.5:r['matched_baseline_id']=best[1]
  outputs[str(q)]=reconcile(q,base,rescue)
 payload={'schema':'fqie_bidirectional_count_v1','uses_gt':False,'policy':{'reject_requires':['semantic=NO','whole=NO'],'rescue_match_iou':.5,'unknown':'retain_baseline'},'input_hashes':{'task_index':sha(h/'bidirectional_task_index.json'),'unary_decisions':sha(h/'identity_unary_decisions.jsonl')},'queries':outputs}
 a.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'BIDIRECTIONAL_RECONCILIATION_FROZEN_BEFORE_GT','queries':len(outputs),'accepted':sum(x['decision']=='ACCEPTED' for x in rows),'rejected':sum(x['decision']=='REJECTED' for x in rows),'unknown':sum(x['decision']=='UNKNOWN' for x in rows),'out':str(a.out)},indent=2))
if __name__=='__main__':main()
