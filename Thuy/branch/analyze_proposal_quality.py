#!/usr/bin/env python3
"""DEV-only proposal quality audit; GT is used only for post-hoc bins."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
def iou(a,b):
 x0,y0=max(a[0],b[0]),max(a[1],b[1]);x1,y1=min(a[2],b[2]),min(a[3],b[3]);z=max(0,x1-x0)*max(0,y1-y0);u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z;return z/u if u else 0
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();h=a.experiment
 ann={int(x['query_id']):x for x in (json.loads(z) for z in a.annotations.read_text().splitlines() if z.strip())};c=[json.loads(x) for x in (h/'consensus_candidates.jsonl').read_text().splitlines() if x];u={x['candidate_id']:x for x in (json.loads(z) for z in (h/'identity_unary_decisions.jsonl').read_text().splitlines() if z.strip())};rows=[]
 for x in c:
  gt=ann.get(int(x['query_id']),{}).get('target_instances',[]);best=max((iou(x['box_xyxy'],z['box_xyxy']) for z in gt),default=0);d=u[x['candidate_id']];rows.append({'candidate_id':x['candidate_id'],'query_id':x['query_id'],'score':x.get('max_score',0),'support_count':x.get('support_count',0),'best_iou':best,'semantic':d['semantic'],'whole':d['whole']})
 def bucket(v):return '0' if v==0 else '0-0.25' if v<.25 else '0.25-0.5' if v<.5 else '0.5-0.75' if v<.75 else '0.75-1'
 out={'status':'FQIE_PROPOSAL_QUALITY_DEV_AUDIT','uses_gt_for_inference':False,'candidates':len(rows),'iou_bins':Counter(bucket(x['best_iou']) for x in rows),'support_bins':Counter(x['support_count'] for x in rows),'mean_iou_by_support':{str(k):sum(x['best_iou'] for x in rows if x['support_count']==k)/sum(x['support_count']==k for x in rows) for k in sorted({x['support_count'] for x in rows})}}
 (a.out).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
