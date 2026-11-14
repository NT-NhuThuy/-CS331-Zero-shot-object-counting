#!/usr/bin/env python3
"""Post-hoc LVIS matching for rescue candidates; produces a small human-review queue."""
import argparse,json,random
from pathlib import Path
def iou(a,b):
 x0,y0=max(a[0],b[0]),max(a[1],b[1]);x1,y1=min(a[2],b[2]),min(a[3],b[3]);z=max(0,x1-x0)*max(0,y1-y0);u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z;return z/u if u else 0
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=20261113);a=p.parse_args();h=a.experiment;root=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC');lvis=json.loads((root/'data/lvis_v1_val_source/lvis_v1_val.json').read_text());im={int(x['id']):x for x in lvis['images']};cat={int(x['id']):x for x in lvis['annotations']};anns={}
 for z in lvis['annotations']:
  b=z['bbox'];anns.setdefault((int(z['image_id']),int(z['category_id'])),[]).append([b[0],b[1],b[0]+b[2],b[1]+b[3]])
 pub={int(x['id']):x for x in json.loads((h/'pool/inference_coco.json').read_text())['images']};c=[json.loads(x) for x in (h/'consensus_candidates.jsonl').read_text().splitlines() if x];u={x['candidate_id']:x for x in map(json.loads,(h/'identity_unary_decisions.jsonl').read_text().splitlines()) if x};rng=random.Random(a.seed);rows=[]
 for x in c:
  m=pub[int(x['query_id'])];boxes=anns.get((int(m['source_image_id']),int(m['lvis_category_id'])),[]);scores=sorted((iou(x['box_xyxy'],b) for b in boxes),reverse=True);best=scores[0] if scores else 0;status='NO_GT_MATCH' if best<.3 else 'BORDERLINE' if best<.7 else 'GT_MATCH';rows.append({'candidate_id':x['candidate_id'],'query_id':int(x['query_id']),'source_image_id':int(m['source_image_id']),'query':m['query_name'],'proposal_box':x['box_xyxy'],'best_gt_iou':best,'auto_status':status,'semantic':u.get(x['candidate_id'],{}).get('semantic','UNKNOWN'),'whole_instance':u.get(x['candidate_id'],{}).get('whole','UNKNOWN'),'human_review_required':status!='GT_MATCH' or u.get(x['candidate_id'],{}).get('semantic')=='YES'})
 accepted={z['component_id'] for q in json.loads(a.evidence.read_text())['queries'].values() for z in q.get('accepted_instances',[])}
 borderline=[x for x in rows if x['auto_status']=='BORDERLINE'];false_pool=[x for x in rows if x['auto_status']=='NO_GT_MATCH' and x['candidate_id'] not in accepted];sample=rng.sample(false_pool,min(3,len(false_pool)));accepted_rows=[x for x in rows if x['candidate_id'] in accepted]
 queue=[];seen=set()
 for x in accepted_rows+borderline+sample:
  if x['candidate_id'] not in seen:queue.append(x);seen.add(x['candidate_id'])
 out={'status':'FQIE_POSTHOC_CANDIDATE_AUDIT','uses_gt_for_inference':False,'candidates':rows,'human_review_queue':queue,'summary':{'total':len(rows),'gt_match':sum(x['auto_status']=='GT_MATCH' for x in rows),'borderline':sum(x['auto_status']=='BORDERLINE' for x in rows),'no_gt_match':sum(x['auto_status']=='NO_GT_MATCH' for x in rows),'accepted_candidates':len(accepted_rows),'human_review_count':len(queue)}};a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='candidates'},indent=2))
if __name__=='__main__':main()
