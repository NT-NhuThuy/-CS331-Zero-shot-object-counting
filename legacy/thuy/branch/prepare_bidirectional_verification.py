#!/usr/bin/env python3
"""Freeze GT-blind verification tasks for both CountGD++ proposals and rescues."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def build_tasks(prediction:dict, public:dict[int,dict], rescues:list[dict]) -> list[dict]:
 tasks=[]
 for i,a in enumerate(prediction['annotations']):
  q=int(a['image_id']);x,y,w,h=map(float,a['bbox'])
  if q not in public or w<=0 or h<=0: raise ValueError(f'invalid baseline proposal {i}')
  tasks.append({'task_id':f'baseline:{q}:{a.get("id",i)}','query_id':q,'role':'baseline',
                'instance_id':f'baseline:{q}:{a.get("id",i)}','box_xyxy':[x,y,x+w,y+h],
                'source_image_id':int(public[q]['source_image_id']),'query_caption':public[q]['query_caption'],
                'proposal_score':float(a.get('score',0.0)),'decision':'UNKNOWN','evidence_refs':['full_p0/predictions.json']})
 for i,r in enumerate(rescues):
  q=int(r['query_id']);box=r['box_xyxy']
  if q not in public or len(box)!=4: raise ValueError(f'invalid rescue candidate {i}')
  tasks.append({'task_id':f'rescue:{r["candidate_id"]}','query_id':q,'role':'rescue',
                'instance_id':f'rescue:{r["candidate_id"]}','candidate_id':r['candidate_id'],'box_xyxy':box,
                'source_image_id':int(public[q]['source_image_id']),'query_caption':public[q]['query_caption'],
                'decision':'UNKNOWN','evidence_refs':[f'consensus_candidates.jsonl:{r["candidate_id"]}']})
 return sorted(tasks,key=lambda x:x['task_id'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();h=a.experiment
 if a.out.exists():raise FileExistsError(a.out)
 pred=h/'full_p0/predictions.json';pool=h/'pool/inference_coco.json';cand=h/'consensus_candidates.jsonl'
 for x in (pred,pool,cand):
  if not x.is_file():raise FileNotFoundError(x)
 public={int(x['id']):x for x in json.loads(pool.read_text())['images']}
 rescues=[json.loads(x) for x in cand.read_text().splitlines() if x.strip()]
 tasks=build_tasks(json.loads(pred.read_text()),public,rescues)
 payload={'schema':'fqie_bidirectional_verification_manifest_v1','uses_gt':False,
          'policy':{'unknown':'retain_baseline','rejected_baseline':'remove','accepted_unmatched_rescue':'add'},
          'input_hashes':{str(x.relative_to(h)):sha(x) for x in (pred,pool,cand)},'tasks':tasks}
 a.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'BIDIRECTIONAL_TASKS_FROZEN_BEFORE_GT','tasks':len(tasks),'baseline_tasks':sum(x['role']=='baseline' for x in tasks),'rescue_tasks':sum(x['role']=='rescue' for x in tasks),'out':str(a.out)},indent=2))
if __name__=='__main__':main()
