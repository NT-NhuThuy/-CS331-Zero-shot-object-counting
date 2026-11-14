#!/usr/bin/env python3
"""Materialize an isolated unary-verifier run from a bidirectional manifest."""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path

ROOT=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC')
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--manifest',type=Path);a=p.parse_args();a.experiment=a.experiment.resolve();a.out=a.out.resolve()
 src=(a.manifest or (a.experiment/'bidirectional_verification_manifest.json')).resolve()
 if a.out.exists():raise FileExistsError(a.out)
 m=json.loads(src.read_text());a.out.mkdir(parents=True)
 source_status=json.loads((a.experiment/'consensus_lock.json').read_text())['status']
 for name in ('identity_cycle.py','common.py'):
  text=(a.experiment/name).read_text().replace(source_status,'BIDIRECTIONAL_UNARY_TASKS_LOCKED_BEFORE_GT')
  for tag in ('V11','V12','V13','V14','V15'):
   text=text.replace(tag+'_UNARY_IDENTITY_FROZEN_BEFORE_GT','BIDIRECTIONAL_UNARY_FROZEN_BEFORE_GT').replace(tag+'_WHOLE_INSTANCE_IDENTITY_FROZEN_BEFORE_GT','BIDIRECTIONAL_IDENTITY_FROZEN_BEFORE_GT')
  (a.out/name).write_text(text)
 public=json.loads((a.experiment/'pool/inference_coco.json').read_text())
 (a.out/'pool').mkdir();(a.out/'pool/inference_coco.json').write_text(json.dumps(public))
 tasks=[]
 for x in m['tasks']:
  if not x.get('route_selected',True):continue
  tasks.append({'candidate_id':x['task_id'],'query_id':x['query_id'],'box_xyxy':x['box_xyxy'],'source_image_id':x['source_image_id'],'support_views':[x['role']],'source_model':'CountGD++-frozen' if x['role']=='baseline' else 'CountGD++-tile'})
 cp=a.out/'consensus_candidates.jsonl';cp.write_text(''.join(json.dumps(x)+'\n' for x in tasks))
 (a.out/'consensus_lock.json').write_text(json.dumps({'status':'BIDIRECTIONAL_UNARY_TASKS_LOCKED_BEFORE_GT','candidates_sha256':sha(cp),'uses_gt':False,'source_manifest_sha256':sha(src)},indent=2)+'\n')
 # Immutable source images are shared read-only; no dataset/image copy.
 (a.out/'images').symlink_to(a.experiment/'images')
 (a.out/'bidirectional_task_index.json').write_text(json.dumps({'schema':'fqie_bidirectional_task_index_v1','uses_gt':False,'tasks':m['tasks']},indent=2)+'\n')
 print(json.dumps({'status':'BIDIRECTIONAL_UNARY_RUN_STAGED','tasks':len(tasks),'all_indexed_tasks':len(m['tasks']),'out':str(a.out)},indent=2))
if __name__=='__main__':main()
