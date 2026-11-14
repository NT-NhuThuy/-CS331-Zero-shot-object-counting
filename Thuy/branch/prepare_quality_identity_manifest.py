#!/usr/bin/env python3
"""Prepare a quality-gated identity manifest without mutating historical runs."""
import argparse,hashlib,json,math
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--min-support',type=int,default=3);a=p.parse_args();h=a.experiment
 if a.out.exists():raise FileExistsError(a.out)
 lock=json.loads((h/'consensus_lock.json').read_text());c=[json.loads(x) for x in (h/'consensus_candidates.jsonl').read_text().splitlines() if x.strip()];pub={int(x['id']):x for x in json.loads((h/'pool/inference_coco.json').read_text())['images']};rows=[]
 for x in c:
  if int(x.get('support_count',len(x.get('support_views',[]))))<a.min_support:continue
  q=int(x['query_id']);m=pub[q];b=[float(v) for v in x['box_xyxy']];valid=all(math.isfinite(v) for v in b) and 0<=b[0]<b[2]<=m['width'] and 0<=b[1]<b[3]<=m['height'];image=h/'images'/m['file_name']
  if not image.is_file():raise FileNotFoundError(image)
  rows.append({'task_id':x['candidate_id'],'candidate_id':x['candidate_id'],'query_id':q,'query':m['query_name'],'image_path':str(image),'image_sha256':sha(image),'box_xyxy':b,'geometry_valid':valid,'support_count':x.get('support_count',len(x.get('support_views',[]))),'support_views':x.get('support_views',[])})
 payload={'schema':'fqie_quality_gated_identity_manifest_v1','uses_gt':False,'status':'QUALITY_GATED_IDENTITY_MANIFEST_LOCKED_BEFORE_GT','min_support_views':a.min_support,'consensus_lock_sha256':sha(h/'consensus_lock.json'),'items':rows};a.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':payload['status'],'input_candidates':len(c),'kept':len(rows),'manifest_sha256':sha(a.out)},indent=2))
if __name__=='__main__':main()
