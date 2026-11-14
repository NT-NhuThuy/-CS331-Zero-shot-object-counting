#!/usr/bin/env python3
"""Build a DEV-ONLY hard-case manifest from a frozen baseline and LVIS GT.

GT is used only to label/stratify a development annotation pool; the manifest
must never be reused as a blind test cohort.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from collections import defaultdict
ROOT=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();h=a.experiment
 pred=json.loads((h/'full_p0/predictions.json').read_text());pub={int(x['id']):x for x in json.loads((h/'pool/inference_coco.json').read_text())['images']};rows=json.loads((h/'pool/evaluator_manifest.json').read_text())['rows'];lvis=json.loads((ROOT/'data/lvis_v1_val_source/lvis_v1_val.json').read_text());gt=defaultdict(list)
 for x in lvis['annotations']:gt[(int(x['image_id']),int(x['category_id']))].append(x)
 counts=defaultdict(int)
 for x in pred['annotations']:counts[int(x['image_id'])]+=1
 out=[]
 for r in rows:
  q=int(r['id']);m=pub[q];g=gt[(int(m['source_image_id']),int(m['lvis_category_id']))];p0=counts[q]
  if p0==0 and g:kind='zero_present'
  elif p0==0 and not g:kind='zero_absent'
  elif p0>0 and g and g.__len__()>=10:kind='dense_present'
  elif p0>0 and g:kind='nonzero_present'
  else:kind='nonzero_absent'
  out.append({'query_id':q,'source_image_id':m['source_image_id'],'lvis_category_id':m['lvis_category_id'],'query_name':m['query_name'],'baseline_count':p0,'gt_count_dev_only':len(g),'case_type':kind,'requires_human_region_annotation':kind in {'zero_present','zero_absent','dense_present'}})
 payload={'schema':'fqie_hard_case_dev_manifest_v1','status':'DEV_ONLY_GT_STRATIFIED_BEFORE_HUMAN_ANNOTATION','uses_gt_for_inference':False,'baseline_prediction_sha256':sha(h/'full_p0/predictions.json'),'rows':out}
 a.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps({'rows':len(out),'case_counts':{k:sum(x['case_type']==k for x in out) for k in sorted({x['case_type'] for x in out})},'out':str(a.out)},indent=2))
if __name__=='__main__':main()
