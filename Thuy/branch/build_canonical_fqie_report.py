#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 specs={'v7':'fqie_contract_output_v7_20261113','v8':'fqie_contract_output_v8_20261113','v10':'fqie_contract_output_v10_20261113'};out={'schema':'fqie_canonical_validation_report_v1','uses_gt_for_inference':False,'cohorts':{}}
 for tag,d in specs.items():
  h=a.root/'target_aware_counting'/'output'/d;full=h/'full_cohort_output.json';cm=h/'full_cohort_count_metrics.json';im=h/'full_cohort_interval_metrics.json'
  if not cm.is_file(): cm=h.parent/'fqie_safe_iic_validation_20261112_v10'/'full_cohort_count_metrics_20261113.json'
  if not im.is_file(): im=h.parent/'fqie_safe_iic_validation_20261112_v10'/'full_cohort_interval_metrics_20261113.json'
  for f in (full,cm,im):
   if not f.is_file():raise FileNotFoundError(f)
  x=json.loads(full.read_text());out['cohorts'][tag]={'artifact_dir':str(h),'coverage':x['coverage'],'input_hashes':x['input_hashes'],'full_cohort_output_sha256':sha(full),'count_metrics':json.loads(cm.read_text()),'interval_metrics':json.loads(im.read_text()),'metric_artifact_hashes':{'count':sha(cm),'interval':sha(im)}}
 a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'CANONICAL_REPORT_BUILT','cohorts':list(out['cohorts']),'out':str(a.out)},indent=2))
if __name__=='__main__':main()
