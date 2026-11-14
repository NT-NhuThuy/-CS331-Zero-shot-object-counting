#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 specs={'v7':'fqie_contract_output_v7_20261113','v8':'fqie_contract_output_v8_20261113','v10':'fqie_contract_output_v10_20261113','v11':'fqie_contract_output_v11_20261113'};report={'schema':'fqie_final_cpu_validation_report_v1','uses_gt_for_inference':False,'cohorts':{}}
 for tag,d in specs.items():
  h=a.root/'target_aware_counting'/'output'/d;files={'full_cohort':'full_cohort_output.json','count':'full_cohort_count_metrics.json','interval':'full_cohort_interval_metrics.json','bootstrap':'paired_bootstrap_corrected_20261113.json','sign_test':'paired_sign_test_20261113.json','absent_safety':'absent_safety_audit_20261113.json'}
  # V10 metrics were initially stored in the experiment folder; canonicalize by reading the same values.
  paths={k:h/v for k,v in files.items()}
  if tag=='v10':
   exp=a.root/'target_aware_counting'/'output'/'fqie_safe_iic_validation_20261112_v10'
   paths['count']=exp/'full_cohort_count_metrics_20261113.json';paths['interval']=exp/'full_cohort_interval_metrics_20261113.json'
  if tag=='v11':
   paths['count']=h/'count_metrics.json';paths['interval']=h/'interval_metrics.json';paths['bootstrap']=h/'bootstrap.json';paths['sign_test']=h/'sign_test.json';paths['absent_safety']=h/'absent_safety.json'
  if any(not x.is_file() for x in paths.values()): raise FileNotFoundError([str(x) for x in paths.values() if not x.is_file()])
  report['cohorts'][tag]={'artifacts':{k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()},'coverage':load(paths['full_cohort'])['coverage'],'count_metrics':load(paths['count']),'interval_metrics':load(paths['interval']),'bootstrap':load(paths['bootstrap']),'sign_test':load(paths['sign_test']),'absent_safety':load(paths['absent_safety'])}
 a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'FINAL_CPU_VALIDATION_REPORT_BUILT','cohorts':list(report['cohorts']),'out':str(a.out)},indent=2))
if __name__=='__main__':main()
