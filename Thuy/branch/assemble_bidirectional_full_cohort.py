#!/usr/bin/env python3
"""Coverage-complete wrapper for a frozen bidirectional reconciliation."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--reconciliation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 rows=json.loads((a.experiment/'pool/evaluator_manifest.json').read_text())['rows'];pred=json.loads((a.experiment/'full_p0/predictions.json').read_text())['annotations'];by={}
 for z in pred:by.setdefault(int(z['image_id']),[]).append(z)
 frozen=json.loads(a.reconciliation.read_text())['queries'];q={}
 for r in rows:
  k=str(int(r['id']))
  if k in frozen:q[k]={**frozen[k],'evidence_source':'bidirectional_verified'};continue
  n=len(by.get(int(r['id']),[]));unknown=[] if not n else [{'instance_id':f'baseline:{k}:{i}','status':'UNKNOWN'} for i in range(n)]
  q[k]={'query_id':int(k),'count':n,'count_lower_bound':0,'count_upper_bound':n,'count_status':'partial' if n else 'exact','accepted_instances':[],'rejected_regions':[],'unknown_regions':unknown,'evidence_trace':[{'type':'unverified_baseline_fallback','status':'UNKNOWN'}],'query_abstained':False,'evidence_source':'baseline_unverified'}
 out={'schema':'fqie_bidirectional_count_v1_full_cohort','uses_gt':False,'queries':q}
 a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'queries':len(q),'verified_queries':len(frozen),'baseline_fallback_queries':len(q)-len(frozen),'out':str(a.out)},indent=2))
if __name__=='__main__':main()
