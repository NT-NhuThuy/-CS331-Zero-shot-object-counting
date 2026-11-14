#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from fqie_contracts import require_valid_count
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();x=json.loads(a.output.read_text());m=json.loads(a.manifest.read_text())['rows'];expected={str(int(r['id'])) for r in m};queries=x.get('queries',{})
 if x.get('schema')!='fqie_verified_instance_count_v1_full_cohort':raise ValueError('schema')
 if set(queries)!=expected:raise ValueError(f'coverage expected={len(expected)} got={len(queries)}')
 accepted=0
 for q,v in queries.items():
  require_valid_count(v)
  for z in v['accepted_instances']:
   for f in ('box_xyxy','semantic','whole_instance','identity','view_provenance','evidence_refs','artifact_hashes'):
    if f not in z:raise ValueError(f'accepted:{q}:missing:{f}')
   accepted+=1
 print(json.dumps({'status':'FULL_COHORT_OUTPUT_VALID','queries':len(queries),'accepted_instances':accepted},indent=2))
if __name__=='__main__':main()
