#!/usr/bin/env python3
"""Freeze deterministic DEV hard-case selection before FQIE evaluation."""
import argparse,hashlib,json
from pathlib import Path
ALLOWED={'zero_present','zero_absent','dense_present','nonzero_present','nonzero_absent'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 src=json.loads(a.manifest.read_text());rows=src.get('rows',[])
 if src.get('status')!='DEV_ONLY_GT_STRATIFIED_BEFORE_HUMAN_ANNOTATION':raise RuntimeError('manifest must be DEV-only')
 if not rows:raise RuntimeError('empty manifest')
 if any(x.get('case_type') not in ALLOWED for x in rows):raise RuntimeError('unknown case type')
 counts={k:sum(x['case_type']==k for x in rows) for k in sorted(ALLOWED)}
 payload={'schema':'fqie_hard_case_selection_lock_v1','status':'DEV_HARD_CASE_SELECTION_FROZEN_BEFORE_FQIE','uses_gt_for_inference':False,'source_manifest_sha256':sha(a.manifest),'selection_rule':'all rows from baseline-stratified DEV manifest; no FQIE outputs used','case_counts':counts,'rows':rows}
 a.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':payload['status'],'case_counts':counts,'sha256':sha(a.out)},indent=2))
if __name__=='__main__':main()
