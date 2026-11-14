#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();h=a.experiment
 names=['full_p0/lock.json','zero_tile_lock.json','consensus_lock.json','identity_final_freeze.json'];checks=[]
 for n in names:
  p=h/n
  if not p.is_file():raise FileNotFoundError(p)
  x=json.loads(p.read_text());checks.append({'file':n,'status':x.get('status'),'uses_gt':x.get('uses_gt'),'hash_fields':sorted(k for k in x if 'sha256' in k.lower() or 'hash' in k.lower())})
 if any(x['uses_gt'] is not False for x in checks):raise ValueError('uses_gt violation')
 print(json.dumps({'status':'FQIE_PROVENANCE_AUDIT_PASS','experiment':str(h),'checks':checks},indent=2))
if __name__=='__main__':main()
