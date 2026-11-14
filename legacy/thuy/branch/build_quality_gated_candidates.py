#!/usr/bin/env python3
"""Build a GT-free quality-gated candidate manifest for a new validation run."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--min-support',type=int,default=3);a=p.parse_args()
 rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()];kept=[x for x in rows if int(x.get('support_count',len(x.get('support_views',[]))))>=a.min_support]
 payload={'schema':'fqie_quality_gated_candidates_v1','status':'QUALITY_GATE_FROZEN_BEFORE_GT','uses_gt':False,'input_sha256':__import__('hashlib').sha256(a.input.read_bytes()).hexdigest(),'min_support_views':a.min_support,'input_candidates':len(rows),'kept_candidates':len(kept),'rows':kept};a.out.write_text('\n'.join(json.dumps(x,separators=(',',':')) for x in kept)+'\n');a.out.with_suffix('.lock.json').write_text(json.dumps({k:v for k,v in payload.items() if k!='rows'},indent=2)+'\n');print(json.dumps({k:payload[k] for k in ('status','min_support_views','input_candidates','kept_candidates')},indent=2))
if __name__=='__main__':main()
