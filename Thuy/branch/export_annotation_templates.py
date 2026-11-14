#!/usr/bin/env python3
"""Create human-annotation templates from a frozen DEV manifest."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();m=json.loads(a.manifest.read_text())
 rows=[]
 for x in m['rows']:
  rows.append({'query_id':x['query_id'],'source_image_id':x['source_image_id'],'query':x['query_name'],'image_width':None,'image_height':None,'gt_count':None,'target_instances':[],'negative_regions':[],'ambiguous_regions':[],'annotator_status':'needs_review','selection_reason':x['case_type'],'baseline_count':x['baseline_count'],'annotation_notes':''})
 a.out.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows));print(json.dumps({'templates':len(rows),'out':str(a.out)},indent=2))
if __name__=='__main__':main()
