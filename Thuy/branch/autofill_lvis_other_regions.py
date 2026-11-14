#!/usr/bin/env python3
"""Add other-category LVIS boxes as explicit DEV negative regions.

These are not claimed to be hard confusers; they remain region_type=other and
require human review before semantic-confuser metrics.
"""
import argparse,json
from collections import defaultdict
from pathlib import Path
ROOT=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC')
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--pool',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();lvis=json.loads((ROOT/'data/lvis_v1_val_source/lvis_v1_val.json').read_text());pub={int(x['id']):x for x in json.loads(a.pool.read_text())['images']};cats=defaultdict(list)
 for z in lvis['annotations']:cats[int(z['image_id'])].append(z)
 rows=[]
 for line in a.input.read_text().splitlines():
  x=json.loads(line);m=pub[int(x['query_id'])];target={z['instance_id'] for z in x['target_instances']};neg=[]
  for z in cats[int(m['source_image_id'])]:
   if f"gt_{z['id']}" in target:continue
   neg.append({'region_id':f"other_{z['id']}",'box_xyxy':[float(z['bbox'][0]),float(z['bbox'][1]),float(z['bbox'][0]+z['bbox'][2]),float(z['bbox'][1]+z['bbox'][3])],'region_type':'other','source':'LVIS_OTHER_CATEGORY_DEV_ONLY','category_id':int(z['category_id'])})
  x['negative_regions']=neg;x['annotation_notes']=x.get('annotation_notes','')+' Other-category boxes auto-filled; semantic confuser/part-whole labels require human review.';x['annotator_status']='needs_review';rows.append(x)
 a.out.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows));print(json.dumps({'rows':len(rows),'negative_regions':sum(len(x['negative_regions']) for x in rows),'status':'DEV_ONLY_NEEDS_HUMAN_REVIEW'},indent=2))
if __name__=='__main__':main()
