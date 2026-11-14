#!/usr/bin/env python3
"""Populate target-instance boxes/count from LVIS GT for DEV annotation only.

This does not invent hard negatives or part/whole labels; those remain
needs_review for human verification.
"""
import argparse,json
from collections import defaultdict
from pathlib import Path
ROOT=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC')
def main():
 p=argparse.ArgumentParser();p.add_argument('--template',type=Path,required=True);p.add_argument('--pool',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 lvis=json.loads((ROOT/'data/lvis_v1_val_source/lvis_v1_val.json').read_text());gt=defaultdict(list)
 for x in lvis['annotations']:gt[(int(x['image_id']),int(x['category_id']))].append(x)
 pub={int(x['id']):x for x in json.loads(a.pool.read_text())['images']};rows=[]
 for line in a.template.read_text().splitlines():
  x=json.loads(line);m=pub[int(x['query_id'])];anns=gt[(int(m['source_image_id']),int(m['lvis_category_id']))]
  x['image_width']=int(m['width']);x['image_height']=int(m['height']);x['gt_count']=len(anns);x['target_instances']=[{'instance_id':f"gt_{z['id']}",'box_xyxy':[float(z['bbox'][0]),float(z['bbox'][1]),float(z['bbox'][0]+z['bbox'][2]),float(z['bbox'][1]+z['bbox'][3])],'region_type':'whole_target','source':'LVIS_GT_DEV_ONLY'} for z in anns];x['annotation_notes']='Target boxes/count auto-filled from LVIS GT for DEV_ONLY; hard-negative and part/whole review still required';x['annotator_status']='needs_review';rows.append(x)
 a.out.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows));print(json.dumps({'rows':len(rows),'targets':sum(x['gt_count'] for x in rows),'status':'DEV_ONLY_GT_FILLED_NEEDS_HUMAN_REVIEW'},indent=2))
if __name__=='__main__':main()
