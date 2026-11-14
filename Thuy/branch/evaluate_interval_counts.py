#!/usr/bin/env python3
"""Evaluate partial count intervals without collapsing them to exact counts."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--evidence',type=Path);a=p.parse_args();h=a.experiment;root=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC');lvis=json.loads((root/'data/lvis_v1_val_source/lvis_v1_val.json').read_text());gt={}
 for z in lvis['annotations']:gt[(int(z['image_id']),int(z['category_id']))]=gt.get((int(z['image_id']),int(z['category_id'])),0)+1
 pub={int(x['id']):x for x in json.loads((h/'pool/inference_coco.json').read_text())['images']};man=json.loads((h/'pool/evaluator_manifest.json').read_text())['rows'];q=json.loads((a.evidence or (h/'evidence_output_safe_iic.json')).read_text())['queries'];rows=[]
 for r in man:
  m=pub[int(r['id'])];g=gt.get((int(m['source_image_id']),int(m['lvis_category_id'])),0);v=q.get(str(r['id']))
  if not v:continue
  lo,hi=int(v['count_lower_bound']),int(v['count_upper_bound']);rows.append({'id':int(r['id']),'status':r['query_status'],'gt':g,'lower':lo,'upper':hi,'covered':lo<=g<=hi,'status_count':v['count_status']})
 def summary(xs):return {'n':len(xs),'interval_coverage':sum(x['covered'] for x in xs)/len(xs) if xs else 0,'mean_width':sum(x['upper']-x['lower'] for x in xs)/len(xs) if xs else 0,'lower_mae':sum(abs(x['lower']-x['gt']) for x in xs)/len(xs) if xs else 0,'upper_mae':sum(abs(x['upper']-x['gt']) for x in xs)/len(xs) if xs else 0}
 out={'status':'FQIE_INTERVAL_COUNT_DEV_METRICS','uses_gt_for_inference':False,'all':summary(rows),'present':summary([x for x in rows if x['status']=='present']),'absent':summary([x for x in rows if x['status']=='verified_absent']),'rows':rows};a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
