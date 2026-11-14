#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC');h=a.experiment;lvis=json.loads((root/'data/lvis_v1_val_source/lvis_v1_val.json').read_text());gt={}
 for z in lvis['annotations']:gt[(int(z['image_id']),int(z['category_id']))]=gt.get((int(z['image_id']),int(z['category_id'])),0)+1
 man=json.loads((h/'pool/evaluator_manifest.json').read_text())['rows'];pub={int(x['id']):x for x in json.loads((h/'pool/inference_coco.json').read_text())['images']};pred=json.loads((h/'full_p0/predictions.json').read_text());base={}
 for z in pred['annotations']:base[int(z['image_id'])]=base.get(int(z['image_id']),0)+1
 ev=json.loads(a.evidence.read_text())['queries'];rows=[]
 for r in man:
  v=ev.get(str(int(r['id'])));
  if not v or v.get('evidence_source')!='rescue_verified':continue
  q=int(r['id']);m=pub[q];g=gt.get((int(m['source_image_id']),int(m['lvis_category_id'])),0);b=base.get(q,0);s=v['count_lower_bound'];rows.append({'status':r['query_status'],'gt':g,'baseline':b,'rescue':s})
 def met(xs,k):return {'n':len(xs),'mae':sum(abs(x[k]-x['gt']) for x in xs)/len(xs) if xs else 0,'bias':sum(x[k]-x['gt'] for x in xs)/len(xs) if xs else 0,'exact':sum(x[k]==x['gt'] for x in xs)/len(xs) if xs else 0}
 out={'status':'FQIE_RESCUE_ONLY_METRICS','uses_gt_for_inference':False,'rescue_queries':len(rows),'baseline':met(rows,'baseline'),'rescue':met(rows,'rescue'),'present_baseline':met([x for x in rows if x['status']=='present'],'baseline'),'present_rescue':met([x for x in rows if x['status']=='present'],'rescue'),'absent_baseline':met([x for x in rows if x['status']=='verified_absent'],'baseline'),'absent_rescue':met([x for x in rows if x['status']=='verified_absent'],'rescue')}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
