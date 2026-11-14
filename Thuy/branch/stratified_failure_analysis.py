#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 root=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC');h=a.experiment;lvis=json.loads((root/'data/lvis_v1_val_source/lvis_v1_val.json').read_text());gt={}
 for z in lvis['annotations']:gt[(int(z['image_id']),int(z['category_id']))]=gt.get((int(z['image_id']),int(z['category_id'])),0)+1
 man=json.loads((h/'pool/evaluator_manifest.json').read_text())['rows'];pub={int(x['id']):x for x in json.loads((h/'pool/inference_coco.json').read_text())['images']};pred=json.loads((h/'full_p0/predictions.json').read_text());base={}
 for z in pred['annotations']:base[int(z['image_id'])]=base.get(int(z['image_id']),0)+1
 ev=json.loads(a.evidence.read_text())['queries'];rows=[]
 for r in man:
  q=int(r['id']);m=pub[q];g=gt.get((int(m['source_image_id']),int(m['lvis_category_id'])),0);b=base.get(q,0);s=ev[str(q)]['count_lower_bound'];rows.append({'id':q,'status':r['query_status'],'stratum':'zero' if b==0 else 'nonzero','baseline':b,'safe':s,'gt':g,'delta':abs(s-g)-abs(b-g),'accepted':len(ev[str(q)]['accepted_instances'])})
 def summary(xs):return {'n':len(xs),'baseline_mae':sum(abs(x['baseline']-x['gt']) for x in xs)/len(xs) if xs else 0,'safe_mae':sum(abs(x['safe']-x['gt']) for x in xs)/len(xs) if xs else 0,'baseline_exact':sum(x['baseline']==x['gt'] for x in xs)/len(xs) if xs else 0,'safe_exact':sum(x['safe']==x['gt'] for x in xs)/len(xs) if xs else 0,'improved':sum(x['delta']<0 for x in xs),'worsened':sum(x['delta']>0 for x in xs),'accepted_queries':sum(x['accepted']>0 for x in xs),'accepted_instances':sum(x['accepted'] for x in xs)}
 out={'status':'FQIE_STRATIFIED_FAILURE_ANALYSIS','uses_gt_for_inference':False,'all':summary(rows),'by_baseline_evidence':{k:summary([x for x in rows if x['stratum']==k]) for k in ('zero','nonzero')},'by_status':{k:summary([x for x in rows if x['status']==k]) for k in ('present','verified_absent')}}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
