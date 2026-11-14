#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from collections import defaultdict
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--evidence',type=Path);p.add_argument('--policy',choices=('lower_bound','preserve_baseline'),default='lower_bound');a=p.parse_args();h=a.experiment;root=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC');lvis=json.loads((root/'data/lvis_v1_val_source/lvis_v1_val.json').read_text());gt={}
 for z in lvis['annotations']:gt[(int(z['image_id']),int(z['category_id']))]=gt.get((int(z['image_id']),int(z['category_id'])),0)+1
 pub={int(x['id']):x for x in json.loads((h/'pool/inference_coco.json').read_text())['images']};man=json.loads((h/'pool/evaluator_manifest.json').read_text())['rows'];base=defaultdict(int)
 for z in json.loads((h/'full_p0/predictions.json').read_text())['annotations']:base[int(z['image_id'])]+=1
 evidence=a.evidence or (h/'evidence_output_safe_iic.json')
 payload=json.loads(evidence.read_text());safe_queries=payload['queries'];rows=[];missing=[]
 if a.policy=='preserve_baseline' and payload.get('schema')!='fqie_verified_instance_count_v1_full_cohort':
  raise ValueError('preserve_baseline requires a full-cohort contract output with evidence_source')
 for r in man:
  q=int(r['id']);v=safe_queries.get(str(q))
  if v is None: missing.append(q)
  m=pub[q];g=gt.get((int(m['source_image_id']),int(m['lvis_category_id'])),0);b=base.get(q,0);safe=(v['count_lower_bound'] if v else b) if a.policy=='lower_bound' else (v['count_lower_bound'] if v and v.get('evidence_source')=='rescue_verified' else b);rows.append({'id':q,'status':r['query_status'],'gt':g,'full':b,'safe_iic':safe,'upper':v.get('count_upper_bound',safe) if v else safe,'evidence_source':v.get('evidence_source','baseline_unverified') if v else 'baseline_unverified'})
 def met(k,ss):
  z=[x for x in rows if ss(x)];return {'n':len(z),'mae':sum(abs(x[k]-x['gt']) for x in z)/len(z),'bias':sum(x[k]-x['gt'] for x in z)/len(z),'exact':sum(x[k]==x['gt'] for x in z)/len(z)}
 out={'status':'SAFE_IIC_OFFLINE_COUNT_METRICS','policy':a.policy,'uses_gt_for_inference':False,'manifest_queries':len(man),'matched_queries':len(rows),'missing_evidence_queries':missing,'baseline':met('full',lambda x:1),'safe_iic':met('safe_iic',lambda x:1),'present_baseline':met('full',lambda x:x['status']=='present'),'present_safe_iic':met('safe_iic',lambda x:x['status']=='present'),'absent_baseline':met('full',lambda x:x['status']=='verified_absent'),'absent_safe_iic':met('safe_iic',lambda x:x['status']=='verified_absent'),'improved':sum(abs(x['safe_iic']-x['gt'])<abs(x['full']-x['gt']) for x in rows),'worsened':sum(abs(x['safe_iic']-x['gt'])>abs(x['full']-x['gt']) for x in rows)}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
