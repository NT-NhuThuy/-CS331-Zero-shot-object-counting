#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
def wilson(k,n,z=1.959963984540054):
 if not n:return [0,0]
 p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-h,c+h]
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--policy',choices=('lower_bound','preserve_baseline'),default='lower_bound');a=p.parse_args()
 root=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC');h=a.experiment;lvis=json.loads((root/'data/lvis_v1_val_source/lvis_v1_val.json').read_text());gt={}
 for z in lvis['annotations']:gt[(int(z['image_id']),int(z['category_id']))]=gt.get((int(z['image_id']),int(z['category_id'])),0)+1
 man=json.loads((h/'pool/evaluator_manifest.json').read_text())['rows'];pub={int(x['id']):x for x in json.loads((h/'pool/inference_coco.json').read_text())['images']};pred=json.loads((h/'full_p0/predictions.json').read_text());base={}
 for z in pred['annotations']:base[int(z['image_id'])]=base.get(int(z['image_id']),0)+1
 payload=json.loads(a.evidence.read_text());ev=payload['queries']
 if a.policy=='preserve_baseline' and payload.get('schema')!='fqie_verified_instance_count_v1_full_cohort':raise ValueError('preserve_baseline requires a full-cohort contract output with evidence_source')
 absent=[r for r in man if r['query_status']=='verified_absent'];b=sum(base.get(int(r['id']),0)>0 for r in absent)
 def safe(r):
  q=str(int(r['id']));v=ev.get(q);fallback=base.get(int(r['id']),0)
  return (v['count_lower_bound'] if v else fallback) if a.policy=='lower_bound' else (v['count_lower_bound'] if v and v.get('evidence_source')=='rescue_verified' else fallback)
 s=sum(safe(r)>0 for r in absent);n=len(absent);br=b/n;sr=s/n
 out={'status':'FQIE_ABSENT_SAFETY_AUDIT','policy':a.policy,'n_absent':n,'baseline':{'false_promotion':b,'rate':br,'wilson95':wilson(b,n)},'safe_iic':{'false_promotion':s,'rate':sr,'wilson95':wilson(s,n)},'risk_difference_safe_minus_baseline':sr-br,'note':'Non-inferiority requires a pre-registered margin; this report does not choose one.'}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
