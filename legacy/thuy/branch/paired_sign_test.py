#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
def exact_sign(xs):
 xs=[x for x in xs if x];n=len(xs);w=sum(x<0 for x in xs)
 if not n:return {'n_non_tie':0,'wins':0,'losses':0,'p_two_sided':1.0}
 k=min(w,n-w);p=sum(math.comb(n,i) for i in range(k+1))*2/(2**n);return {'n_non_tie':n,'wins':w,'losses':n-w,'p_two_sided':min(1.0,p)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--policy',choices=('lower_bound','preserve_baseline'),default='lower_bound');a=p.parse_args()
 root=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC');h=a.experiment;lvis=json.loads((root/'data/lvis_v1_val_source/lvis_v1_val.json').read_text());gt={}
 for z in lvis['annotations']:gt[(int(z['image_id']),int(z['category_id']))]=gt.get((int(z['image_id']),int(z['category_id'])),0)+1
 man=json.loads((h/'pool/evaluator_manifest.json').read_text())['rows'];pub={int(x['id']):x for x in json.loads((h/'pool/inference_coco.json').read_text())['images']};pred=json.loads((h/'full_p0/predictions.json').read_text());base={}
 for z in pred['annotations']:base[int(z['image_id'])]=base.get(int(z['image_id']),0)+1
 payload=json.loads(a.evidence.read_text());ev=payload['queries'];rows=[]
 if a.policy=='preserve_baseline' and payload.get('schema')!='fqie_verified_instance_count_v1_full_cohort':raise ValueError('preserve_baseline requires a full-cohort contract output with evidence_source')
 for r in man:
  q=int(r['id']);m=pub[q];g=gt.get((int(m['source_image_id']),int(m['lvis_category_id'])),0);v=ev.get(str(q));b=base.get(q,0);s=(v['count_lower_bound'] if v else b) if a.policy=='lower_bound' else (v['count_lower_bound'] if v and v.get('evidence_source')=='rescue_verified' else b);rows.append({'status':r['query_status'],'delta':abs(s-g)-abs(b-g),'base':b,'safe':s,'gt':g})
 out={'status':'FQIE_PAIRED_EXACT_SIGN_TEST','policy':a.policy,'all':exact_sign([x['delta'] for x in rows]),'present':exact_sign([x['delta'] for x in rows if x['status']=='present']),'absent':exact_sign([x['delta'] for x in rows if x['status']=='verified_absent']),'absent_false_positive':{'baseline':sum(x['base']>0 for x in rows if x['status']=='verified_absent'),'safe_iic':sum(x['safe']>0 for x in rows if x['status']=='verified_absent'),'n':sum(x['status']=='verified_absent' for x in rows)}}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
