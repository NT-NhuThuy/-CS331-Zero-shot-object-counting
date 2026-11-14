#!/usr/bin/env python3
import argparse,json,hashlib,random,math
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=20261113);p.add_argument('--resamples',type=int,default=10000);p.add_argument('--policy',choices=('lower_bound','preserve_baseline'),default='lower_bound');a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 root=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC');h=a.experiment
 lvis=json.loads((root/'data/lvis_v1_val_source/lvis_v1_val.json').read_text());gt={}
 for z in lvis['annotations']:gt[(int(z['image_id']),int(z['category_id']))]=gt.get((int(z['image_id']),int(z['category_id'])),0)+1
 man=json.loads((h/'pool/evaluator_manifest.json').read_text())['rows'];pub={int(x['id']):x for x in json.loads((h/'pool/inference_coco.json').read_text())['images']};pred=json.loads((h/'full_p0/predictions.json').read_text());base={}
 for z in pred['annotations']:base[int(z['image_id'])]=base.get(int(z['image_id']),0)+1
 payload=json.loads(a.evidence.read_text());ev=payload['queries'];rows=[]
 if a.policy=='preserve_baseline' and payload.get('schema')!='fqie_verified_instance_count_v1_full_cohort':raise ValueError('preserve_baseline requires a full-cohort contract output with evidence_source')
 for r in man:
  q=int(r['id']);m=pub[q];g=gt.get((int(m['source_image_id']),int(m['lvis_category_id'])),0);v=ev.get(str(q));b=base.get(q,0);s=(v['count_lower_bound'] if v else b) if a.policy=='lower_bound' else (v['count_lower_bound'] if v and v.get('evidence_source')=='rescue_verified' else b);rows.append({'cluster':int(m['source_image_id']),'status':r['query_status'],'delta':abs(s-g)-abs(b-g)})
 groups={}
 for x in rows:groups.setdefault(x['cluster'],[]).append(x)
 rng=random.Random(a.seed);clusters=list(groups.values())
 def stat(sample,status=None):
  xs=[x for c in sample for x in c if status is None or x['status']==status]
  return sum(x['delta'] for x in xs)/len(xs) if xs else float('nan')
 def ci(status):
  vals=[stat([rng.choice(clusters) for _ in clusters],status) for _ in range(a.resamples)];vals.sort();n=len(vals);return {'n_queries':sum(x['status']==status for x in rows) if status else len(rows),'delta_mae':stat(clusters,status),'bootstrap95':[vals[int(.025*n)],vals[int(.975*n)]],'bootstrap_fraction_delta_lt_zero':sum(v<0 for v in vals)/n}
 out={'status':'FQIE_PAIRED_BOOTSTRAP_CLUSTERED_BY_SOURCE_IMAGE','policy':a.policy,'seed':a.seed,'resamples':a.resamples,'clusters':len(clusters),'rows':len(rows),'all':ci(None),'present':ci('present'),'absent':ci('verified_absent')}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
