#!/usr/bin/env python3
"""Post-freeze ablations from one immutable bidirectional evidence bundle."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def met(rows,key):
 return {'n':len(rows),'mae':sum(abs(x[key]-x['gt']) for x in rows)/len(rows),'bias':sum(x[key]-x['gt'] for x in rows)/len(rows),'exact':sum(x[key]==x['gt'] for x in rows)/len(rows)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 root=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC');lvis=json.loads((root/'data/lvis_v1_val_source/lvis_v1_val.json').read_text());gt=Counter((int(z['image_id']),int(z['category_id'])) for z in lvis['annotations']);man=json.loads((a.experiment/'pool/evaluator_manifest.json').read_text())['rows'];pub={int(x['id']):x for x in json.loads((a.experiment/'pool/inference_coco.json').read_text())['images']};base=Counter(int(z['image_id']) for z in json.loads((a.experiment/'full_p0/predictions.json').read_text())['annotations']);q=json.loads(a.output.read_text())['queries'];rows=[]
 for r in man:
  i=int(r['id']);v=q[str(i)];b=base[i];removed=len(v['rejected_regions']);added=sum(x.get('role')=='rescue' for x in v['accepted_instances']);g=gt[(int(pub[i]['source_image_id']),int(pub[i]['lvis_category_id']))];rows.append({'status':r['query_status'],'gt':g,'baseline':b,'rescue_only':b+added,'rejection_only':b-removed,'full':b-removed+added})
 arms={k:{'all':met(rows,k),'present':met([x for x in rows if x['status']=='present'],k),'absent':met([x for x in rows if x['status']=='verified_absent'],k)} for k in ('baseline','rescue_only','rejection_only','full')}
 out={'status':'BIDIRECTIONAL_POSTFREEZE_ABLATION','uses_gt_for_inference':False,'output_sha256':sha(a.output),'arms':arms}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
