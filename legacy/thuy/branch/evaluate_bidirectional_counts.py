#!/usr/bin/env python3
"""Post-freeze GT-only evaluation of a full bidirectional contract."""
import argparse,json
from collections import Counter
from pathlib import Path
def met(rows,key):
 return {'n':len(rows),'mae':sum(abs(x[key]-x['gt']) for x in rows)/len(rows),'bias':sum(x[key]-x['gt'] for x in rows)/len(rows),'exact':sum(x[key]==x['gt'] for x in rows)/len(rows)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC')
 lvis=json.loads((root/'data/lvis_v1_val_source/lvis_v1_val.json').read_text());gt=Counter((int(z['image_id']),int(z['category_id'])) for z in lvis['annotations']);man=json.loads((a.experiment/'pool/evaluator_manifest.json').read_text())['rows'];pub={int(x['id']):x for x in json.loads((a.experiment/'pool/inference_coco.json').read_text())['images']};base=Counter(int(z['image_id']) for z in json.loads((a.experiment/'full_p0/predictions.json').read_text())['annotations']);q=json.loads(a.output.read_text())['queries'];rows=[{'status':r['query_status'],'gt':gt[(int(pub[int(r['id'])]['source_image_id']),int(pub[int(r['id'])]['lvis_category_id']))],'baseline':base[int(r['id'])],'method':q[str(int(r['id']))]['count']} for r in man]
 out={'status':'BIDIRECTIONAL_OFFLINE_METRICS','uses_gt_for_inference':False,'baseline':met(rows,'baseline'),'bidirectional':met(rows,'method'),'present_baseline':met([x for x in rows if x['status']=='present'],'baseline'),'present_bidirectional':met([x for x in rows if x['status']=='present'],'method'),'absent_baseline':met([x for x in rows if x['status']=='verified_absent'],'baseline'),'absent_bidirectional':met([x for x in rows if x['status']=='verified_absent'],'method'),'improved':sum(abs(x['method']-x['gt'])<abs(x['baseline']-x['gt']) for x in rows),'worsened':sum(abs(x['method']-x['gt'])>abs(x['baseline']-x['gt']) for x in rows)}
 a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
