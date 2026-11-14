#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 h=a.experiment;pub={int(x['id']):x for x in json.loads((h/'pool/inference_coco.json').read_text())['images']};c={x['candidate_id']:x for x in map(json.loads,(h/'consensus_candidates.jsonl').read_text().splitlines()) if x};u={x['candidate_id']:x for x in map(json.loads,(h/'identity_unary_decisions.jsonl').read_text().splitlines()) if x};q={}
 for cid,x in c.items():
  z=pub[int(x['query_id'])];q.setdefault(int(x['query_id']),{'query_id':int(x['query_id']),'source_image_id':int(z['source_image_id']),'query':z['query_name'],'image_width':int(z['width']),'image_height':int(z['height']),'gt_count':None,'target_instances':[],'negative_regions':[],'ambiguous_regions':[],'annotator_status':'needs_review','selection_reason':'rescue_consensus','baseline_count':None,'candidate_regions':[],'annotation_notes':''})['candidate_regions'].append({'candidate_id':cid,'box_xyxy':x['box_xyxy'],'support_views':x.get('support_views',[]),'semantic':u.get(cid,{}).get('semantic','UNKNOWN'),'whole_instance':u.get(cid,{}).get('whole','UNKNOWN')})
 a.out.write_text(''.join(json.dumps(v,sort_keys=True)+'\n' for _,v in sorted(q.items())));print(json.dumps({'status':'RESCUE_ANNOTATION_TEMPLATE_CREATED','queries':len(q),'candidates':sum(len(x['candidate_regions']) for x in q.values()),'out':str(a.out)},indent=2))
if __name__=='__main__':main()
