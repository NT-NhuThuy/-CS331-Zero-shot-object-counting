#!/usr/bin/env python3
"""Paired offline evaluator for the frozen FQIE ablation matrix.

Inference decisions are already frozen before this script runs. GT is read only
for post-hoc metrics and is never used to select a variant or threshold.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

ROOT=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/ZOC')

def counts(path: Path) -> dict[int,int]:
    d=json.loads(path.read_text());out={int(x['id']):0 for x in d.get('images',[])}
    for a in d.get('annotations',[]):out[int(a['image_id'])]=out.get(int(a['image_id']),0)+1
    return out

def promotions(path: Path) -> dict[int,int]:
    if not path.exists(): return {}
    out={}
    for line in path.read_text().splitlines():
        if line.strip():
            x=json.loads(line);out[int(x['query_id'])]=out.get(int(x['query_id']),0)+int(x.get('count',1))
    return out

def metrics(rows,key):
    def one(xs):
        return {'n':len(xs),'mae':sum(abs(x[key]-x['gt']) for x in xs)/len(xs),
                'bias':sum(x[key]-x['gt'] for x in xs)/len(xs),
                'exact':sum(x[key]==x['gt'] for x in xs)/len(xs)} if xs else {'n':0,'mae':None,'bias':None,'exact':None}
    return {'all':one(rows),'present':one([x for x in rows if x['status']=='present']),
            'absent':one([x for x in rows if x['status']=='verified_absent'])}

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--reference-promotions',type=Path);p.add_argument('--reference-gated-promotions',type=Path)
    a=p.parse_args();h=a.experiment
    for label,path in [('reference promotions',a.reference_promotions),('reference-gated promotions',a.reference_gated_promotions)]:
        if path is not None and not path.is_file():raise FileNotFoundError(f'{label} artifact not found: {path}')
    pool=json.loads((h/'pool/inference_coco.json').read_text());manifest=json.loads((h/'pool/evaluator_manifest.json').read_text())['rows']
    base=counts(h/'full_p0/predictions.json');iid=promotions(h/'identity_promotions.jsonl');ref=promotions(a.reference_promotions) if a.reference_promotions else {};gated=promotions(a.reference_gated_promotions) if a.reference_gated_promotions else {}
    # A=baseline; B=proposal-only is supplied by an optional proposal count file;
    # C=identity; D=reference-only; E=identity+reference. Missing optional files
    # remain explicit rather than silently being reported as zero.
    proposal=promotions(h/'proposal_only_promotions.jsonl')
    if not proposal: proposal=None
    source_by_q={int(x['id']):x for x in pool['images']};gt_by_src={}
    lvis=json.loads((ROOT/'data/lvis_v1_val_source/lvis_v1_val.json').read_text())
    for x in lvis['annotations']:
        k=(int(x['image_id']),int(x['category_id']));gt_by_src[k]=gt_by_src.get(k,0)+1
    rows=[]
    for r in manifest:
        q=int(r['id']);m=source_by_q[q];gt=gt_by_src.get((int(m['source_image_id']),int(m['lvis_category_id'])),0)
        row={'query_id':q,'status':r['query_status'],'gt':gt,'A':base.get(q,0),'C':base.get(q,0)+iid.get(q,0),'E':base.get(q,0)+gated.get(q,0)}
        row['B']=base.get(q,0)+(proposal or {}).get(q,0);row['D']=base.get(q,0)+ref.get(q,0);rows.append(row)
    variants=['A','B','C','D','E'];out={'status':'FQIE_ABLATION_MATRIX_OFFLINE','uses_gt_for_inference':False,'experiment':str(h),'reference_artifact':str(a.reference_promotions) if a.reference_promotions else None,'variants':{}}
    for v in variants:
        if v=='B' and proposal is None:out['variants'][v]={'status':'MISSING_PROPOSAL_ONLY_ARTIFACT'}
        elif v=='D' and a.reference_promotions is None:out['variants'][v]={'status':'MISSING_REFERENCE_PROMOTIONS_ARTIFACT'}
        elif v=='E' and a.reference_gated_promotions is None:out['variants'][v]={'status':'MISSING_REFERENCE_GATED_PROMOTIONS_ARTIFACT'}
        else:out['variants'][v]={'status':'READY','metrics':metrics(rows,v),'improved_vs_A':sum(abs(x[v]-x['gt'])<abs(x['A']-x['gt']) for x in rows),'worsened_vs_A':sum(abs(x[v]-x['gt'])>abs(x['A']-x['gt']) for x in rows)}
    (h/'ablation_matrix.json').write_text(json.dumps(out,indent=2)+'\n');(h/'ablation_rows.jsonl').write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in rows));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
