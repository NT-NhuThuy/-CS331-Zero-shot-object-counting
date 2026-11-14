#!/usr/bin/env python3
"""Assemble a coverage-complete, GT-blind FQIE output from baseline + rescue evidence."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    h=a.experiment
    manifest=h/'pool/evaluator_manifest.json'
    baseline=h/'full_p0/predictions.json'
    for x in (manifest,baseline,a.evidence):
        if not x.is_file(): raise FileNotFoundError(x)
    rows=json.loads(manifest.read_text())['rows']
    pred=json.loads(baseline.read_text())
    evidence=json.loads(a.evidence.read_text())['queries']
    by_image={}
    for z in pred['annotations']:
        by_image.setdefault(int(z['image_id']),[]).append(z)
    queries={}
    for r in rows:
        q=str(int(r['id']))
        # A probe with no admitted physical instance is inconclusive.  It must
        # not turn a baseline prediction into a purported verified rescue.
        if q in evidence and evidence[q]['accepted_instances']:
            queries[q]={**evidence[q],'evidence_source':'rescue_verified'}
            continue
        proposals=[]
        for z in by_image.get(int(r['id']),[]):
            b=z['bbox']; box=[b[0],b[1],b[0]+b[2],b[1]+b[3]]
            proposals.append({'region_id':f'baseline_q{q}_a{z.get("id",len(proposals))}',
                              'box_xyxy':box,'status':'UNKNOWN','source_model':'CountGD++-frozen',
                              'evidence_refs':['full_p0/predictions.json'],
                              'artifact_hashes':{'full_p0/predictions.json':sha(baseline)}})
        if not proposals:
            queries[q]={'count':0,'count_lower_bound':0,'count_upper_bound':0,'evidence_source':'baseline_zero',
                        'count_status':'exact','accepted_instances':[],
                        'rejected_regions':[],'unknown_regions':[],
                        'evidence_trace':[{'type':'baseline_zero_evidence','status':'EXACT_ZERO'}],
                        'query_abstained':False}
        else:
            # Preserve the frozen baseline scalar.  The [0, n] interval is an
            # evidence bound, not permission for downstream metrics to use 0.
            queries[q]={'count':len(proposals),'count_lower_bound':0,'count_upper_bound':len(proposals),'evidence_source':'baseline_unverified',
                        'count_status':'partial','accepted_instances':[],
                        'rejected_regions':[],'unknown_regions':proposals,
                        'evidence_trace':[{'type':'baseline_unverified_proposals','count':len(proposals),
                                           'status':'UNKNOWN'}],
                        'query_abstained':False}
    payload={'schema':'fqie_verified_instance_count_v1_full_cohort','uses_gt':False,
             'coverage':{'manifest_queries':len(rows),'output_queries':len(queries),
                         'rescue_evidence_queries':sum(str(r['id']) in evidence for r in rows),
                         'baseline_only_queries':sum(str(r['id']) not in evidence for r in rows)},
             'input_hashes':{'evaluator_manifest.json':sha(manifest),'predictions.json':sha(baseline),
                             'rescue_evidence.json':sha(a.evidence)},'queries':queries}
    if len(queries)!=len(rows) or set(queries)!={str(int(r['id'])) for r in rows}:
        raise ValueError('coverage mismatch')
    a.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'FULL_COHORT_ASSEMBLED','queries':len(queries),**payload['coverage'],'out':str(a.out)},indent=2))
if __name__=='__main__': main()
