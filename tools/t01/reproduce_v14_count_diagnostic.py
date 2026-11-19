"""Exact 32-image SHA+query overlap: R10 FP32 vs archived V14 AMP counts.
Not an official parity check: V14 lacks raw native indices and uses AMP.
Reads evaluator strata ONLY AFTER R10 predictions are sealed in ZIP.
"""
from __future__ import annotations
import argparse, hashlib, json, os, tarfile
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

R='UNICOUNT_T01_R10_PSEUDO3_DIAGNOSTIC/'
V='target_aware_counting/output/fqie_bidirectional_holdout_20261116_v14/'

def compare(r10_zip: Path,v14_tar: Path) -> dict:
    with ZipFile(r10_zip) as z:
        assert z.testzip() is None
        roster=json.loads(z.read(R+'roster_gt_free.json'))
        trace=[json.loads(line) for line in z.read(R+'per_pair_gpu_trace.jsonl').decode().splitlines()]
        assert len(roster)==len(trace)==32
    with tarfile.open(v14_tar,'r:gz') as t:
        def j(name):return json.load(t.extractfile(V+name))
        imgs={x['id']:x for x in j('pool/inference_coco.json')['images']}
        status={x['id']:x['query_status'] for x in j('pool/evaluator_manifest.json')['rows']}
        ctr=Counter(int(x['image_id']) for x in j('full_p0/predictions.json')['annotations'])
        args=j('full_p0/run/config_args_raw.json')
        assert args.get('amp') is True and args.get('num_pseudo_exemplars')==3
        rows=[]
        for r,c in zip(roster,trace):
            i=r['identity']['image_id']; p=imgs[i]
            assert os.path.basename(r['image'])==p['file_name'] and r['query']+' .'==p['query_caption']
            assert hashlib.sha256(t.extractfile(V+'images/'+p['file_name']).read()).hexdigest()==r['image_sha256']
            rows.append({'image_id':i,'query':r['query'],'stratum':status[i],
                         'r10_count':c['count'],'v14_count':ctr[i],'equal':c['count']==ctr[i]})
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
        return h.hexdigest()
    return {'schema':'t01_v14_amp_fp32_count_diagnostic_v1',
            'status':'DIAGNOSTIC_ONLY_NOT_G0_PARITY',
            'r10_zip_sha256':sha(r10_zip),'v14_tar_sha256':sha(v14_tar),
            'matched_image_sha_and_query':len(rows),
            'count_equal':sum(x['equal'] for x in rows),
            'by_stratum':{s:{'pairs':len(x),'counts_equal':sum(y['equal'] for y in x)} for s in ('present','verified_absent')
                         for x in [[q for q in rows if q['stratum']==s]]},
            'mismatches':[x for x in rows if not x['equal']],
            'mismatch_source':'V14 AMP vs R10 FP32 is not a matched operating point',
            'missing_from_v14':['raw second-pass native indices','float original boxes at parity precision','full runtime/dirty source attestation'],
            'g0':'HOLD','rows':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--r10-zip',type=Path,required=True)
    p.add_argument('--v14-tar',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    result=compare(a.r10_zip,a.v14_tar)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
    print(result['status'],result['count_equal'],'/',result['matched_image_sha_and_query'])
