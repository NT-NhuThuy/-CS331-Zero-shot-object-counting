"""R10: CPU reconstruction from preserved raw GPU tensors; not independent V14 parity."""
import argparse, hashlib, io, json, sys
from pathlib import Path
from zipfile import ZipFile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from countlab.integrations.countgd_official import candidates_from_prediction, canonical_sha
PREFIX='UNICOUNT_T01_R10_PSEUDO3_DIAGNOSTIC/'

def audit(zip_path, dims_path):
    dims=json.loads(Path(dims_path).read_text())
    with ZipFile(zip_path) as z:
        def j(name):return json.loads(z.read(PREFIX+name))
        roster=j('roster_gt_free.json')
        trace=[json.loads(s) for s in z.read(PREFIX+'per_pair_gpu_trace.jsonl').decode().splitlines()]
        cand=[json.loads(s) for s in z.read(PREFIX+'candidates_diagnostic.jsonl').decode().splitlines()]
        digest=j('protocol_manifest.json')['protocol_sha256']
        assert len(roster)==len(trace)==len(cand)==32
        records=[]
        for i,(r,t,c) in enumerate(zip(roster,trace,cand)):
            assert r['index']==t['index']==c['index']==i
            image_id=str(r['identity']['image_id'])
            rawbytes=z.read(PREFIX+t['raw_sidecar'])
            assert hashlib.sha256(rawbytes).hexdigest()==t['raw_sidecar_sha256']
            raw=np.load(io.BytesIO(rawbytes),allow_pickle=False)
            for stage in ('first','second'):
                p={'boxes_cxcywh_normalized':raw[stage+'_boxes'].tolist(),
                   'token_probabilities':raw[stage+'_probs'].tolist(),
                   'input_ids':raw[stage+'_ids'].tolist()}
                assert canonical_sha(p)==t[stage+'_stage_raw_sha256'],f'{stage} SHA mismatch index {i}'
                if stage=='second': second=p
            ns=canonical_sha([digest,r['image_sha256'],image_id,r['query']])
            replay=candidates_from_prediction(second,width=dims[image_id]['width'],height=dims[image_id]['height'],
                                              namespace=ns,role='baseline',view_id='full')
            assert len(replay)==t['count']==len(c['candidates']), f'count error pair {i}'
            for a,b in zip(replay,c['candidates']):
                assert a['candidate_id']==b['candidate_id']
                assert a['native_query_index']==b['native_query_index']
                assert abs(a['confidence']-b['confidence'])<=1e-7
                assert all(abs(x-y)<=1e-6 for x,y in zip(a['box_xyxy'],b['box_xyxy']))
            records.append({'r10_index':i,'image_id':image_id,'query':r['query'],'count':len(replay),'passed':True})
    return {'schema':'t01_r10_cpu_raw_replay_v1','status':'RAW_CPU_PARSER_REPLAY_PASS_NOT_INDEPENDENT_PARITY',
            'pairs':len(records),'candidates':sum(x['count'] for x in records),
            'stage_raw_sha_checked':64,'g0':'HOLD','external_present_parity':'NOT_MEASURED',
            'external_absent_parity':'NOT_MEASURED','records':records}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--r10-zip',type=Path,required=True)
    p.add_argument('--image-dimensions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    result=audit(a.r10_zip,a.image_dimensions)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
    print(result['status'],result['pairs'],result['candidates'])
