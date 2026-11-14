#!/usr/bin/env python3
"""Build auditable count bounds from frozen candidate/identity evidence."""
from __future__ import annotations
import json
import hashlib
from collections import defaultdict
from pathlib import Path
from safe_iic import safe_cluster
from fqie_contracts import require_valid_candidate, require_valid_count

def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def build_query_output(query_id,candidates,pairs):
    result=safe_cluster(candidates,pairs);components=[x for x in result['components'] if x['query_id']==query_id];unknown=[x for x in result['unresolved'] if x['query_id']==query_id]
    lower=len(components);upper=lower+len(unknown)
    status='exact' if not unknown else 'partial'
    by_id={x['candidate_id']:x for x in candidates}
    accepted=[]
    for x in components:
        rep=by_id[x['representative']]
        accepted.append({'component_id':x['representative'],'members':x['members'],'support_edges':x['support_edges'],
                         'box_xyxy':rep['box_xyxy'],'semantic':rep['semantic'],'whole_instance':rep['whole_instance'],
                         'identity':rep['identity'],'source_image_id':rep['source_image_id'],'view_id':rep['view_id'],
                         'view_provenance':rep['view_provenance'],'evidence_refs':rep['evidence_refs'],
                         'artifact_hashes':rep['artifact_hashes']})
    trace=[{'type':'identity_component','component_id':x['representative'],'members':x['members'],'status':x['count_status'],
            'evidence_refs':by_id[x['representative']]['evidence_refs'],'artifact_hashes':by_id[x['representative']]['artifact_hashes']} for x in components]
    trace.extend({'type':'identity_ambiguity','a':x['a'],'b':x['b'],'status':'UNKNOWN'} for x in unknown)
    trace.extend({'type':'large_component','component_id':x['component_id'],'size':x['size'],
                  'status':x['status'],'threshold':result['large_component_threshold']} for x in result['large_components'] if x['component_id'] in {c['representative'] for c in components})
    output={'count':lower,'count_lower_bound':lower,'count_upper_bound':upper,'count_status':status,'accepted_instances':accepted,'rejected_regions':[],'unknown_regions':unknown,'evidence_trace':trace,'query_abstained':False}
    require_valid_count(output)
    return output

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--manifest',type=Path);a=p.parse_args();h=a.experiment
    if a.out.exists(): raise FileExistsError(f'refusing to overwrite existing output: {a.out}')
    required=('consensus_candidates.jsonl','identity_unary_decisions.jsonl','identity_pair_decisions.jsonl')
    missing=[name for name in required if not (h/name).is_file()]
    if missing: raise FileNotFoundError(f'evidence bundle incomplete; missing: {missing}')
    manifest=a.manifest or a.out.with_name(a.out.stem+'.input_manifest.json')
    if manifest.exists(): raise FileExistsError(f'refusing to overwrite existing manifest: {manifest}')
    manifest.write_text(json.dumps({'schema':'fqie_evidence_input_manifest_v1','uses_gt':False,'files':{name:_sha256(h/name) for name in required}},indent=2,sort_keys=True)+'\n')
    c={x['candidate_id']:x for x in map(json.loads,(h/'consensus_candidates.jsonl').read_text().splitlines()) if x};u={x['candidate_id']:x for x in map(json.loads,(h/'identity_unary_decisions.jsonl').read_text().splitlines()) if x}
    public={int(x['id']):x for x in json.loads((h/'pool/inference_coco.json').read_text())['images']} if (h/'pool/inference_coco.json').is_file() else {}
    candidates=[]
    for cid,x in c.items():
        d=u[cid];candidates.append({'candidate_id':cid,'query_id':int(x['query_id']),'view_provenance':x.get('support_views',[]),'semantic':d['semantic'],'whole_instance':d['whole']})
        candidates[-1].update({
            'box_xyxy':x['box_xyxy'],
            'source_image_id':int(x.get('source_image_id',public.get(int(x['query_id']),{}).get('source_image_id',x['query_id']))),
            'view_id':str((x.get('support_views') or ['unknown_view'])[0]),
            'proposal_score':float(x.get('proposal_score',x.get('max_score',x.get('score',0.0)))),
            'source_model':str(x.get('source_model','CountGD++-frozen')),
            'identity':'UNKNOWN',
            'evidence_refs':[f'consensus_candidates.jsonl:{cid}',f'identity_unary_decisions.jsonl:{cid}'],
            'artifact_hashes':{
                'consensus_candidates.jsonl':_sha256(h/'consensus_candidates.jsonl'),
                'identity_unary_decisions.jsonl':_sha256(h/'identity_unary_decisions.jsonl'),
                'identity_pair_decisions.jsonl':_sha256(h/'identity_pair_decisions.jsonl'),
            },
            'decision':'ACCEPTED' if d['semantic']=='YES' and d['whole']=='YES' else 'UNKNOWN',
        })
        require_valid_candidate(candidates[-1])
    pairs=[json.loads(z) for z in (h/'identity_pair_decisions.jsonl').read_text().splitlines() if z.strip()];byq=defaultdict(list)
    for x in candidates:byq[x['query_id']].append(x)
    payload={'schema':'fqie_verified_instance_count_v1','uses_gt':False,'queries':{str(q):build_query_output(q,xs,pairs) for q,xs in sorted(byq.items())}}
    a.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps({'queries':len(byq),'out':str(a.out)},indent=2))
if __name__=='__main__':main()
