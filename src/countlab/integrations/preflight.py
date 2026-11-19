"""Explicit readiness audit for real online inference. No downloads, no GPU calls."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from .countgd_native import PROTOCOL_ID


def hash_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda:f.read(1024*1024), b''):
            h.update(buf)
    return h.hexdigest()


def inspect(config: dict) -> dict:
    from .countgd_official import PROTOCOL_ID as OFFICIAL_PROTOCOL_ID, inspect_assets
    if config.get('protocol_id') == OFFICIAL_PROTOCOL_ID:
        result = inspect_assets(config.get('official_countgd', {}))
        if config.get('verification_budget', 0) > 0 or config.get('identity', {}).get('enabled'):
            for pkg in ('transformers', 'qwen_vl_utils'):
                if importlib.util.find_spec(pkg) is None:
                    result['blockers'].append(f'Python dependency missing: {pkg}')
            qwen = config.get('qwen', {})
            if qwen.get('revision', 'main') in ('main', 'latest', 'REPLACE_WITH_EXACT_COMMIT_SHA', ''):
                result['blockers'].append('exact pinned Qwen3-VL revision not set')
        if config.get('identity', {}).get('enabled') and config['identity'].get('max_pairs', 0) < 1:
            result['blockers'].append('identity enabled but verification budget is zero')
        if result['blockers']:
            result['status'] = 'BLOCKED'
        return result
    blockers=[]; verified=[]
    if config.get('protocol_id')!=PROTOCOL_ID:
        blockers.append('Unexpected protocol: do not claim CountGD++ V14 parity')
    native=config.get('native_countgd',{})
    for field in ('repo','checkpoint','bert_dir'):
        value=native.get(field)
        if not value or str(value).startswith('/REPLACE'):
            blockers.append(f'{field}: requires local runtime path')
        elif not Path(value).exists():
            blockers.append(f'{field}: not found {value}')
        else:
            verified.append(f'{field}: {value}')
    repo=native.get('repo')
    if repo and Path(repo).is_dir():
        required=Path(repo)/'models/GroundingDINO/groundingdino_app.py'
        if not required.is_file():
            blockers.append('official CountGD++ groundingdino_app.py missing')
    checkpoint=native.get('checkpoint')
    if checkpoint and Path(checkpoint).is_file():
        verified.append('checkpoint_sha256: '+hash_file(Path(checkpoint)))
    for pkg in ('PIL','torch','transformers','qwen_vl_utils'):
        if importlib.util.find_spec(pkg) is None:
            blockers.append(f'Python dependency missing: {pkg}')
        else:
            verified.append(f'Python dependency available: {pkg}')
    if config.get('verification_budget',0)>0:
        qwen=config.get('qwen')
        if not qwen or qwen.get('revision','main') in ('main','latest','REPLACE_WITH_EXACT_COMMIT_SHA',''):
            blockers.append('exact pinned Qwen3-VL revision not set')
    if config.get('identity',{}).get('enabled') and config.get('identity',{}).get('max_pairs',0)<1:
        blockers.append('identity enabled but verification budget is zero')
    return {'status':'READY_FOR_GPU_SMOKE_NOT_MODEL_PARITY' if not blockers else 'BLOCKED',
            'protocol_id':config.get('protocol_id'),
            'blockers':blockers, 'verified':verified,
            'note':'Passing preflight does not imply GPU, checkpoint parity or benchmark accuracy'}


def main():
    p=argparse.ArgumentParser(description='Inspect local native CountGD++/Qwen runtime without running GPU')
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    result=inspect(json.loads(a.config.read_text()))
    if a.output:
        with a.output.open('x',encoding='utf-8') as f:
            json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 0 if result['status'].startswith('READY') else 2

if __name__=='__main__':
    raise SystemExit(main())
