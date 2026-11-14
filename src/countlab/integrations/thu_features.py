"""GT-free, exact-key join to Thư's frozen V4/A3 node features.

This is a real adapter for legacy/thu/.../frozen_v4_a3_unified_nodes.npz.
It is NOT a reimplementation of the trained Joint V3.2 scalar count head,
and the supplied 1301-pair FSC features are EXPOSED research data. Never use
these frozen features to tune a new paper holdout or imply LVIS transfer.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from countlab.method.schema import _validate_no_gt, parse_input


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attach_features(manifest: dict, npz: Path, candidate_map: dict,
                    calibration: dict | None = None) -> dict:
    """Join query-local node index to candidate id under a frozen source mapping.

    candidate_map: {'candidate_id': {'node_graph_id': int, 'raw_query_index': int}}
    calibration is optional and must be locked on *new TRAIN/CAL*, never FSC
    TEST. Without calibration the features remain diagnostic only; they do not
    alter the unified router or point count.
    """
    import numpy as np
    import math
    _validate_no_gt(manifest)
    _validate_no_gt(candidate_map)
    if manifest.get('schema') != 'countlab_candidate_pool_v1':
        raise ValueError('wrong input schema')
    with np.load(npz, allow_pickle=False) as features:
        required={'node_graph_id','raw_query_index','v4_contribution','a3_score'}
        if set(features) != required:
            raise ValueError('frozen feature keys differ; version new data')
        count=len(features['node_graph_id'])
        if any(len(features[k])!=count for k in required):
            raise ValueError('unaligned feature arrays')
        index={}
        for i in range(count):
            key=(int(features['node_graph_id'][i]), int(features['raw_query_index'][i]))
            if key in index:
                raise ValueError(f'duplicate frozen feature key: {key}')
            values=(float(features['v4_contribution'][i]),float(features['a3_score'][i]))
            if any(not math.isfinite(x) for x in values):
                raise ValueError('nonfinite feature')
            index[key]=values
    attached=json.loads(json.dumps(manifest))
    ids={x['candidate_id'] for x in attached.get('baseline',[])+attached.get('rescue',[])}
    if set(candidate_map) != ids:
        raise ValueError('mapping must cover exactly the candidate IDs; no silent mis-join')
    if calibration is not None:
        _validate_no_gt(calibration)
        if calibration.get('status') != 'FROZEN_CAL_TRAIN_ONLY' or calibration.get('uses_holdout') is not False:
            raise ValueError('router calibration must be separately locked and training-only')
        weights=calibration.get('weights')
        if set(weights or {}) != {'v4_contribution','a3_score','intercept'}:
            raise ValueError('calibration weights missing')
        if not all(math.isfinite(float(x)) for x in weights.values()):
            raise ValueError('calibration must be finite')
    diag={}
    for item in attached['baseline']+attached['rescue']:
        cid=item['candidate_id']
        m=candidate_map[cid]
        key=(int(m['node_graph_id']),int(m['raw_query_index']))
        if key not in index:
            raise ValueError(f'missing frozen node {key} for {cid}')
        v4,a3=index[key]
        diag[cid]={'v4_contribution':v4,'a3_score':a3,'node_graph_id':key[0], 'raw_query_index':key[1]}
        if calibration:
            z=(float(weights['intercept'])+float(weights['v4_contribution'])*v4+
               float(weights['a3_score'])*a3)
            item['risk_hint']=float(1/(1+math.exp(-max(-40,min(40,z)))))
    attached.setdefault('metadata',{})['thu_frozen_features']={
        'source':'FSC-147 1301 pairs, exposed exploratory',
        'feature_sha256':sha(npz),'candidate_map_sha256':hashlib.sha256(json.dumps(candidate_map,sort_keys=True).encode()).hexdigest(),
        'calibration_used':calibration is not None,
        'calibration_sha256':hashlib.sha256(json.dumps(calibration,sort_keys=True).encode()).hexdigest() if calibration else None,
        'per_candidate':diag,
        'transfer_warning':'Do not apply these FSC frozen features to LVIS or new cohorts without exact-key reconstruction'}
    parse_input(attached)
    return attached
