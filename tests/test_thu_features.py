import numpy as np
import pytest
from countlab.integrations.thu_features import attach_features


def fixture():
    manifest={'schema':'countlab_candidate_pool_v1','image_id':'im','query_id':'q','query_text':'chair',
              'baseline_count':1,'baseline':[{'candidate_id':'c1','box_xyxy':[0,0,10,10]}], 'rescue':[],'identity_edges':[]}
    mapping={'c1':{'node_graph_id':9,'raw_query_index':3}}
    return manifest,mapping


def test_real_npz_exact_join_without_silent_count_correction(tmp_path):
    path=tmp_path/'features.npz'
    np.savez(path,node_graph_id=np.array([9]),raw_query_index=np.array([3]),
             v4_contribution=np.array([1.02]),a3_score=np.array([.7]))
    manifest,mapping=fixture()
    joined=attach_features(manifest,path,mapping)
    assert 'risk_hint' not in joined['baseline'][0]
    assert joined['metadata']['thu_frozen_features']['per_candidate']['c1']['a3_score'] == .7
    cal={'status':'FROZEN_CAL_TRAIN_ONLY','uses_holdout':False,
         'weights':{'v4_contribution':1.,'a3_score':-1.,'intercept':0.}}
    joined=attach_features(manifest,path,mapping,cal)
    assert 0<joined['baseline'][0]['risk_hint']<1
    assert 'risk_hint' not in manifest['baseline'][0]
    with pytest.raises(ValueError,match='mapping must cover'):
        attach_features(manifest,path,{})
    with pytest.raises(ValueError,match='training-only'):
        attach_features(manifest,path,mapping,{**cal,'uses_holdout':True})
