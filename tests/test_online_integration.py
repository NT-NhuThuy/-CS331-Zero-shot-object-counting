"""Contract tests for real-adapter glue. Fake weights mean NOT GPU-parity tests."""
from pathlib import Path
import json
import pytest
from PIL import Image
from countlab.integrations.countgd_native import normalized_box_to_xyxy, proposals_from_prediction, PROTOCOL_ID
from countlab.integrations.recovery import tile_bounds, recover_from_native
from countlab.integrations.qwen_unary import parse_verdict, verification_prompts, QwenRegionUnary
from countlab.integrations.online import execute_online
from countlab.method.schema import Proposal


def _model_result(*, boxes=((.5,.5,.5,.5),), scores=(.9,), text=(True,), size=(100,100)):
    return {"boxes_cxcywh_normalized":list(boxes), "native_scores":list(scores),
            "native_text_pass":list(text), "original_size":size}


def test_normalized_geometry_and_offset():
    assert normalized_box_to_xyxy([.5,.5,.5,.5],100,80,offset=(50,10)) == [75,30,125,70]
    assert normalized_box_to_xyxy([2,2,.1,.1],100,100) is None
    assert len(proposals_from_prediction(_model_result(), threshold=.23, role='baseline',view_id='full')) == 1
    assert proposals_from_prediction(_model_result(text=(False,)), threshold=.23, role='baseline',view_id='full') == []
    with pytest.raises(ValueError, match='unaligned'):
        proposals_from_prediction(_model_result(scores=()),threshold=.23,role='baseline',view_id='full')


def test_tile_bounds_image_coordinates():
    bounds=tile_bounds(100,80,grids=(2,),overlap=0.,max_tiles=4)
    assert bounds[0] == ('g2_r0_c0',(0,0,50,40))
    assert bounds[-1] == ('g2_r1_c1',(50,40,100,80))
    with pytest.raises(ValueError,match='budget'):
        tile_bounds(100,80,grids=(8,),max_tiles=4)


class FakeProvider:
    provenance={"kind":"fake"}
    def __init__(self, *, baseline=1, rescue=1):
        self.base_count, self.rescue_count=baseline,rescue
        self.calls=[]
    def infer(self,path,query,*,role,view_id,offset=(0,0),image_id=''):
        from PIL import Image
        size=Image.open(path).size
        self.calls.append((role,view_id,size,offset))
        n=self.base_count if role=='baseline' else self.rescue_count
        return proposals_from_prediction(_model_result(boxes=[(.5,.5,.4,.4)]*n,
                                                       scores=[.9]*n,text=[True]*n,size=size),
                                         threshold=.23,role=role,view_id=view_id,offset=offset,image_id=image_id)


class FakeQwen:
    model_id='qwen-test';revision='test-sha';calls=0;fallbacks=0
    def verdict(self,image,query,proposal):
        self.calls+=2
        return ('NO','NO') if proposal.role=='baseline' else ('YES','YES')


def _config(*, budget=32, trigger=1):
    return {"protocol_id":PROTOCOL_ID,"recovery":{"trigger_baseline_le":trigger,"grids":[2],"overlap":0.,"max_tiles":4},
            "verification_budget":budget,"method_policy":{"baseline_rescue_match_iou":.5,
                "identity_min_same_iou":.1,"identity_overlap_ambiguity_iou":.3},"abstain_on_conflict":True}


def test_execute_online_generates_real_tiles_and_preserves_frozen_baseline(tmp_path):
    path=tmp_path/'image.png';Image.new('RGB',(100,80)).save(path)
    backend=FakeProvider(baseline=1,rescue=1)
    manifest,result=execute_online(image_path=path,query='chair',image_id='test',query_id='1',
                                    config=_config(),backend=backend,verifier=FakeQwen())
    assert len(backend.calls)==5 and len(manifest['rescue'])==4
    assert backend.calls[-1][3] == (50,40)
    assert manifest['rescue'][-1]['box_xyxy']==[65,52,85,68]
    assert manifest['baseline'][0]['evidence']=={'semantic':'NO','whole_instance':'NO'}
    assert result['evidence_trace'][-1]['verifier_calls']==10
    assert result['evaluation_fallback_count']>=0
    assert result['method_options']['enable_identity']
    assert 'gt' not in json.dumps(manifest).lower()


def test_missing_unary_no_false_rejection(tmp_path):
    path=tmp_path/'image.png';Image.new('RGB',(100,80)).save(path)
    backend=FakeProvider(baseline=1,rescue=1)
    manifest,result=execute_online(image_path=path,query='chair',image_id='test',query_id='2',
                                    config=_config(budget=0, trigger=-1),backend=backend)
    assert result['count']==1
    assert manifest['baseline'][0]['evidence']=={'semantic':'UNKNOWN','whole_instance':'UNKNOWN'}
    assert not manifest['rescue']


def test_qwen_strict_parser_and_injected_generator(tmp_path):
    img=tmp_path/'x.png';Image.new('RGB',(100,100)).save(img)
    assert parse_verdict('No')=='NO'
    assert parse_verdict('YES because it resembles an object')=='UNKNOWN'
    assert parse_verdict('YES\nNO')=='UNKNOWN'
    assert 'whole physical' in verification_prompts('chair')[1].lower()
    q=QwenRegionUnary(revision='abc123',generator=lambda full,crop,prompt:'YES')
    p=Proposal.parse({'candidate_id':'r1','box_xyxy':[1,1,20,20]},role='rescue')
    assert q.verdict(img,'chair',p)==('YES','YES') and q.calls==2
    with pytest.raises(ValueError, match='revision'):
        QwenRegionUnary(revision='main',generator=lambda *_:'YES')


def test_legacy_countgd_backend_is_opt_in_and_requires_config(tmp_path):
    from countlab.integrations.countgd_native import NativeCountGD
    fake=type('Backend',(),{'spec':{'source':'fake'},'predict':lambda self,image,q:_model_result()})()
    model=NativeCountGD({'threshold':.23},backend=fake)
    assert model.protocol_id==PROTOCOL_ID
    assert len(model.infer(tmp_path/'anything.png','chair'))==1


def test_pairwise_online_qwen_identity_is_explicit_not_iou_oracle(tmp_path):
    from countlab.integrations.qwen_unary import verify_identity_pairs
    image=tmp_path/'x.png';Image.new('RGB',(100,100)).save(image)
    a=Proposal.parse({'candidate_id':'r1','view_id':'tile1','box_xyxy':[0,0,20,20]},'rescue')
    b=Proposal.parse({'candidate_id':'r2','view_id':'tile2','box_xyxy':[2,2,22,22]},'rescue')
    model=QwenRegionUnary(revision='abc123',generator=lambda full,crop,prompt:'YES')
    edge=verify_identity_pairs(image,[a,b],{'r1':('YES','YES'),'r2':('YES','YES')},model,max_pairs=1)
    assert edge==[{'a':'r1','b':'r2','relation':'SAME'}] and model.calls==1
    assert verify_identity_pairs(image,[a,b],{'r1':('UNKNOWN','YES'),'r2':('YES','YES')},model,max_pairs=1)==[]
    assert verify_identity_pairs(image,[a,b],{'r1':('YES','YES'),'r2':('YES','YES')},model,max_pairs=0)==[]


def test_online_routing_uses_attached_thu_frozen_features_before_verification(tmp_path):
    import numpy as np
    path=tmp_path/'image.png';Image.new('RGB',(100,80)).save(path)
    npz=tmp_path/'frozen.npz'
    np.savez(npz, node_graph_id=np.array([0,0]), raw_query_index=np.array([0,1]),
             v4_contribution=np.array([1.,1.]), a3_score=np.array([.01,.99]))
    mapping={f'baseline:full:{i}':{'node_graph_id':0,'raw_query_index':i} for i in range(2)}
    mapper=tmp_path/'mapping.json';mapper.write_text(json.dumps(mapping))
    calibration={'status':'FROZEN_CAL_TRAIN_ONLY','uses_holdout':False,
                 'weights':{'v4_contribution':0.,'a3_score':8.,'intercept':-4.}}
    cal=tmp_path/'cal.json';cal.write_text(json.dumps(calibration))
    cfg=_config(budget=1,trigger=-1)
    cfg['thu_frozen_features']={'npz_path':str(npz),'candidate_map_path':str(mapper),
                                'calibration_path':str(cal)}
    manifest,result=execute_online(image_path=path, query='chair', image_id='im',query_id='q',
                                    config=cfg,backend=FakeProvider(baseline=2,rescue=0),verifier=FakeQwen())
    assert manifest['baseline'][1]['risk_hint']>manifest['baseline'][0]['risk_hint']
    assert manifest['baseline'][1]['evidence']['semantic']=='NO'
    assert manifest['baseline'][0]['evidence']['semantic']=='UNKNOWN'
    assert result['evaluation_fallback_count']==1
