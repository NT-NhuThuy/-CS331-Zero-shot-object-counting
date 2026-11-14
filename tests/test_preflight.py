from countlab.integrations.preflight import inspect
from countlab.integrations.countgd_native import PROTOCOL_ID


def test_missing_gpu_resources_are_reported_without_running_models():
    r=inspect({'protocol_id':PROTOCOL_ID,'native_countgd':{'repo':'/REPLACE_WITH_ROOT','checkpoint':'/REPLACE_WITH_CKPT','bert_dir':'/REPLACE_WITH_BERT'},
               'verification_budget':1,'qwen':{'revision':'REPLACE_WITH_EXACT_COMMIT_SHA'}})
    assert r['status']=='BLOCKED'
    assert any('checkpoint' in x for x in r['blockers'])
    assert any('revision' in x for x in r['blockers'])
    assert 'GPU_SMOKE' not in r['status']
