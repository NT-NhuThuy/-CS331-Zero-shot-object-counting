import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_exploratory_v14_31_of_32_not_parity():
    p=ROOT/'docs/method/parity_logs/v14_reproduction_diagnostic.json'
    d=json.loads(p.read_text())
    assert d['status']=='DIAGNOSTIC_ONLY_NOT_G0_PARITY'
    assert d['matched_image_sha_and_query']==32
    assert d['count_equal']==31
    assert d['by_stratum']=={'present':{'pairs':15,'counts_equal':14},
                             'verified_absent':{'pairs':17,'counts_equal':17}}
    assert d['mismatches'][0]['image_id']==113
    assert d['mismatches'][0]['r10_count']==18
    assert d['mismatches'][0]['v14_count']==19
    assert d['g0']=='HOLD'
