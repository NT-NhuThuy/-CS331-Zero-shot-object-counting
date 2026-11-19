"""CPU-only T01 parity gate tests; no GPU, no GT in inference."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from countlab.integrations.t01_parity import (
    sha_bytes,read_jsonl,_id,_unique_index,_candidate_map,_read_strata,
    load_r10,compare,PROTOCOL_SHA)


def fixture(tmp_path, r10):
    rows=load_r10(r10)['rows']
    ref=tmp_path/'reference.jsonl'
    ref.write_text(''.join(json.dumps({k: row[k] for k in ('image_sha256','query','candidates')})+'\n'
                   for row in rows),encoding='utf-8')
    m=tmp_path/'reference_manifest.json'
    m.write_text(json.dumps({'protocol_sha256':PROTOCOL_SHA,
        'source_type':'INDEPENDENT_GPU_REFERENCE_RAW_OR_NATIVE',
        'run_id':'separate-matched-fp32-run','source_commit':'2ecbc4331d6730d017f52a4a97bbcbae7d5d714a',
        'reference_predictions_sha256':sha_bytes(ref.read_bytes())}))
    strata=tmp_path/'strata.jsonl'
    strata.write_text(''.join(json.dumps({'image_sha256':x['image_sha256'],'query':x['query'],
        'status':'present' if i<15 else 'verified_absent',
        'verification_source':'unit-test-only'})+'\n' for i,x in enumerate(rows)))
    appro=tmp_path/'approval.json'
    appro.write_text(json.dumps({'protocol_approved':True,'roster_approved':True,
        'reference_approved':True,'tolerances_preapproved':True,
        'approver':'unit-test-simulated','decision_timestamp_utc':'2026-11-19T00:00:00Z'}))
    return ref,m,strata,appro


@pytest.fixture(scope='session')
def r10():
    p=ROOT/'docs/method/parity_logs/UNICOUNT_T01_R10_GPU_EVIDENCE.zip'
    assert p.is_file()
    return p


def test_raw_real_gpu_32(r10):
    d=load_r10(r10)
    assert len(d['rows'])==32
    assert sum(len(r['candidates']) for r in d['rows'])==38
    assert sum(not r['candidates'] for r in d['rows'])==24


def test_hash_detects_zip_tamper(r10,tmp_path):
    import zipfile
    fake=tmp_path/'bad.zip'
    with zipfile.ZipFile(r10) as src, zipfile.ZipFile(fake,'w') as target:
        for info in src.infolist():
            data=src.read(info.filename)
            if info.filename.endswith('raw/000.npz'): data+=b'tamper'
            target.writestr(info.filename,data)
    with pytest.raises(ValueError, match='SHA mismatch'):
        load_r10(fake)


def test_exact_matched_pass_is_simulation_not_real_signoff(tmp_path,r10):
    ref,m,s,a=fixture(tmp_path,r10)
    result=compare(r10,ref,m,s,a)
    assert result['status']=='G0_TECHNICAL_PARITY_PASS_SIGNOFF_RECORDED'
    assert result['counts']['present']['passed']==15
    assert result['counts']['verified_absent']['passed']==17


def test_missing_approval_holds(tmp_path,r10):
    ref,m,s,a=fixture(tmp_path,r10)
    result=compare(r10,ref,m,s)
    assert result['status']=='HOLD_PENDING_G0_APPROVAL'
    assert not result['team_approval_recorded']


def test_reject_diff_protocol(tmp_path,r10):
    ref,m,s,a=fixture(tmp_path,r10)
    d=json.loads(m.read_text());d['protocol_sha256']='0'*64;m.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='protocol different'):
        compare(r10,ref,m,s,a)


def test_reject_no_independent_provenance(tmp_path,r10):
    ref,m,s,a=fixture(tmp_path,r10)
    d=json.loads(m.read_text());d['source_type']='V14_POSTPROCESSED_AMPLITUDE';m.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Independent GPU reference'):
        compare(r10,ref,m,s,a)


def test_reject_mismatch_score(tmp_path,r10):
    ref,m,s,a=fixture(tmp_path,r10)
    rr=read_jsonl(ref.read_bytes());row=next(r for r in rr if r['candidates'])
    row['candidates'][0]['confidence']+=.01
    ref.write_text(''.join(json.dumps(x)+'\n' for x in rr))
    md=json.loads(m.read_text());md['reference_predictions_sha256']=sha_bytes(ref.read_bytes());m.write_text(json.dumps(md))
    result=compare(r10,ref,m,s,a)
    assert result['status']=='HOLD_PARITY_MISMATCH'


def test_reject_box_mismatch(tmp_path,r10):
    ref,m,s,a=fixture(tmp_path,r10)
    rr=read_jsonl(ref.read_bytes());row=next(r for r in rr if r['candidates'])
    row['candidates'][0]['box_xyxy'][0]+=.2
    ref.write_text(''.join(json.dumps(x)+'\n' for x in rr))
    md=json.loads(m.read_text());md['reference_predictions_sha256']=sha_bytes(ref.read_bytes());m.write_text(json.dumps(md))
    assert compare(r10,ref,m,s,a)['status']=='HOLD_PARITY_MISMATCH'


def test_reject_missing_native(tmp_path,r10):
    ref,m,s,a=fixture(tmp_path,r10)
    rr=read_jsonl(ref.read_bytes());row=next(r for r in rr if r['candidates'])
    row['candidates'][0].pop('native_query_index')
    ref.write_text(''.join(json.dumps(x)+'\n' for x in rr))
    md=json.loads(m.read_text());md['reference_predictions_sha256']=sha_bytes(ref.read_bytes());m.write_text(json.dumps(md))
    with pytest.raises(ValueError, match='Native query index required'):
        compare(r10,ref,m,s,a)


def test_reject_missing_pair(tmp_path,r10):
    ref,m,s,a=fixture(tmp_path,r10)
    ref.write_text('\n'.join(ref.read_text().splitlines()[:-1])+'\n')
    md=json.loads(m.read_text());md['reference_predictions_sha256']=sha_bytes(ref.read_bytes());m.write_text(json.dumps(md))
    with pytest.raises(ValueError,match='coverage'):
        compare(r10,ref,m,s,a)


def test_reject_no_absent_stratum(tmp_path,r10):
    ref,m,s,a=fixture(tmp_path,r10)
    rr=read_jsonl(s.read_bytes());
    for row in rr:row['status']='present'
    s.write_text(''.join(json.dumps(r)+'\n' for r in rr))
    assert compare(r10,ref,m,s,a)['status']=='HOLD_PARITY_MISMATCH'


def test_reject_ref_with_gt_leak(tmp_path,r10):
    ref,m,s,a=fixture(tmp_path,r10)
    rr=read_jsonl(ref.read_bytes());rr[0]['gt_count']=4
    ref.write_text(''.join(json.dumps(x)+'\n' for x in rr))
    md=json.loads(m.read_text());md['reference_predictions_sha256']=sha_bytes(ref.read_bytes());m.write_text(json.dumps(md))
    with pytest.raises(ValueError,match='GT/label'):
        compare(r10,ref,m,s,a)


def test_reject_changed_native_idx(tmp_path,r10):
    ref,m,s,a=fixture(tmp_path,r10)
    rr=read_jsonl(ref.read_bytes());row=next(r for r in rr if r['candidates'])
    row['candidates'][0]['native_query_index']=899
    ref.write_text(''.join(json.dumps(x)+'\n' for x in rr))
    md=json.loads(m.read_text());md['reference_predictions_sha256']=sha_bytes(ref.read_bytes());m.write_text(json.dumps(md))
    assert compare(r10,ref,m,s,a)['status']=='HOLD_PARITY_MISMATCH'


def test_reject_ref_hash(tmp_path,r10):
    ref,m,s,a=fixture(tmp_path,r10)
    ref.write_text(ref.read_text()+'\n')
    with pytest.raises(ValueError,match='SHA mismatch'):
        compare(r10,ref,m,s,a)


def test_reject_missing_strata_provenance(tmp_path,r10):
    ref,m,s,a=fixture(tmp_path,r10)
    rr=read_jsonl(s.read_bytes());rr[0]['verification_source']='';
    s.write_text(''.join(json.dumps(x)+'\n' for x in rr))
    with pytest.raises(ValueError,match='provenance'):
        compare(r10,ref,m,s,a)


def test_reject_changed_tolerances(tmp_path,r10):
    ref,m,s,a=fixture(tmp_path,r10)
    with pytest.raises(ValueError,match='Tolerances frozen'):
        compare(r10,ref,m,s,a,box_atol=0.01)


def test_reject_nested_gt_leak(tmp_path,r10):
    ref,m,s,a=fixture(tmp_path,r10)
    rr=read_jsonl(ref.read_bytes());row=next(r for r in rr if r['candidates'])
    row['candidates'][0]['gt_count']=1
    ref.write_text(''.join(json.dumps(x)+'\n' for x in rr))
    md=json.loads(m.read_text());md['reference_predictions_sha256']=sha_bytes(ref.read_bytes());m.write_text(json.dumps(md))
    with pytest.raises(ValueError,match='GT/label'):
        compare(r10,ref,m,s,a)
