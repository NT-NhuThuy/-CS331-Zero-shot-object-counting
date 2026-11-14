from pathlib import Path
import json
import pytest
from countlab.contracts import CandidateEvidence, decision_from_unary, validate_instance_output
from countlab.audit import sign_test, paired_audit
from countlab.provenance import freeze_manifest, sha256_file


def candidate(**kw):
    fields=dict(candidate_id="c1", image_id="im1", query_id="q", role="baseline", semantic="YES", whole_instance="YES", decision="ACCEPTED", box_xyxy=(0,0,1,1))
    fields.update(kw)
    return CandidateEvidence(**fields)


def test_accept_gate():
    candidate().validate()
    with pytest.raises(ValueError):candidate(semantic="UNKNOWN").validate()
    assert decision_from_unary("NO", "YES") == "UNKNOWN"
    assert decision_from_unary("NO", "NO") == "REJECTED"


def test_box():
    with pytest.raises(ValueError):candidate(box_xyxy=(3,2,1,5)).validate()


def test_interval_contract():
    o=dict(count=2,count_lower_bound=1,count_upper_bound=3,count_status="partial",accepted_instances=[],rejected_regions=[],unknown_regions=[{"id":"u"}],evidence_trace=[{"type":"test"}],query_abstained=False)
    assert not validate_instance_output(o)
    with pytest.raises(ValueError):validate_instance_output({**o,"count_status":"exact"})
    with pytest.raises(ValueError):validate_instance_output({**o,"count":4})


def test_exact_zero_warning():
    o=dict(count=0,count_lower_bound=0,count_upper_bound=0,count_status="exact",accepted_instances=[],rejected_regions=[],unknown_regions=[],evidence_trace=[{"type":"test"}],query_abstained=False)
    assert validate_instance_output(o)


def test_sign_test():
    assert sign_test(22,3) == pytest.approx(0.00015652179718017578)
    assert sign_test(0,0) == 1.0


def test_paired_audit_cluster():
    rows=[dict(id="1",source_image_id="A",gt=2,baseline=0,method=1,status="present"),
          dict(id="2",source_image_id="A",gt=0,baseline=2,method=0,status="absent"),
          dict(id="3",source_image_id="B",gt=1,baseline=1,method=2,status="present")]
    r=paired_audit(rows,bootstrap=25,seed=12)
    assert r["n_source_clusters"]==2 and r["wins"]==2 and r["losses"]==1
    assert r["method_mae"]==pytest.approx(2/3)


def test_no_duplicate_pair():
    row=dict(id="1",source_image_id="A",gt=1,baseline=2,method=1)
    with pytest.raises(ValueError):paired_audit([row,row],bootstrap=0)


def test_freeze_create_only(tmp_path):
    f=tmp_path/"x"; f.write_text("abc")
    m=tmp_path/"manifest.json"
    record=freeze_manifest(m,{"x":f})
    assert record["artifacts"]["x"]["sha256"]==sha256_file(f)
    with pytest.raises(FileExistsError):freeze_manifest(m,{"x":f})
