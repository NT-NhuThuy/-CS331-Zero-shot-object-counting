from copy import deepcopy
from countlab.thu_t12_contract import audit_roster, validate_scoring_contract

H = "a" * 64
J = "b" * 64

def row(**updates):
    result = {"dataset":"FSC", "image_id":"img1", "query_id":"q-neutral-1",
              "query_text":"strawberry", "source_image_id":"src1", "source_group_id":"group1",
              "image_sha256":H, "split":"EXPOSED_REFERENCE", "exposure_status":"EXPOSED",
              "lineage_status":"HISTORICAL_EXPOSED", "eligibility":"PRESENCE_ONLY", "provenance_evidence_sha256":""}
    result.update(updates)
    return result

def contract():
    return {"schema":"thu_t12_scoring_contract_v1",
            "arms":{a:{"implementation_id":f"impl.{a}","method_commit":"sha-v1", "unary_budget":4,"pair_budget":0,"recovery_budget":0,"budget_unit":"observed_calls"} for a in ("A0","A1","A3","A4","A5")},
            "frozen_pool_manifest_sha256":H,"paired_roster_sha256":H,"policy_sha256":H,
            "gt_access":"OFFLINE_EVALUATOR_ONLY", "missing_evidence":"UNKNOWN_NOT_NO",
            "cost_accounting":"MEASURED_NOT_REPLAY_ESTIMATE", "abstention_scoring":"REPORT_RATE_AND_FORCED_FALLBACK",
            "metric_definitions":{"present":"MAE_PRESENT_ONLY","absent":"AZA_EXACT_ZERO","false_zero":"FZR_PLUS_ON_PRESENT","overall":"MAE_WITH_FALLBACK"},
            "selection_split":"FIXED_PRE_REGISTERED", "source_disjoint_status":"VERIFIED_EXTERNALLY"}

def test_exposed_roster_pass_but_g0_hold():
    result=audit_roster([row()])
    assert result["metadata_status"]=="PASS" and result["g0_data_decision"]=="HOLD"

def test_source_group_leakage_detected():
    a=row(split="DEV",exposure_status="UNEXPOSED",lineage_status="VERIFIED",provenance_evidence_sha256=H)
    b=row(image_id="img2",query_id="q2",source_image_id="src2",source_group_id="group1",image_sha256=J)
    result=audit_roster([a,b])
    assert result["metadata_status"]=="FAIL"
    assert any("source group leakage" in v for v in result["issues"])

def test_byte_hash_group_conflict_detected():
    result=audit_roster([row(),row(image_id="img2",query_id="q2",source_image_id="src2",source_group_id="group2")])
    assert any("identical image" in v for v in result["issues"])

def test_clean_gt_value_blocked():
    x=row(split="DEV",exposure_status="UNEXPOSED",lineage_status="VERIFIED",provenance_evidence_sha256=H,count_target="12")
    assert audit_roster([x])["metadata_status"]=="FAIL"

def test_final_sealed_requires_access_lock():
    x=row(split="FINAL_SEALED",exposure_status="UNEXPOSED",lineage_status="VERIFIED",provenance_evidence_sha256=H)
    assert any("SEALED" in v for v in audit_roster([x])["issues"])

def test_duplicate_query_blocked():
    assert any("duplicate" in v for v in audit_roster([row(),row()])["issues"])

def test_complete_scoring_contract_validates_metadata_only():
    assert validate_scoring_contract(contract())["status"]=="PASS"

def test_incomplete_provenance_stays_hold():
    c=contract();c["source_disjoint_status"]="HOLD"
    assert validate_scoring_contract(c)["status"]=="HOLD"

def test_budget_negative_or_missing_blocked():
    c=contract();c["arms"]["A5"]["pair_budget"]=-1
    assert validate_scoring_contract(c)["status"]=="HOLD"

def test_replay_cost_not_observed_cost():
    c=contract();c["cost_accounting"]="MASKED_SLOT_ESTIMATE"
    assert validate_scoring_contract(c)["status"]=="HOLD"

def test_arm_mismatch_blocked():
    c=contract();c["arms"]["B5"]=c["arms"].pop("A5")
    assert validate_scoring_contract(c)["status"]=="HOLD"
