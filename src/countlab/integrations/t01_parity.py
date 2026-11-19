"""T01/G0: independent, GT-isolated count/native-box parity gate.

This reviewer tool deliberately DOES NOT run CountGD++ or generate a reference.
It rejects V14 AMP/postprocessed-count-only outputs as a matched FP32 reference.
A numeric pass is not a substitute for an actual team sign-off.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from zipfile import ZipFile

PROTOCOL_SHA = "121786020bead2e87d58e9dd6896d022ba2068d58feecfbbc4026113b57f497e"
R10_ROOT = "UNICOUNT_T01_R10_PSEUDO3_DIAGNOSTIC/"
_STRATA = {"present", "verified_absent"}
_FORBIDDEN_PREDICTION_KEYS = {"ground_truth", "gt_count", "gt", "label", "query_status", "presence_label", "verified_absent"}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(data: bytes):
    return json.loads(data.decode("utf-8"))


def read_jsonl(data: bytes) -> list[dict]:
    rows = [load_json(line) for line in data.splitlines() if line.strip()]
    if not all(isinstance(x, dict) for x in rows):
        raise ValueError("JSONL rows must be objects")
    return rows


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def _id(row):
    _require(isinstance(row.get("image_sha256"), str) and len(row["image_sha256"]) == 64,
             "image_sha256 missing/invalid")
    _require(isinstance(row.get("query"), str) and row["query"].strip(), "query missing")
    return (row["image_sha256"], row["query"])


def _no_gt(row):
    if isinstance(row, dict):
        for k, value in row.items():
            if str(k).lower() in _FORBIDDEN_PREDICTION_KEYS:
                raise ValueError("GT/label not allowed in inference/reference predictions: " + str(k))
            _no_gt(value)
    elif isinstance(row, list):
        for value in row:
            _no_gt(value)


def _unique_index(rows: list[dict]) -> dict:
    result = {}
    for row in rows:
        _no_gt(row)
        k = _id(row)
        _require(k not in result, "Duplicate prediction pair")
        result[k] = row
    return result


def load_r10(zip_path: Path):
    with ZipFile(zip_path) as z:
        _require(z.testzip() is None, "Candidate ZIP CRC failed")
        p = load_json(z.read(R10_ROOT + "protocol_manifest.json"))
        _require(p.get("protocol_sha256") == PROTOCOL_SHA,
                 "Candidate protocol hash does not match frozen R10")
        _require(p.get("status") == "CANDIDATE_FROZEN_P02_FP32_NOT_TEAM_APPROVED",
                 "Candidate provenance mismatch")
        roster = load_json(z.read(R10_ROOT + "roster_gt_free.json"))
        traces = read_jsonl(z.read(R10_ROOT + "per_pair_gpu_trace.jsonl"))
        candrows = read_jsonl(z.read(R10_ROOT + "candidates_diagnostic.jsonl"))
        _require(len(roster) == len(traces) == len(candrows) == 32, "R10 32 pair coverage missing")
        result=[]
        for i,(r,t,c) in enumerate(zip(roster,traces,candrows)):
            _require(r["index"] == t["index"] == c["index"] == i,"R10 ordering drift")
            _require(t.get("status") == "GPU_TWO_PASS_PSEUDO3_PASS" and t.get("detector_calls") == 2,
                     "R10 GPU two-pass incomplete")
            candidates=c.get("candidates", [])
            _require(len(candidates) == t.get("count"), "R10 count mismatch")
            unique=set()
            for item in candidates:
                idx=item.get("native_query_index")
                _require(isinstance(idx,int) and not isinstance(idx,bool) and 0 <= idx < 900,
                         "Missing or invalid R10 native query index")
                _require(idx not in unique,"R10 duplicate native index")
                unique.add(idx)
            _require(sha_bytes(z.read(R10_ROOT+t["raw_sidecar"])) == t["raw_sidecar_sha256"],
                     "R10 raw NPZ SHA mismatch")
            row={"image_sha256":r["image_sha256"],"query":r["query"],
                 "image_id":r["identity"].get("image_id"),"candidates":candidates}
            result.append(row)
        return {"protocol_sha256":p["protocol_sha256"],"rows":result,
                "archive_sha256":sha_bytes(zip_path.read_bytes()),"cohort":"P02_EXPOSED_DIAGNOSTIC"}


def _candidate_map(row: dict):
    cands=row.get("candidates")
    _require(isinstance(cands,list),"Candidate list missing")
    result={}
    for c in cands:
        _require(isinstance(c,dict),"Candidate record invalid")
        key=c.get("native_query_index")
        _require(isinstance(key,int) and not isinstance(key,bool) and 0 <= key < 900,
                 "Native query index required")
        _require(key not in result,"Duplicate native query index")
        _require(isinstance(c.get("box_xyxy"),list) and len(c["box_xyxy"])==4,
                 "Original-pixel box required")
        _require(all(isinstance(a,(int,float)) and math.isfinite(a) for a in c["box_xyxy"]),
                 "Invalid/nonfinite box")
        _require(isinstance(c.get("confidence"),(int,float)) and math.isfinite(c["confidence"]),
                 "Missing/nonfinite confidence")
        result[key]=c
    return result


def _read_strata(path: Path) -> dict:
    rows=read_jsonl(path.read_bytes())
    m={}
    for r in rows:
        k=_id(r)
        _require(k not in m, "Duplicate stratum pair")
        _require(r.get("status") in _STRATA,"Only present/verified_absent supported")
        _require(isinstance(r.get("verification_source"),str) and r["verification_source"].strip(),
                 "Verification provenance missing")
        m[k]=r["status"]
    return m


def compare(r10_zip: Path, reference_jsonl: Path, reference_manifest: Path,
            strata_jsonl: Path, approval_json: Path|None = None,
            box_atol: float = 1e-4, score_atol: float = 1e-6) -> dict:
    """Apply pre-registered absolute tolerances without retuning/IoU matching.

    Reference is from a distinct, externally produced, GT-free matched FP32 run.
    Reference creation, cohort approval and human attestation are OUTSIDE this tool.
    """
    _require(box_atol==1e-4 and score_atol==1e-6, "Tolerances frozen; no post-hoc adjustment")
    r10=load_r10(r10_zip)
    ref_manifest=load_json(reference_manifest.read_bytes())
    _require(ref_manifest.get("protocol_sha256")==r10["protocol_sha256"],
             "Reference protocol different from R10 FP32; cannot certify parity")
    _require(ref_manifest.get("source_type")=="INDEPENDENT_GPU_REFERENCE_RAW_OR_NATIVE",
             "Independent GPU reference provenance not supplied")
    _require(ref_manifest.get("run_id") not in (None,"", "R10") and
             isinstance(ref_manifest.get("run_id"),str),"Independent run_id missing")
    _require(ref_manifest.get("source_commit")==
             "2ecbc4331d6730d017f52a4a97bbcbae7d5d714a",
             "Reference source revision mismatch")
    _require(ref_manifest.get("reference_predictions_sha256") == sha_bytes(reference_jsonl.read_bytes()),
             "Reference predictions SHA mismatch")
    ref=_unique_index(read_jsonl(reference_jsonl.read_bytes()))
    r10rows=_unique_index(r10["rows"])
    labels=_read_strata(strata_jsonl)
    _require(len(r10rows)==len(ref)==len(labels)==32 and set(r10rows)==set(ref)==set(labels),
             "Reference/strata coverage must match all 32 image-SHA+query pairs")
    records=[]
    for k,row in r10rows.items():
        refrow=ref[k]
        a=_candidate_map(row);b=_candidate_map(refrow)
        miss=sorted(set(a)-set(b));extra=sorted(set(b)-set(a))
        max_box=max_score=0.0
        box_ok=score_ok=True
        for idx in sorted(set(a)&set(b)):
            aa=a[idx];bb=b[idx]
            diffs=[abs(float(x)-float(y)) for x,y in zip(aa["box_xyxy"],bb["box_xyxy"])]
            max_box=max(max_box,*diffs)
            max_score=max(max_score,abs(aa["confidence"]-bb["confidence"]))
            if any(x>box_atol for x in diffs):box_ok=False
            if abs(aa["confidence"]-bb["confidence"])>score_atol:score_ok=False
        passed=not miss and not extra and box_ok and score_ok
        records.append({"image_sha256":k[0],"query":k[1],"stratum":labels[k],
                        "r10_count":len(a),"reference_count":len(b),"passed":passed,
                        "missing_native_indices":miss,"extra_native_indices":extra,
                        "max_box_abs_error_px":max_box,"max_score_abs_error":max_score})
    counts={s:{"total":sum(x["stratum"]==s for x in records),
               "passed":sum(x["stratum"]==s and x["passed"] for x in records)} for s in sorted(_STRATA)}
    numeric_pass=all(x["passed"] for x in records) and all(x["total"]>0 for x in counts.values())
    approval=None
    if approval_json is not None:
        approval=load_json(approval_json.read_bytes())
    approved=bool(approval and all(approval.get(k) is True for k in
        ("protocol_approved","roster_approved","reference_approved","tolerances_preapproved"))
        and isinstance(approval.get("approver"),str) and approval["approver"].strip()
        and isinstance(approval.get("decision_timestamp_utc"),str) and approval["decision_timestamp_utc"].strip())
    status=("G0_TECHNICAL_PARITY_PASS_SIGNOFF_RECORDED" if numeric_pass and approved else
            "HOLD_PENDING_G0_APPROVAL" if numeric_pass else
            "HOLD_PARITY_MISMATCH")
    return {"schema":"t01_matched_reference_parity_v1","status":status,
            "technical_numeric_parity": "PASS" if numeric_pass else "FAIL",
            "team_approval_recorded":approved,"g0":"PASS_FOR_T01_TECHNICAL_SCOPE" if numeric_pass and approved else "HOLD",
            "counts":counts,"box_atol_px":box_atol,"score_atol":score_atol,
            "candidate_archive_sha256":r10["archive_sha256"],
            "reference_predictions_sha256":sha_bytes(reference_jsonl.read_bytes()),
            "details":records,
            "note":"Approval metadata is a record supplied by humans, not a cryptographic proof of independence."}


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--r10-zip",type=Path,required=True)
    ap.add_argument("--reference-jsonl",type=Path)
    ap.add_argument("--reference-manifest",type=Path)
    ap.add_argument("--strata-jsonl",type=Path)
    ap.add_argument("--team-approval",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args(argv)
    try:
        if all([args.reference_jsonl,args.reference_manifest,args.strata_jsonl]):
            result=compare(args.r10_zip,args.reference_jsonl,args.reference_manifest,
                           args.strata_jsonl,args.team_approval)
        else:
            r10=load_r10(args.r10_zip)
            result={"schema":"t01_matched_reference_parity_v1",
                    "status":"HOLD_REFERENCE_LABELS_AND_SIGNOFF_REQUIRED",
                    "g0":"HOLD","r10_pairs_checked":len(r10["rows"]),
                    "r10_archive_sha256":r10["archive_sha256"],
                    "numeric_parity":"NOT_MEASURED",
                    "note":"No matched independent FP32 reference + verified separate strata + approval was supplied."}
    except (ValueError,KeyError,FileNotFoundError,RuntimeError,AssertionError) as e:
        result={"schema":"t01_matched_reference_parity_v1","status":"HOLD_INVALID_INPUT",
                "g0":"HOLD","error":str(e)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f"Refuse to overwrite G0 result: {args.out}")
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("details",)},indent=2))
    return 0 if result.get("status")=="G0_TECHNICAL_PARITY_PASS_SIGNOFF_RECORDED" else 2


if __name__=="__main__":
    raise SystemExit(main())
