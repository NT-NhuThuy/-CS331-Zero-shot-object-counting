"""Evaluate frozen same-query Alpha-CLIP region-control margins."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"target_aware_counting/output/fqie_h2_alpha_lvis_full_20261104_v1"
RUN=ROOT/"target_aware_counting/output/fqie_h2_alpha_region_controls_20261104_v1"


def auc(pos,neg):
    return (sum(a>b for a in pos for b in neg)+.5*sum(a==b for a in pos for b in neg))/(len(pos)*len(neg))


def operating(rows,field,retention):
    pos=sorted((r[field] for r in rows if r["label"]=="TARGET_IOU50"),reverse=True)
    threshold=pos[math.ceil(retention*len(pos))-1]
    neg=[r for r in rows if r["label"]=="VERIFIED_ABSENT" and r[field]>=threshold]
    return {"retention":retention,"threshold":threshold,"absent_kept":len(neg),
            "absent_total":sum(r["label"]=="VERIFIED_ABSENT" for r in rows)}


def main():
    freeze=json.loads((RUN/"freeze_scores.json").read_text())
    labels={x["candidate_id"]:x["eval_label"] for x in json.loads((BASE/"eval_labels.json").read_text())["items"]}
    base_freeze=json.loads((BASE/"freeze_scores.json").read_text())
    rows=[]
    for cid,label in labels.items():
        name=cid.replace(":","_")+".json"; rel="scores/"+name
        cp=RUN/rel; bp=BASE/rel
        if hashlib.sha256(cp.read_bytes()).hexdigest()!=freeze["score_file_hashes"][rel]: raise RuntimeError("control changed")
        if hashlib.sha256(bp.read_bytes()).hexdigest()!=base_freeze["score_file_hashes"][rel]: raise RuntimeError("base changed")
        c=json.loads(cp.read_text()); b=json.loads(bp.read_text()); correct=b["views"]["full"]["target_cosine"]
        allv=c["controls"]["all_content"]["target_cosine"]
        comp=c["controls"]["complement_rectangle"]["target_cosine"]
        shift=c["controls"]["far_same_size_rectangle"]["target_cosine"]
        rows.append({"candidate_id":cid,"label":label,"raw":correct,"minus_all":correct-allv,
                     "minus_complement":correct-comp,"minus_shifted":correct-shift,
                     "minus_max_control":correct-max(allv,comp,shift),
                     "shift_iou":c["shifted_vs_correct_iou"]})
    result={"status":"DEVELOPMENT_CONTROL_ABLATION","denominators":{
        k:sum(r["label"]==k for r in rows) for k in ("TARGET_IOU50","VERIFIED_ABSENT","UNKNOWN")},
        "metrics":{},"mask_control_diagnostic":{}}
    pos=[r for r in rows if r["label"]=="TARGET_IOU50"];neg=[r for r in rows if r["label"]=="VERIFIED_ABSENT"]
    for field in ("raw","minus_all","minus_complement","minus_shifted","minus_max_control"):
        result["metrics"][field]={"auc":auc([r[field] for r in pos],[r[field] for r in neg]),
            "target_mean":sum(r[field] for r in pos)/len(pos),"absent_mean":sum(r[field] for r in neg)/len(neg),
            "operating_points":[operating(rows,field,x) for x in (1.0,.95,.9)]}
    result["mask_control_diagnostic"]={
        "target_correct_gt_shifted_fraction":sum(r["minus_shifted"]>0 for r in pos)/len(pos),
        "absent_correct_gt_shifted_fraction":sum(r["minus_shifted"]>0 for r in neg)/len(neg),
        "shifted_boxes_nonoverlap":sum(r["shift_iou"]==0 for r in rows),"all_candidates":len(rows)}
    result["limitations"]=["Only 13 absent proposals from two source images are available in this cohort.",
        "Controls are model-selection diagnostics on development, not confirmation.",
        "A far mask can cover another real instance; no semantic label is assigned to its content."]
    payload=json.dumps(result,indent=2)+"\n";out=RUN/"evaluation.json"
    if out.exists() and out.read_text()!=payload:raise RuntimeError("evaluation differs")
    out.write_text(payload)
    print(json.dumps(result["metrics"]));print(json.dumps(result["mask_control_diagnostic"]))


if __name__=="__main__":main()
