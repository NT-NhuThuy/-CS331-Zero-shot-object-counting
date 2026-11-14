"""Transfer LVIS control-margin thresholds to absent111 top-5 candidates."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"target_aware_counting/output/fqie_h2_alpha_absent111_controls_20261104_v1"
BASE=ROOT/"target_aware_counting/output/fqie_h2_alpha_absent111_top5_20261104_v1"
CAL=ROOT/"target_aware_counting/output/fqie_h2_alpha_region_controls_20261104_v1/evaluation.json"


def main():
    freeze=json.loads((RUN/"freeze_scores.json").read_text());basefreeze=json.loads((BASE/"freeze_scores.json").read_text())
    manifest=json.loads((RUN/"manifest.json").read_text());cal=json.loads(CAL.read_text());rows=[]
    for item in manifest["items"]:
        rel="scores/"+item["candidate_id"].replace(":","_")+".json";cp=RUN/rel;bp=BASE/rel
        if hashlib.sha256(cp.read_bytes()).hexdigest()!=freeze["score_file_hashes"][rel]:raise RuntimeError("control changed")
        if hashlib.sha256(bp.read_bytes()).hexdigest()!=basefreeze["score_file_hashes"][rel]:raise RuntimeError("base changed")
        c=json.loads(cp.read_text());b=json.loads(bp.read_text());correct=b["alpha_full"]
        allv=c["controls"]["all_content"]["target_cosine"];comp=c["controls"]["complement_rectangle"]["target_cosine"]
        shift=c["controls"]["far_same_size_rectangle"]["target_cosine"]
        rows.append({"item_id":item["item_id"],"native_rank":item["native_rank"],"raw":correct,
            "minus_all":correct-allv,"minus_complement":correct-comp,"minus_shifted":correct-shift,
            "minus_max_control":correct-max(allv,comp,shift)})
    result={"status":"DEVELOPMENT_TRANSFER_CONTROL_ABLATION","candidates":len(rows),"queries":111,"metrics":{}}
    for field,metric in cal["metrics"].items():
        points=[]
        for op in metric["operating_points"]:
            kept=[r for r in rows if r[field]>=op["threshold"]]
            points.append({"LVIS_target_retention":op["retention"],"threshold":op["threshold"],
                "absent_candidates_kept":len(kept),"absent_queries_active":len({r["item_id"] for r in kept}),
                "absent_top1_active":sum(r["native_rank"]==1 for r in kept)})
        result["metrics"][field]=points
    result["limitations"]=["Only top-five native candidates are scored.",
        "Margin type and threshold were selected on LVIS development; this remains development transfer.",
        "Region controls add three visual forwards per proposal."]
    payload=json.dumps(result,indent=2)+"\n";out=RUN/"evaluation.json"
    if out.exists() and out.read_text()!=payload:raise RuntimeError("evaluation differs")
    out.write_text(payload);print(json.dumps(result["metrics"]))


if __name__=="__main__":main()
