"""Transfer LVIS text-control thresholds to absent111 top-5 candidates."""
from __future__ import annotations
import glob,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"target_aware_counting/output/fqie_h2_alpha_absent111_text_controls_20261104_v1"
CAL=ROOT/"target_aware_counting/output/fqie_h2_alpha_text_controls_20261104_v1/evaluation.json"


def main():
    cal=json.loads(CAL.read_text());rows=[]
    for p in glob.glob(str(RUN/"scores/*.json")):
        d=json.load(open(p));s=d["scores"];t=s["target"]
        rows.append({"item_id":d["item_id"],"native_rank":d["native_rank"],"raw":t,
            "minus_generic":t-s["generic_object"],"minus_background":t-s["background"],
            "minus_without":t-s["without_target"],"minus_different":t-s["different_target"],
            "minus_max_control":t-max(s[k] for k in ("generic_object","background","without_target","different_target"))})
    if len(rows)!=552:raise RuntimeError("incomplete")
    result={"status":"DEVELOPMENT_TEXT_CONTROL_TRANSFER","candidates":552,"queries":111,"metrics":{}}
    for f,metric in cal["metrics"].items():
        points=[]
        for op in metric["operating_points"]:
            kept=[x for x in rows if x[f]>=op["threshold"]]
            points.append({"LVIS_target_retention":op["retention"],"threshold":op["threshold"],
                "absent_candidates_kept":len(kept),"absent_queries_active":len({x["item_id"] for x in kept}),
                "absent_top1_active":sum(x["native_rank"]==1 for x in kept)})
        result["metrics"][f]=points
    result["limitations"]=["Top-five candidates only.","Prompts and thresholds selected on LVIS development.",
        "Negation margins are empirical scores, not logical negation."]
    payload=json.dumps(result,indent=2)+"\n";out=RUN/"evaluation.json"
    if out.exists() and out.read_text()!=payload:raise RuntimeError("evaluation differs")
    out.write_text(payload);print(json.dumps(result["metrics"]))


if __name__=="__main__":main()
