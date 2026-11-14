"""Evaluate frozen Alpha-CLIP text-control margins on LVIS development."""
from __future__ import annotations
import glob,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"target_aware_counting/output/fqie_h2_alpha_lvis_full_20261104_v1"
RUN=ROOT/"target_aware_counting/output/fqie_h2_alpha_text_controls_20261104_v1"


def auc(p,n):return (sum(a>b for a in p for b in n)+.5*sum(a==b for a in p for b in n))/(len(p)*len(n))
def op(rows,f,r):
    p=sorted((x[f] for x in rows if x["label"]=="TARGET_IOU50"),reverse=True);t=p[math.ceil(r*len(p))-1]
    return {"retention":r,"threshold":t,"absent_kept":sum(x["label"]=="VERIFIED_ABSENT" and x[f]>=t for x in rows)}


def main():
    labels={x["candidate_id"]:x["eval_label"] for x in json.loads((BASE/"eval_labels.json").read_text())["items"]};rows=[]
    for p in glob.glob(str(RUN/"scores/*.json")):
        d=json.load(open(p));s=d["scores"];target=s["target"]
        rows.append({"label":labels[d["candidate_id"]],"raw":target,"minus_generic":target-s["generic_object"],
            "minus_background":target-s["background"],"minus_without":target-s["without_target"],
            "minus_different":target-s["different_target"],
            "minus_max_control":target-max(s[k] for k in ("generic_object","background","without_target","different_target"))})
    pos=[x for x in rows if x["label"]=="TARGET_IOU50"];neg=[x for x in rows if x["label"]=="VERIFIED_ABSENT"]
    result={"status":"DEVELOPMENT_TEXT_CONTROL_ABLATION","metrics":{}}
    for f in ("raw","minus_generic","minus_background","minus_without","minus_different","minus_max_control"):
        result["metrics"][f]={"auc":auc([x[f] for x in pos],[x[f] for x in neg]),
            "target_mean":sum(x[f] for x in pos)/len(pos),"absent_mean":sum(x[f] for x in neg)/len(neg),
            "operating_points":[op(rows,f,r) for r in (1,.95,.9)]}
    result["limitations"]=["Only 13 absent proposals from two source images.",
        "Prompt controls are selected and evaluated on development.","CLIP negation is not assumed to be logically valid."]
    payload=json.dumps(result,indent=2)+"\n";out=RUN/"evaluation.json"
    if out.exists() and out.read_text()!=payload:raise RuntimeError("evaluation differs")
    out.write_text(payload);print(json.dumps(result["metrics"]))


if __name__=="__main__":main()
