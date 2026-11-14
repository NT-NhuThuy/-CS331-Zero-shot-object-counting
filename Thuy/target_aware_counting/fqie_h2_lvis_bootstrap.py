"""Source-clustered paired bootstrap for no-op vs fixed raw+generic rule."""
from __future__ import annotations
import json,math,random
from collections import defaultdict
from pathlib import Path
from fqie_h2_lvis_remove_e2e import ROOT,OUT,SRC,SCORES,FULL,xyxy,match50
DEST=OUT/"fqie_h2_lvis_remove_e2e_20261104_v1"


def quantile(x,q):
    x=sorted(x);i=(len(x)-1)*q;lo=math.floor(i);hi=math.ceil(i)
    return x[lo] if lo==hi else x[lo]*(hi-i)+x[hi]*(i-lo)


def main():
    gt=json.loads((SRC/"input_274/query_pairs_coco.json").read_text());pred=json.loads((SRC/"runs/no_pseudo/predictions.json").read_text())
    images={x["id"]:x for x in gt["images"]};g=defaultdict(list);p=defaultdict(list);score={}
    for a in gt["annotations"]:g[a["image_id"]].append(xyxy(a["bbox"]))
    for path in (SCORES/"scores").glob("*.json"):
        d=json.loads(path.read_text());score[d["candidate_id"]]=d
    for a in pred["annotations"]:
        d=score[f"lvis_qs:{a['image_id']}:{a['id']}"];p[a["image_id"]].append({"box":xyxy(a["bbox"]),"raw":d["alpha_raw_full"],"margin":d["alpha_query_minus_generic_object"]})
    t=json.loads(FULL.read_text())["rules"]["raw_generic_bonferroni95"]["thresholds"]
    per=defaultdict(list)
    for iid,im in images.items():
        if im["query_status"]!="present":continue
        base=[x["box"] for x in p[iid]];filt=[x["box"] for x in p[iid] if x["raw"]>=t["raw"] and x["margin"]>=t["margin"]]
        per[im["source_image_id"]].append({"gt":im["gt_count"],"base":len(base),"filt":len(filt),
            "base_tp":match50(base,g[iid]),"filt_tp":match50(filt,g[iid])})
    sources=sorted(per);rng=random.Random(20261104);mae=[];rmse=[];tp=[];fz=[]
    for _ in range(10000):
        sample=[rng.choice(sources) for _ in sources];rows=[r for s in sample for r in per[s]];n=len(rows)
        mae.append(sum(abs(r["filt"]-r["gt"])-abs(r["base"]-r["gt"]) for r in rows)/n)
        rb=math.sqrt(sum((r["base"]-r["gt"])**2 for r in rows)/n);rf=math.sqrt(sum((r["filt"]-r["gt"])**2 for r in rows)/n);rmse.append(rf-rb)
        tp.append(sum(r["filt_tp"]-r["base_tp"] for r in rows));fz.append(sum((r["filt"]==0)-(r["base"]==0) for r in rows))
    def ci(v):return {"mean":sum(v)/len(v),"ci95":[quantile(v,.025),quantile(v,.975)],"p_improvement":sum(x<0 for x in v)/len(v)}
    result={"status":"DEVELOPMENT_PAIRED_SOURCE_BOOTSTRAP","seed":20261104,"replicates":10000,"source_clusters":len(sources),
      "difference_convention":"fixed_rule minus no_op; negative improves error metrics",
      "mae_difference":ci(mae),"rmse_difference":ci(rmse),"tp50_difference":ci(tp),"false_zero_difference":ci(fz),
      "limitations":["Same LVIS development sources used for calibration.","Bootstrap quantifies sampling variation, not selection bias.",
                     "Absent effect has only two active baseline pairs and is not bootstrapped as population evidence."]}
    out=DEST/"bootstrap.json";payload=json.dumps(result,indent=2)+"\n"
    if out.exists() and out.read_text()!=payload:raise RuntimeError("bootstrap differs")
    out.write_text(payload);print(json.dumps(result))


if __name__=="__main__":main()
