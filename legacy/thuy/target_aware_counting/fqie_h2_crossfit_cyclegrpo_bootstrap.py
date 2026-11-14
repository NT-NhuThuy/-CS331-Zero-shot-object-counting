"""Paired source bootstrap for cross-fit Alpha versus CycleGRPO rescue."""
from __future__ import annotations
import json,math,random
from collections import defaultdict
from fqie_h2_lvis_remove_e2e import OUT,SRC,xyxy,match50

CROSS=OUT/"fqie_h2_alpha_source_crossfit_20261104_v1/manifest.json"
CYCLE=OUT/"fqie_h2_crossfit_cyclegrpo_20261104_v1/freeze_yesno.json"
DEST=OUT/"fqie_h2_crossfit_cyclegrpo_eval_20261104_v1"
def quantile(x,q):
    x=sorted(x);i=(len(x)-1)*q;lo=math.floor(i);hi=math.ceil(i)
    return x[lo] if lo==hi else x[lo]*(hi-i)+x[hi]*(i-lo)
def main():
    alpha={x["candidate_id"]:x["keep"] for x in json.loads(CROSS.read_text())["decisions"]}
    unary={x["candidate_id"]:x["strict_decision"] for x in json.loads(CYCLE.read_text())["items"]}
    gt=json.loads((SRC/"input_274/query_pairs_coco.json").read_text());pred=json.loads((SRC/"runs/no_pseudo/predictions.json").read_text())
    images={x["id"]:x for x in gt["images"]};g=defaultdict(list);p=defaultdict(list)
    for a in gt["annotations"]:g[a["image_id"]].append(xyxy(a["bbox"]))
    for a in pred["annotations"]:p[a["image_id"]].append((f"lvis_qs:{a['image_id']}:{a['id']}",xyxy(a["bbox"])))
    per=defaultdict(list)
    for iid,im in images.items():
        if im["query_status"]!="present":continue
        a=[b for c,b in p[iid] if alpha[c]];r=[b for c,b in p[iid] if alpha[c] or unary.get(c)=="yes"]
        per[im["source_image_id"]].append({"gt":im["gt_count"],"a":len(a),"r":len(r),
          "a_tp":match50(a,g[iid]),"r_tp":match50(r,g[iid])})
    sources=sorted(per);rng=random.Random(20261104);metrics={k:[] for k in ("mae","rmse","tp50","false_zero")}
    for _ in range(10000):
        rows=[r for s in [rng.choice(sources) for _ in sources] for r in per[s]];n=len(rows)
        metrics["mae"].append(sum(abs(x["r"]-x["gt"])-abs(x["a"]-x["gt"]) for x in rows)/n)
        metrics["rmse"].append(math.sqrt(sum((x["r"]-x["gt"])**2 for x in rows)/n)-math.sqrt(sum((x["a"]-x["gt"])**2 for x in rows)/n))
        metrics["tp50"].append(sum(x["r_tp"]-x["a_tp"] for x in rows))
        metrics["false_zero"].append(sum((x["r"]==0)-(x["a"]==0) for x in rows))
    def summary(v,direction):return {"mean":sum(v)/len(v),"ci95":[quantile(v,.025),quantile(v,.975)],
      "bootstrap_fraction_improved":sum((x<0 if direction=="negative" else x>0) for x in v)/len(v)}
    result={"status":"DEVELOPMENT_PAIRED_SOURCE_BOOTSTRAP","seed":20261104,"replicates":10000,"source_clusters":len(sources),
      "difference_convention":"Cycle rescue minus source-cross-fit Alpha; negative improves errors/false-zero, positive improves TP50",
      "mae_difference":summary(metrics["mae"],"negative"),"rmse_difference":summary(metrics["rmse"],"negative"),
      "tp50_difference":summary(metrics["tp50"],"positive"),"false_zero_difference":summary(metrics["false_zero"],"negative"),
      "limitations":["Same LVIS development collection; cross-fit prevents direct threshold leakage but is not external confirmation.",
        "UNKNOWN rescued candidates can affect count metrics.","Bootstrap addresses source sampling variation, not model-selection bias."]}
    DEST.mkdir(parents=True,exist_ok=True);path=DEST/"bootstrap.json";s=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if path.exists() and path.read_text()!=s:raise RuntimeError("locked bootstrap differs")
    path.write_text(s);print(json.dumps(result))
if __name__=="__main__":main()
