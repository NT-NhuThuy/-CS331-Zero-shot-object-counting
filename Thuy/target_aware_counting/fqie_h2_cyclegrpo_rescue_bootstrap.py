"""Paired source-cluster bootstrap: Alpha-only versus CycleGRPO rescue."""
from __future__ import annotations
import json,math,random
from collections import defaultdict
from pathlib import Path
from fqie_h2_lvis_remove_e2e import OUT,SRC,SCORES,FULL,xyxy,match50

CAP=OUT/"fqie_h2_removed_caption_20261104_v2";DEST=OUT/"fqie_h2_cyclegrpo_rescue_bootstrap_20261104_v2"
def quantile(x,q):
    x=sorted(x);i=(len(x)-1)*q;lo=math.floor(i);hi=math.ceil(i)
    return x[lo] if lo==hi else x[lo]*(hi-i)+x[hi]*(i-lo)
def main():
    gt=json.loads((SRC/"input_274/query_pairs_coco.json").read_text());pred=json.loads((SRC/"runs/no_pseudo/predictions.json").read_text())
    images={x["id"]:x for x in gt["images"]};g=defaultdict(list);p=defaultdict(list);score={}
    for a in gt["annotations"]:g[a["image_id"]].append(xyxy(a["bbox"]))
    for path in (SCORES/"scores").glob("*.json"):
        d=json.loads(path.read_text());score[d["candidate_id"]]=d
    decisions={x["candidate_id"]:x["strict_decision"] for x in json.loads((CAP/"freeze_yesno.json").read_text())["items"]}
    for a in pred["annotations"]:
        cid=f"lvis_qs:{a['image_id']}:{a['id']}";d=score[cid]
        p[a["image_id"]].append({"cid":cid,"box":xyxy(a["bbox"]),"raw":d["alpha_raw_full"],"margin":d["alpha_query_minus_generic_object"]})
    t=json.loads(FULL.read_text())["rules"]["raw_generic_bonferroni95"]["thresholds"]
    per=defaultdict(list)
    for iid,im in images.items():
        if im["query_status"]!="present":continue
        alpha=[x["box"] for x in p[iid] if x["raw"]>=t["raw"] and x["margin"]>=t["margin"]]
        rescue=[x["box"] for x in p[iid] if (x["raw"]>=t["raw"] and x["margin"]>=t["margin"]) or decisions.get(x["cid"])=="yes"]
        per[im["source_image_id"]].append({"gt":im["gt_count"],"a":len(alpha),"r":len(rescue),
          "a_tp":match50(alpha,g[iid]),"r_tp":match50(rescue,g[iid])})
    sources=sorted(per);rng=random.Random(20261104);mae=[];rmse=[];tp=[];fz=[]
    for _ in range(10000):
        rows=[r for s in [rng.choice(sources) for _ in sources] for r in per[s]];n=len(rows)
        mae.append(sum(abs(r["r"]-r["gt"])-abs(r["a"]-r["gt"]) for r in rows)/n)
        rmse.append(math.sqrt(sum((r["r"]-r["gt"])**2 for r in rows)/n)-math.sqrt(sum((r["a"]-r["gt"])**2 for r in rows)/n))
        tp.append(sum(r["r_tp"]-r["a_tp"] for r in rows));fz.append(sum((r["r"]==0)-(r["a"]==0) for r in rows))
    def ci(v,direction):return {"mean":sum(v)/len(v),"ci95":[quantile(v,.025),quantile(v,.975)],
                                "bootstrap_fraction_improved":sum((x<0 if direction=="negative" else x>0) for x in v)/len(v)}
    result={"status":"DEVELOPMENT_PAIRED_SOURCE_BOOTSTRAP","seed":20261104,"replicates":10000,"source_clusters":len(sources),
      "difference_convention":"rescue minus Alpha-only; negative improves error/false-zero, positive improves TP50",
      "mae_difference":ci(mae,"negative"),"rmse_difference":ci(rmse,"negative"),
      "tp50_difference":ci(tp,"positive"),"false_zero_difference":ci(fz,"negative"),
      "limitations":["Only 6 labeled removed targets and 13 labeled removed absents.","UNKNOWN rescued boxes can change count metrics.",
                     "Bootstrap quantifies source sampling variation, not calibration or selection bias."]}
    DEST.mkdir(parents=True,exist_ok=True);path=DEST/"bootstrap.json";payload=json.dumps(result,indent=2)+"\n"
    if path.exists() and path.read_text()!=payload:raise RuntimeError("bootstrap changed")
    path.write_text(payload);print(json.dumps(result))
if __name__=="__main__":main()
