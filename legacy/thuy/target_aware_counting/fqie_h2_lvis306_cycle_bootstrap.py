"""Paired source bootstrap for Cycle-only versus no-op on LVIS306."""
from __future__ import annotations
import json,math,random
from collections import defaultdict
from fqie_h2_lvis_remove_e2e import OUT,SRC,xyxy,match50
RUN=OUT/"fqie_h2_lvis306_cycle_unary_20261104_v1";DEST=OUT/"fqie_h2_lvis306_cycle_eval_20261104_v1"
def q(v,p):
    v=sorted(v);i=(len(v)-1)*p;a=math.floor(i);b=math.ceil(i);return v[a] if a==b else v[a]*(b-i)+v[b]*(i-a)
def main():
    cycle={x["candidate_id"]:x["strict_decision"]=="yes" for x in json.loads((RUN/"freeze_yesno.json").read_text())["items"]}
    gt=json.loads((SRC/"input_274/query_pairs_coco.json").read_text());pred=json.loads((SRC/"runs/no_pseudo/predictions.json").read_text())
    images={x["id"]:x for x in gt["images"]};g=defaultdict(list);p=defaultdict(list)
    for a in gt["annotations"]:g[a["image_id"]].append(xyxy(a["bbox"]))
    for a in pred["annotations"]:p[a["image_id"]].append((f"lvis_qs:{a['image_id']}:{a['id']}",xyxy(a["bbox"])))
    per=defaultdict(list)
    for iid,im in images.items():
        if im["query_status"]!="present":continue
        base=[b for c,b in p[iid]];filt=[b for c,b in p[iid] if cycle[c]]
        per[im["source_image_id"]].append({"gt":im["gt_count"],"b":len(base),"f":len(filt),"bt":match50(base,g[iid]),"ft":match50(filt,g[iid])})
    sources=sorted(per);rng=random.Random(20261104);vals={k:[] for k in ("mae","rmse","tp50","false_zero")}
    for _ in range(10000):
        rows=[r for s in [rng.choice(sources) for _ in sources] for r in per[s]];n=len(rows)
        vals["mae"].append(sum(abs(x["f"]-x["gt"])-abs(x["b"]-x["gt"]) for x in rows)/n)
        vals["rmse"].append(math.sqrt(sum((x["f"]-x["gt"])**2 for x in rows)/n)-math.sqrt(sum((x["b"]-x["gt"])**2 for x in rows)/n))
        vals["tp50"].append(sum(x["ft"]-x["bt"] for x in rows));vals["false_zero"].append(sum((x["f"]==0)-(x["b"]==0) for x in rows))
    def s(v,d):return {"mean":sum(v)/len(v),"ci95":[q(v,.025),q(v,.975)],"bootstrap_fraction_improved":sum((x<0 if d=="negative" else x>0) for x in v)/len(v)}
    result={"status":"DEVELOPMENT_PAIRED_SOURCE_BOOTSTRAP","seed":20261104,"replicates":10000,"source_clusters":len(sources),
      "difference_convention":"Cycle minus no-op; negative improves errors/false-zero, positive improves TP50",
      "mae_difference":s(vals["mae"],"negative"),"rmse_difference":s(vals["rmse"],"negative"),"tp50_difference":s(vals["tp50"],"positive"),
      "false_zero_difference":s(vals["false_zero"],"negative"),"limitations":["Same LVIS development sources used for feature selection.",
        "Bootstrap measures source sampling variation, not model-selection bias or external transfer.","Absent population effect is not included in this present-pair bootstrap."]}
    DEST.mkdir(parents=True,exist_ok=True);path=DEST/"bootstrap.json";text=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if path.exists() and path.read_text()!=text:raise RuntimeError("locked bootstrap differs")
    path.write_text(text);print(json.dumps(result))
if __name__=="__main__":main()
