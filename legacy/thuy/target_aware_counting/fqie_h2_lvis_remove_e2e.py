"""End-to-end LVIS query-pair diagnostic for fixed semantic remove rules."""
from __future__ import annotations
import json,math
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"target_aware_counting/output"
SRC=OUT/"lvis_query_switch_val_20261010_205455";SCORES=OUT/"fqie_h2_alpha_lvis_two_text_20261104_v1"
FULL=OUT/"fqie_h2_full_mass_rule_20261104_v1/evaluation.json";DEST=OUT/"fqie_h2_lvis_remove_e2e_20261104_v1"


def xyxy(b):return [b[0],b[1],b[0]+b[2],b[1]+b[3]]
def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb>inter else 0
def match50(pred,gt):
    edges=[[j for j,g in enumerate(gt) if iou(p,g)>=.5] for p in pred];owner={}
    def dfs(i,seen):
        for j in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in owner or dfs(owner[j],seen):owner[j]=i;return True
        return False
    return sum(dfs(i,set()) for i in range(len(pred)))


def main():
    gt=json.loads((SRC/"input_274/query_pairs_coco.json").read_text());pred=json.loads((SRC/"runs/no_pseudo/predictions.json").read_text())
    images={x["id"]:x for x in gt["images"]};g=defaultdict(list);p=defaultdict(list)
    for a in gt["annotations"]:g[a["image_id"]].append(xyxy(a["bbox"]))
    score={}
    for path in (SCORES/"scores").glob("*.json"):
        d=json.loads(path.read_text());score[d["candidate_id"]]=d
    for a in pred["annotations"]:
        cid=f"lvis_qs:{a['image_id']}:{a['id']}";d=score[cid]
        p[a["image_id"]].append({"box":xyxy(a["bbox"]),"raw":d["alpha_raw_full"],"margin":d["alpha_query_minus_generic_object"]})
    thresholds=json.loads(FULL.read_text())["rules"]
    rules={"no_op":{},"raw95":thresholds["raw95"]["thresholds"],"generic95":thresholds["generic95"]["thresholds"],
           "raw_generic_bonferroni95":thresholds["raw_generic_bonferroni95"]["thresholds"]}
    result={"status":"LVIS_DEVELOPMENT_END_TO_END_REMOVE_DIAGNOSTIC","denominators":{
        "query_pairs":len(images),"present_pairs":sum(x["query_status"]=="present" for x in images.values()),
        "verified_absent_pairs":sum(x["query_status"]=="verified_absent" for x in images.values()),"gt_instances":len(gt["annotations"])},"rules":{}}
    for name,t in rules.items():
        def keep(x):return ("raw" not in t or x["raw"]>=t["raw"]) and ("margin" not in t or x["margin"]>=t["margin"])
        errors=[];sq=[];tp=0;boxes=0;abs_mass=0;abs_active=0;false_zero=0
        for iid,im in images.items():
            kept=[x["box"] for x in p[iid] if keep(x)];count=len(kept);boxes+=count
            if im["query_status"]=="present":
                e=abs(count-im["gt_count"]);errors.append(e);sq.append((count-im["gt_count"])**2);tp+=match50(kept,g[iid]);false_zero+=count==0
            else:
                abs_mass+=count;abs_active+=count>0
        result["rules"][name]={"thresholds":t,"all_predicted_boxes":boxes,
          "present_count_mae":sum(errors)/len(errors),"present_count_rmse":math.sqrt(sum(sq)/len(sq)),
          "present_tp50":tp,"present_gt_instances":len(gt["annotations"]),"present_false_zero_pairs":false_zero,
          "verified_absent_count_mass":abs_mass,"verified_absent_active_pairs":abs_active}
    result["limitations"]=["LVIS VAL is development and was used for threshold calibration.",
      "Only frozen no-pseudo M0 proposals are filtered; A4/B0 candidate recall is not tested here.",
      "TP50 is maximum bipartite IoU>=0.5 matching, not COCO AP.","No independent confirmation or TEST access."]
    DEST.mkdir(parents=True,exist_ok=True);out=DEST/"evaluation.json";payload=json.dumps(result,indent=2)+"\n"
    if out.exists() and out.read_text()!=payload:raise RuntimeError("evaluation differs")
    out.write_text(payload);print(json.dumps(result["rules"]))


if __name__=="__main__":main()
