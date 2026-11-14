"""Post-freeze LVIS diagnostic of CycleGRPO unary rescue for Alpha removals."""
from __future__ import annotations
import json,math,hashlib
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"target_aware_counting/output"
CAP=OUT/"fqie_h2_removed_caption_20261104_v2"
BASE=OUT/"fqie_h2_alpha_lvis_full_20261104_v1"
SCORES=OUT/"fqie_h2_alpha_lvis_two_text_20261104_v1"
RULES=OUT/"fqie_h2_full_mass_rule_20261104_v1/evaluation.json"
SRC=OUT/"lvis_query_switch_val_20261010_205455"
DEST=OUT/"fqie_h2_cyclegrpo_rescue_20261104_v1"

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def xyxy(b):return [b[0],b[1],b[0]+b[2],b[1]+b[3]]
def iou(a,b):
    z=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return z/(aa+bb-z) if aa+bb>z else 0
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
    frozen=json.loads((CAP/"freeze_yesno.json").read_text())
    labels={x["candidate_id"]:x for x in json.loads((BASE/"eval_labels.json").read_text())["items"]}
    decisions={x["candidate_id"]:x["strict_decision"] for x in frozen["items"]}
    locked=Counter(labels[c]["eval_label"] for c in decisions)
    yes=Counter(labels[c]["eval_label"] for c,d in decisions.items() if d=="yes")
    no=Counter(labels[c]["eval_label"] for c,d in decisions.items() if d=="no")

    gt=json.loads((SRC/"input_274/query_pairs_coco.json").read_text())
    pred=json.loads((SRC/"runs/no_pseudo/predictions.json").read_text())
    images={x["id"]:x for x in gt["images"]};g=defaultdict(list);p=defaultdict(list)
    for a in gt["annotations"]:g[a["image_id"]].append(xyxy(a["bbox"]))
    scores={}
    for path in (SCORES/"scores").glob("*.json"):
        d=json.loads(path.read_text());scores[d["candidate_id"]]=d
    for a in pred["annotations"]:
        cid=f"lvis_qs:{a['image_id']}:{a['id']}";d=scores[cid]
        p[a["image_id"]].append({"candidate_id":cid,"box":xyxy(a["bbox"]),"raw":d["alpha_raw_full"],
                                  "margin":d["alpha_query_minus_generic_object"]})
    t=json.loads(RULES.read_text())["rules"]["raw_generic_bonferroni95"]["thresholds"]
    def alpha(x):return x["raw"]>=t["raw"] and x["margin"]>=t["margin"]
    def summarize(rescue):
        errors=[];sq=[];tp=boxes=abs_mass=abs_active=false_zero=0
        for iid,im in images.items():
            kept=[x["box"] for x in p[iid] if alpha(x) or (rescue and decisions.get(x["candidate_id"])=="yes")]
            count=len(kept);boxes+=count
            if im["query_status"]=="present":
                e=count-im["gt_count"];errors.append(abs(e));sq.append(e*e);tp+=match50(kept,g[iid]);false_zero+=count==0
            else:abs_mass+=count;abs_active+=count>0
        return {"all_predicted_boxes":boxes,"present_count_mae":sum(errors)/len(errors),
                "present_count_rmse":math.sqrt(sum(sq)/len(sq)),"present_tp50":tp,
                "present_false_zero_pairs":false_zero,"verified_absent_count_mass":abs_mass,
                "verified_absent_active_pairs":abs_active}
    result={"status":"DEVELOPMENT_DIAGNOSTIC_NOT_A_PASSED_COMPONENT",
      "frozen_inputs":{"caption_manifest_sha256":sha(CAP/"manifest.json"),
                       "caption_outputs_sha256":sha(CAP/"freeze_outputs.json"),
                       "yesno_outputs_sha256":sha(CAP/"freeze_yesno.json")},
      "removed_set":{"total":len(decisions),"label_denominators":dict(locked),
                     "yes_by_label":dict(yes),"no_by_label":dict(no),
                     "target_rescue_rate":yes["TARGET_IOU50"]/locked["TARGET_IOU50"],
                     "verified_absent_false_activation_rate":yes["VERIFIED_ABSENT"]/locked["VERIFIED_ABSENT"]},
      "end_to_end":{"alpha_only":summarize(False),"alpha_plus_cyclegrpo_yes_rescue":summarize(True)},
      "limitations":["LVIS VAL is development and was already used for Alpha threshold calibration.",
        "Only 6 removed TARGET_IOU50 and 13 removed VERIFIED_ABSENT candidates are labeled; 21 are UNKNOWN.",
        "The same CycleGRPO checkpoint produced region tokens and the unary decision; this is not independent evidence.",
        "Free-form captions remain evidence only and were not converted to labels.",
        "No TEST access, no confidence claim, no final selector claim."]}
    DEST.mkdir(parents=True,exist_ok=True);path=DEST/"evaluation.json";payload=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if path.exists() and path.read_text()!=payload:raise RuntimeError("evaluation changed")
    path.write_text(payload);print(json.dumps(result,indent=2))

if __name__=="__main__":main()
