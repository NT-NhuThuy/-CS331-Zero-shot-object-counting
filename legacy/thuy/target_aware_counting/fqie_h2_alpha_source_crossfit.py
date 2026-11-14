"""Five-fold source-disjoint cross-fit of the fixed Alpha remove policy.

Fold assignment reads labels but never model scores. Each held-out threshold is
calibrated only from strict positives belonging to the other source images.
"""
from __future__ import annotations
import hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
from fqie_h2_lvis_remove_e2e import OUT,SRC,SCORES,xyxy,match50

LABELS=OUT/"fqie_h2_alpha_lvis_full_20261104_v1/eval_labels.json"
DEST=OUT/"fqie_h2_alpha_source_crossfit_20261104_v1"
K=5

def threshold(values,retention):
    values=sorted(values,reverse=True);return values[math.ceil(retention*len(values))-1]
def write_once(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);payload=json.dumps(obj,indent=2,sort_keys=True)+"\n"
    if path.exists() and path.read_text()!=payload:raise RuntimeError(f"locked output differs: {path}")
    path.write_text(payload)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    label_doc=json.loads(LABELS.read_text());labels={x["candidate_id"]:x for x in label_doc["items"]}
    by_source=defaultdict(list)
    for x in label_doc["items"]:by_source[x["source_image_id"]].append(x)
    # Greedy balance by strict-positive mass; label-only and score-blind.
    folds=[[] for _ in range(K)];loads=[0]*K
    for sid,rows in sorted(by_source.items(),key=lambda z:(-sum(x["eval_label"]=="TARGET_IOU50" for x in z[1]),z[0])):
        f=min(range(K),key=lambda i:(loads[i],len(folds[i]),i));folds[f].append(sid)
        loads[f]+=sum(x["eval_label"]=="TARGET_IOU50" for x in rows)
    fold_of={sid:f for f,sids in enumerate(folds) for sid in sids}
    scores={}
    for path in (SCORES/"scores").glob("*.json"):
        d=json.loads(path.read_text());scores[d["candidate_id"]]=d
    if set(scores)!=set(labels):raise RuntimeError("score/label candidate mismatch")
    thresholds=[]
    for f in range(K):
        train=[scores[cid] for cid,l in labels.items() if l["eval_label"]=="TARGET_IOU50" and fold_of[l["source_image_id"]]!=f]
        thresholds.append({"raw":threshold([x["alpha_raw_full"] for x in train],.975),
                           "margin":threshold([x["alpha_query_minus_generic_object"] for x in train],.975),
                           "calibration_target_n":len(train)})
    decisions={}
    for cid,l in labels.items():
        f=fold_of[l["source_image_id"]];d=scores[cid];t=thresholds[f]
        decisions[cid]={"fold":f,"keep":d["alpha_raw_full"]>=t["raw"] and d["alpha_query_minus_generic_object"]>=t["margin"]}

    gt=json.loads((SRC/"input_274/query_pairs_coco.json").read_text());pred=json.loads((SRC/"runs/no_pseudo/predictions.json").read_text())
    images={x["id"]:x for x in gt["images"]};g=defaultdict(list);p=defaultdict(list)
    for a in gt["annotations"]:g[a["image_id"]].append(xyxy(a["bbox"]))
    for a in pred["annotations"]:
        cid=f"lvis_qs:{a['image_id']}:{a['id']}";p[a["image_id"]].append((cid,xyxy(a["bbox"])))
    errors=[];sq=[];tp=boxes=abs_mass=abs_active=false_zero=0
    for iid,im in images.items():
        kept=[box for cid,box in p[iid] if decisions[cid]["keep"]];n=len(kept);boxes+=n
        if im["query_status"]=="present":
            e=n-im["gt_count"];errors.append(abs(e));sq.append(e*e);tp+=match50(kept,g[iid]);false_zero+=n==0
        else:abs_mass+=n;abs_active+=n>0
    kept_by_label=Counter(labels[cid]["eval_label"] for cid,d in decisions.items() if d["keep"])
    total_by_label=Counter(x["eval_label"] for x in labels.values())
    result={"status":"DEVELOPMENT_SOURCE_DISJOINT_CROSS_FIT","folds":K,
      "fold_assignment_policy":"greedy strict-positive count balance by source; scores unread during assignment",
      "fold_target_loads":loads,"thresholds":thresholds,
      "candidate_metrics":{"total_by_label":dict(total_by_label),"kept_by_label":dict(kept_by_label),
        "strict_target_retention":kept_by_label["TARGET_IOU50"]/total_by_label["TARGET_IOU50"]},
      "end_to_end":{"all_predicted_boxes":boxes,"present_count_mae":sum(errors)/len(errors),
        "present_count_rmse":math.sqrt(sum(sq)/len(sq)),"present_tp50":tp,
        "present_false_zero_pairs":false_zero,"verified_absent_count_mass":abs_mass,
        "verified_absent_active_pairs":abs_active},
      "limitations":["Cross-fit removes direct threshold evaluation leakage but uses the same 71 LVIS development sources overall.",
        "Only 13 verified-absent candidates and two verified-absent query pairs are available.",
        "Present-query proposals below IoU50 remain UNKNOWN, never negative.","No TEST access or independent dataset claim."]}
    manifest={"status":"LOCKED_CROSS_FIT_DECISIONS","labels_sha256":sha(LABELS),"score_manifest_sha256":sha(SCORES/"manifest.json"),
      "folds":[{"fold":f,"source_image_ids":sorted(sids)} for f,sids in enumerate(folds)],
      "decisions":[{"candidate_id":cid,**decisions[cid]} for cid in sorted(decisions)]}
    write_once(DEST/"manifest.json",manifest);write_once(DEST/"evaluation.json",result)
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
