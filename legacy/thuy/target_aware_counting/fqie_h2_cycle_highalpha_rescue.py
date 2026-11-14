"""One predeclared conservative rescue: Cycle YES or top-quartile Alpha evidence."""
from __future__ import annotations
import json,math
from collections import Counter,defaultdict
from pathlib import Path
from fqie_h2_lvis_remove_e2e import OUT,SRC,SCORES,xyxy,match50
BASE=OUT/"fqie_h2_alpha_lvis_full_20261104_v1";CYCLE=OUT/"fqie_h2_lvis306_cycle_unary_20261104_v1/freeze_yesno.json"
FOLDS=OUT/"fqie_h2_alpha_source_crossfit_20261104_v1/manifest.json";FSC_CYCLE=OUT/"fqie_h2_absent111_cycle_top5_20261104_v1/freeze_yesno.json"
FSC_LEDGER=OUT/"fqie_h2_evidence_ledger_20261104_v1/absent111_top5_evidence.jsonl";DEST=OUT/"fqie_h2_cycle_highalpha_rescue_20261104_v1"
def q75(v):
    v=sorted(v);return v[math.ceil(.75*len(v))-1]
def main():
    labels={x["candidate_id"]:x for x in json.loads((BASE/"eval_labels.json").read_text())["items"]}
    cycle={x["candidate_id"]:x["strict_decision"]=="yes" for x in json.loads(CYCLE.read_text())["items"]};scores={}
    for p in (SCORES/"scores").glob("*.json"):
        d=json.loads(p.read_text());scores[d["candidate_id"]]=d
    fold_doc=json.loads(FOLDS.read_text());fold_of={sid:x["fold"] for x in fold_doc["folds"] for sid in x["source_image_ids"]}
    thresholds=[]
    for f in range(5):
        tr=[scores[c] for c,l in labels.items() if l["eval_label"]=="TARGET_IOU50" and fold_of[l["source_image_id"]]!=f]
        thresholds.append({"raw":q75([x["alpha_raw_full"] for x in tr]),"margin":q75([x["alpha_query_minus_generic_object"] for x in tr]),"n":len(tr)})
    keep={}
    for c,l in labels.items():
        t=thresholds[fold_of[l["source_image_id"]]];d=scores[c];keep[c]=cycle[c] or (d["alpha_raw_full"]>=t["raw"] and d["alpha_query_minus_generic_object"]>=t["margin"])
    allpos=[scores[c] for c,l in labels.items() if l["eval_label"]=="TARGET_IOU50"]
    transfer={"raw":q75([x["alpha_raw_full"] for x in allpos]),"margin":q75([x["alpha_query_minus_generic_object"] for x in allpos]),"n":len(allpos)}
    fcycle={x["candidate_id"]:x["strict_decision"]=="yes" for x in json.loads(FSC_CYCLE.read_text())["items"]}
    fledger={x["candidate_id"]:x for x in (json.loads(s) for s in FSC_LEDGER.read_text().splitlines() if s.strip())};fkeep={}
    for c,y in fcycle.items():
        z=fledger[c]["features"];fkeep[c]=y or (z["alpha_raw_full"]>=transfer["raw"] and z["alpha_query_minus_generic_object"]>=transfer["margin"])
    gt=json.loads((SRC/"input_274/query_pairs_coco.json").read_text());pred=json.loads((SRC/"runs/no_pseudo/predictions.json").read_text())
    images={x["id"]:x for x in gt["images"]};g=defaultdict(list);p=defaultdict(list)
    for a in gt["annotations"]:g[a["image_id"]].append(xyxy(a["bbox"]))
    for a in pred["annotations"]:p[a["image_id"]].append((f"lvis_qs:{a['image_id']}:{a['id']}",xyxy(a["bbox"])))
    errors=[];sq=[];tp=boxes=abs_mass=abs_active=fz=0
    for iid,im in images.items():
        boxes_i=[b for c,b in p[iid] if keep[c]];n=len(boxes_i);boxes+=n
        if im["query_status"]=="present":
            e=n-im["gt_count"];errors.append(abs(e));sq.append(e*e);tp+=match50(boxes_i,g[iid]);fz+=n==0
        else:abs_mass+=n;abs_active+=n>0
    fq=defaultdict(int)
    manifest=json.loads((OUT/"fqie_h2_absent111_cycle_top5_20261104_v1/manifest.json").read_text())
    item={x["candidate_id"]:x["item_id"] for x in manifest["items"]}
    for c,k in fkeep.items():fq[item[c]]+=k
    result={"status":"DEVELOPMENT_PREDECLARED_HIGH_ALPHA_RESCUE_DIAGNOSTIC","rule":"Cycle YES OR (raw Alpha and query-minus-generic margin each >= training-target 75th percentile)",
      "crossfit_thresholds":thresholds,"transfer_threshold":transfer,
      "lvis_candidate":{"kept_by_label":dict(Counter(labels[c]["eval_label"] for c,k in keep.items() if k)),"total_kept":sum(keep.values())},
      "lvis_end_to_end":{"all_predicted_boxes":boxes,"present_count_mae":sum(errors)/len(errors),"present_count_rmse":math.sqrt(sum(sq)/len(sq)),
        "present_tp50":tp,"present_false_zero_pairs":fz,"verified_absent_count_mass":abs_mass,"verified_absent_active_pairs":abs_active},
      "fsc_absent_top5":{"boxes_kept":sum(fkeep.values()),"boxes_total":len(fkeep),"queries_active":sum(v>0 for v in fq.values()),"queries_total":111,"top1_active":sum(k and next(x for x in manifest['items'] if x['candidate_id']==c)['native_rank']==1 for c,k in fkeep.items())},
      "limitations":["The top-quartile rescue policy is a single development diagnostic, not independently confirmed.","LVIS thresholds are source-cross-fit; FSC transfer uses all LVIS strict positives.",
        "FSC covers native top-five only; UNKNOWN LVIS proposals remain semantically unlabeled.","No further percentile or feature grid is evaluated in this run."]}
    DEST.mkdir(parents=True,exist_ok=True);path=DEST/"evaluation.json";s=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if path.exists() and path.read_text()!=s:raise RuntimeError("locked evaluation differs")
    path.write_text(s);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
