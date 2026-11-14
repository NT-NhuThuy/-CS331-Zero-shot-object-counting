"""End-to-end LVIS evaluation of frozen Cycle unary and Alpha combinations."""
from __future__ import annotations
import hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
from fqie_h2_lvis_remove_e2e import OUT,SRC,SCORES,FULL,xyxy,match50
RUN=OUT/"fqie_h2_lvis306_cycle_unary_20261104_v1";BASE=OUT/"fqie_h2_alpha_lvis_full_20261104_v1"
DEST=OUT/"fqie_h2_lvis306_cycle_eval_20261104_v1"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    labels={x["candidate_id"]:x for x in json.loads((BASE/"eval_labels.json").read_text())["items"]}
    cycle={x["candidate_id"]:x["strict_decision"]=="yes" for x in json.loads((RUN/"freeze_yesno.json").read_text())["items"]}
    scores={}
    for p in (SCORES/"scores").glob("*.json"):
        d=json.loads(p.read_text());scores[d["candidate_id"]]=d
    t=json.loads(FULL.read_text())["rules"]["raw_generic_bonferroni95"]["thresholds"]
    alpha={cid:d["alpha_raw_full"]>=t["raw"] and d["alpha_query_minus_generic_object"]>=t["margin"] for cid,d in scores.items()}
    if set(labels)!=set(cycle) or set(labels)!=set(alpha):raise RuntimeError("candidate binding mismatch")
    gt=json.loads((SRC/"input_274/query_pairs_coco.json").read_text());pred=json.loads((SRC/"runs/no_pseudo/predictions.json").read_text())
    images={x["id"]:x for x in gt["images"]};g=defaultdict(list);p=defaultdict(list)
    for a in gt["annotations"]:g[a["image_id"]].append(xyxy(a["bbox"]))
    for a in pred["annotations"]:p[a["image_id"]].append((f"lvis_qs:{a['image_id']}:{a['id']}",xyxy(a["bbox"])))
    rules={"no_op":lambda c:True,"alpha":lambda c:alpha[c],"cycle":lambda c:cycle[c],
      "alpha_and_cycle":lambda c:alpha[c] and cycle[c],"alpha_or_cycle":lambda c:alpha[c] or cycle[c]}
    def summarize(fn):
        errors=[];sq=[];tp=boxes=abs_mass=abs_active=fz=0
        for iid,im in images.items():
            kept=[b for c,b in p[iid] if fn(c)];n=len(kept);boxes+=n
            if im["query_status"]=="present":
                e=n-im["gt_count"];errors.append(abs(e));sq.append(e*e);tp+=match50(kept,g[iid]);fz+=n==0
            else:abs_mass+=n;abs_active+=n>0
        return {"all_predicted_boxes":boxes,"present_count_mae":sum(errors)/len(errors),"present_count_rmse":math.sqrt(sum(sq)/len(sq)),
          "present_tp50":tp,"present_false_zero_pairs":fz,"verified_absent_count_mass":abs_mass,"verified_absent_active_pairs":abs_active}
    candidate={}
    for name,fn in rules.items():candidate[name]={"kept_by_label":dict(Counter(labels[c]["eval_label"] for c in labels if fn(c))),"total_kept":sum(fn(c) for c in labels)}
    result={"status":"DEVELOPMENT_EXECUTABLE_CLASS_SEMANTIC_DIAGNOSTIC","denominators":{"query_pairs":274,"present_pairs":174,
      "verified_absent_pairs":100,"proposals":306,"strict_targets":164,"unknown_present_proposals":129,"verified_absent_proposals":13},
      "candidate_metrics":candidate,"end_to_end":{name:summarize(fn) for name,fn in rules.items()},
      "frozen_inputs":{"cycle_manifest_sha256":sha(RUN/"manifest.json"),"cycle_unary_sha256":sha(RUN/"freeze_yesno.json"),"labels_sha256":sha(BASE/"eval_labels.json")},
      "limitations":["LVIS VAL development cohort; thresholds and model selection are not independent confirmation.",
        "UNKNOWN present-query proposals are not semantic negatives, although their keep/drop decisions affect count metrics.",
        "Only frozen no-pseudo M0 proposals are filtered; A4/B0 recall and full-query facets are not evaluated.",
        "Cycle binary output is uncalibrated and does not solve relation grounding, duplicate, part, or identity."]}
    DEST.mkdir(parents=True,exist_ok=True);path=DEST/"evaluation.json";s=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if path.exists() and path.read_text()!=s:raise RuntimeError("locked evaluation differs")
    path.write_text(s);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
