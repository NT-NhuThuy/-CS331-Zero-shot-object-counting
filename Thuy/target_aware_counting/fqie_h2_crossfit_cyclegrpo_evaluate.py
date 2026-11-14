"""Post-freeze evaluation of CycleGRPO rescue for Alpha source cross-fit."""
from __future__ import annotations
import hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
from fqie_h2_lvis_remove_e2e import OUT,SRC,xyxy,match50

LABELS=OUT/"fqie_h2_alpha_lvis_full_20261104_v1/eval_labels.json"
CROSS=OUT/"fqie_h2_alpha_source_crossfit_20261104_v1/manifest.json"
CYCLE=OUT/"fqie_h2_crossfit_cyclegrpo_20261104_v1"
DEST=OUT/"fqie_h2_crossfit_cyclegrpo_eval_20261104_v2"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_once(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);s=json.dumps(x,indent=2,sort_keys=True)+"\n"
    if p.exists() and p.read_text()!=s:raise RuntimeError(f"locked output differs: {p}")
    p.write_text(s)
def main():
    labels={x["candidate_id"]:x for x in json.loads(LABELS.read_text())["items"]}
    alpha={x["candidate_id"]:x["keep"] for x in json.loads(CROSS.read_text())["decisions"]}
    unary={x["candidate_id"]:x["strict_decision"] for x in json.loads((CYCLE/"freeze_yesno.json").read_text())["items"]}
    if set(unary)!={c for c,k in alpha.items() if not k}:raise RuntimeError("unary/removal mismatch")
    removed=Counter(labels[c]["eval_label"] for c in unary);yes=Counter(labels[c]["eval_label"] for c,d in unary.items() if d=="yes")
    no=Counter(labels[c]["eval_label"] for c,d in unary.items() if d=="no")
    gt=json.loads((SRC/"input_274/query_pairs_coco.json").read_text());pred=json.loads((SRC/"runs/no_pseudo/predictions.json").read_text())
    images={x["id"]:x for x in gt["images"]};g=defaultdict(list);p=defaultdict(list)
    for a in gt["annotations"]:g[a["image_id"]].append(xyxy(a["bbox"]))
    for a in pred["annotations"]:
        cid=f"lvis_qs:{a['image_id']}:{a['id']}";p[a["image_id"]].append((cid,xyxy(a["bbox"])))
    def summarize(mode):
        errors=[];sq=[];tp=boxes=abs_mass=abs_active=fz=0
        for iid,im in images.items():
            kept=[box for cid,box in p[iid] if mode=="no_op" or alpha[cid] or (mode=="rescue" and unary.get(cid)=="yes")];n=len(kept);boxes+=n
            if im["query_status"]=="present":
                e=n-im["gt_count"];errors.append(abs(e));sq.append(e*e);tp+=match50(kept,g[iid]);fz+=n==0
            else:abs_mass+=n;abs_active+=n>0
        return {"all_predicted_boxes":boxes,"present_count_mae":sum(errors)/len(errors),
          "present_count_rmse":math.sqrt(sum(sq)/len(sq)),"present_tp50":tp,"present_false_zero_pairs":fz,
          "verified_absent_count_mass":abs_mass,"verified_absent_active_pairs":abs_active}
    kept_targets=sum(l["eval_label"]=="TARGET_IOU50" and (alpha[c] or unary.get(c)=="yes") for c,l in labels.items())
    result={"status":"DEVELOPMENT_SOURCE_DISJOINT_RESCUE_DIAGNOSTIC_NOT_GATE_PASS",
      "frozen_inputs":{"crossfit_manifest_sha256":sha(CROSS),"cycle_manifest_sha256":sha(CYCLE/"manifest.json"),
        "caption_freeze_sha256":sha(CYCLE/"freeze_outputs.json"),"unary_freeze_sha256":sha(CYCLE/"freeze_yesno.json")},
      "removed_set":{"total":len(unary),"label_denominators":dict(removed),"yes_by_label":dict(yes),"no_by_label":dict(no)},
      "candidate_metrics":{"rescued_strict_targets":yes["TARGET_IOU50"],"removed_strict_targets":removed["TARGET_IOU50"],
        "post_rescue_strict_target_kept":kept_targets,"strict_target_total":164,"post_rescue_strict_target_retention":kept_targets/164,
        "rescued_verified_absent":yes["VERIFIED_ABSENT"]},
      "end_to_end":{"no_op":summarize("no_op"),"alpha_crossfit_only":summarize("alpha"),
                    "alpha_crossfit_plus_cycle_yes":summarize("rescue")},
      "limitations":["Cross-fit is source-disjoint per held-out fold but still uses the same LVIS development collection.",
        "Only 13 verified-absent candidates and two absent query pairs exist.","UNKNOWN proposals are not assigned semantic correctness.",
        "CycleGRPO provides both region tokens and unary decisions, so evidence is not independent.","No TEST access or final selector claim."]}
    write_once(DEST/"evaluation.json",result);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
