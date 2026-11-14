"""Five-percent target-risk semantic remove controls with Bonferroni allocation."""
from __future__ import annotations
import json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"target_aware_counting/output"
LEDGER=OUT/"fqie_h2_evidence_ledger_20261104_v1";LABELS=OUT/"fqie_h2_alpha_lvis_full_20261104_v1/eval_labels.json"
DEST=OUT/"fqie_h2_bounded_loss_remove_20261104_v1"
FEATURES={"native":"native_score","raw_alpha":"alpha_raw_full","generic_margin":"alpha_query_minus_generic_object"}
RULES={"raw_alpha_95":["raw_alpha"],"generic_margin_95":["generic_margin"],
       "raw_generic_bonferroni95":["raw_alpha","generic_margin"],
       "native_raw_generic_bonferroni95":["native","raw_alpha","generic_margin"]}


def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write_once(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);b=(json.dumps(v,indent=2)+"\n").encode()
    if p.exists() and p.read_bytes()!=b:raise RuntimeError(f"locked output differs: {p}")
    p.write_bytes(b)
def threshold(pos,feature,retention):
    values=sorted((x["features"][FEATURES[feature]] for x in pos),reverse=True)
    return values[math.ceil(retention*len(values))-1]
def keep(x,names,thresholds):return all(x["features"][FEATURES[n]]>=thresholds[n] for n in names)


def main():
    lvis=rows(LEDGER/"lvis_query_switch_evidence.jsonl");absent=rows(LEDGER/"absent111_top5_evidence.jsonl")
    labels={x["candidate_id"]:x for x in json.loads(LABELS.read_text())["items"]}
    pos=[x for x in lvis if labels[x["candidate_id"]]["eval_label"]=="TARGET_IOU50"]
    result={"status":"DEVELOPMENT_COMPONENT_DIAGNOSTIC","target_risk_budget":.05,
      "threshold_policy":"single feature retains 95%; k-feature conjunction allocates 0.05/k loss to each feature",
      "rules":{}}
    for name,names in RULES.items():
        per_feature_retention=1-.05/len(names)
        ts={n:threshold(pos,n,per_feature_retention) for n in names}
        lk=[x for x in lvis if keep(x,names,ts)];ak=[x for x in absent if keep(x,names,ts)]
        target_kept=sum(labels[x["candidate_id"]]["eval_label"]=="TARGET_IOU50" for x in lk)
        neg_kept=sum(labels[x["candidate_id"]]["eval_label"]=="VERIFIED_ABSENT" for x in lk)
        unknown_total=sum(v["eval_label"]=="UNKNOWN" for v in labels.values())
        unknown_kept=sum(labels[x["candidate_id"]]["eval_label"]=="UNKNOWN" for x in lk)
        result["rules"][name]={"features":names,"per_feature_target_retention":per_feature_retention,
          "thresholds":ts,"strict_target_kept":target_kept,"strict_target_total":len(pos),
          "strict_target_retention":target_kept/len(pos),"lvis_absent_kept":neg_kept,"lvis_absent_total":13,
          "lvis_unknown_kept":unknown_kept,"lvis_unknown_total":unknown_total,
          "absent111_top5_kept":len(ak),"absent111_top5_total":len(absent),
          "absent111_queries_active":len({x["item_id"] for x in ak}),"absent111_top1_active":sum(x["native_rank"]==1 for x in ak)}
    result["limitations"]=["Risk budget is calibrated only on 164 strict IoU50 LVIS positives.",
      "Bonferroni controls worst-case marginal loss but does not establish population coverage.",
      "Unknown present proposals are reported and cannot be called false positives.",
      "Absent111 top-five is development transfer, not independent confirmation."]
    write_once(DEST/"evaluation.json",result);print(json.dumps(result["rules"]))


if __name__=="__main__":main()
