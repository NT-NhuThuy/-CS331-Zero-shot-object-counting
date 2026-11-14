"""Evaluate frozen CycleGRPO unary safety on absent111 native-top proposals."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"target_aware_counting/output"
RUN=OUT/"fqie_h2_absent111_cycle_top1_20261104_v1"
SCORES=OUT/"fqie_h2_alpha_absent111_full_20261104_v1/scores"
RULE=OUT/"fqie_h2_full_mass_rule_20261104_v1/evaluation.json"
DEST=OUT/"fqie_h2_absent111_cycle_top1_eval_20261104_v1"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-h,c+h]
def main():
    manifest=json.loads((RUN/"manifest.json").read_text());frozen=json.loads((RUN/"freeze_yesno.json").read_text())
    decisions={x["candidate_id"]:x["strict_decision"] for x in frozen["items"]};t=json.loads(RULE.read_text())["rules"]["raw_generic_bonferroni95"]["thresholds"]
    rows=[]
    for x in manifest["items"]:
        q=json.loads((SCORES/(x["item_id"].split(":",1)[1]+".json")).read_text())
        s=next(y for y in q["candidates"] if y["candidate_id"]==x["candidate_id"])
        alpha=s["alpha_raw_full"]>=t["raw"] and s["alpha_query_minus_generic_object"]>=t["margin"]
        cycle=decisions[x["candidate_id"]]
        rows.append({"probe_id":x["probe_id"],"item_id":x["item_id"],"candidate_id":x["candidate_id"],
          "query":x["query_for_later_audit_only"],"alpha_keep":alpha,"cycle_decision":cycle,
          "cascade_keep":alpha or cycle=="yes","native_score":x["native_score"]})
    yes=sum(x["cycle_decision"]=="yes" for x in rows);ak=sum(x["alpha_keep"] for x in rows);ck=sum(x["cascade_keep"] for x in rows)
    result={"status":"DEVELOPMENT_VERIFIED_ABSENT_TOP1_SAFETY_DIAGNOSTIC","denominator":len(rows),
      "cycle_yes":yes,"cycle_no":sum(x["cycle_decision"]=="no" for x in rows),
      "cycle_false_activation_rate":yes/len(rows),"cycle_false_activation_wilson95":wilson(yes,len(rows)),
      "alpha_top1_active":ak,"alpha_plus_cycle_rescue_top1_active":ck,
      "cycle_yes_items":[x for x in rows if x["cycle_decision"]=="yes"],
      "frozen_inputs":{"manifest_sha256":sha(RUN/"manifest.json"),"unary_freeze_sha256":sha(RUN/"freeze_yesno.json")},
      "limitations":["One native-top M0 proposal per query; this is not full 6,942-box mass.",
        "FSC absent111 is development transfer, not TEST or independent confirmation.",
        "Wilson interval quantifies binomial sampling only and assumes items are exchangeable.",
        "A unary decision does not address duplicate, part, or relation identity."]}
    DEST.mkdir(parents=True,exist_ok=True);p=DEST/"evaluation.json";s=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if p.exists() and p.read_text()!=s:raise RuntimeError("locked evaluation differs")
    p.write_text(s);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
