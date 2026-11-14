"""Joint frozen target/absent diagnostic for CycleGRPO unary evidence."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"target_aware_counting/output"
TARGET=OUT/"fqie_h2_target164_cycle_unary_20261104_v1";ABS=OUT/"fqie_h2_absent111_cycle_top1_20261104_v1"
ALPHA=OUT/"fqie_h2_alpha_lvis_two_text_20261104_v1/scores";ABS_SCORE=OUT/"fqie_h2_alpha_absent111_full_20261104_v1/scores"
RULE=OUT/"fqie_h2_full_mass_rule_20261104_v1/evaluation.json";DEST=OUT/"fqie_h2_cycle_unary_discrimination_20261104_v1"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-h,c+h]
def main():
    tm=json.loads((TARGET/"manifest.json").read_text());td={x["candidate_id"]:x["strict_decision"] for x in json.loads((TARGET/"freeze_yesno.json").read_text())["items"]}
    am=json.loads((ABS/"manifest.json").read_text());ad={x["candidate_id"]:x["strict_decision"] for x in json.loads((ABS/"freeze_yesno.json").read_text())["items"]}
    t=json.loads(RULE.read_text())["rules"]["raw_generic_bonferroni95"]["thresholds"]
    def akeep(s):return s["alpha_raw_full"]>=t["raw"] and s["alpha_query_minus_generic_object"]>=t["margin"]
    target=[]
    for x in tm["items"]:
        s=json.loads((ALPHA/(x["candidate_id"].replace(":","_")+".json")).read_text())
        target.append((akeep(s),td[x["candidate_id"]]=="yes",x["candidate_id"]))
    absent=[]
    for x in am["items"]:
        q=json.loads((ABS_SCORE/(x["item_id"].split(":",1)[1]+".json")).read_text());s=next(y for y in q["candidates"] if y["candidate_id"]==x["candidate_id"])
        absent.append((akeep(s),ad[x["candidate_id"]]=="yes",x["candidate_id"]))
    def counts(rows):return {"n":len(rows),"alpha_yes":sum(a for a,c,_ in rows),"cycle_yes":sum(c for a,c,_ in rows),
      "alpha_and_cycle_yes":sum(a and c for a,c,_ in rows),"alpha_or_cycle_yes":sum(a or c for a,c,_ in rows)}
    tc,ac=counts(target),counts(absent)
    result={"status":"DEVELOPMENT_UNARY_DISCRIMINATION_DIAGNOSTIC","target":tc,"verified_absent_top1":ac,
      "cycle_target_sensitivity":tc["cycle_yes"]/tc["n"],"cycle_target_sensitivity_wilson95":wilson(tc["cycle_yes"],tc["n"]),
      "cycle_absent_false_activation":ac["cycle_yes"]/ac["n"],"cycle_absent_false_activation_wilson95":wilson(ac["cycle_yes"],ac["n"]),
      "frozen_inputs":{"target_manifest_sha256":sha(TARGET/"manifest.json"),"target_unary_sha256":sha(TARGET/"freeze_yesno.json"),
        "absent_manifest_sha256":sha(ABS/"manifest.json"),"absent_unary_sha256":sha(ABS/"freeze_yesno.json")},
      "limitations":["Target proposals are strict IoU50 LVIS development instances; absent proposals are native-top FSC development transfer.",
        "Cohorts differ by dataset and proposal selection, so this is not a balanced classifier benchmark.",
        "Only one of 6,942 absent proposals per query is tested; full absent count mass is unknown.",
        "Binary answers contain no calibrated confidence and do not solve relation or identity."]}
    DEST.mkdir(parents=True,exist_ok=True);p=DEST/"evaluation.json";s=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if p.exists() and p.read_text()!=s:raise RuntimeError("locked evaluation differs")
    p.write_text(s);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
