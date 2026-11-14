"""Post-freeze paired query-switch evaluation on identical LVIS regions."""
from __future__ import annotations
import hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"target_aware_counting/output"
RUN=OUT/"fqie_h2_lvis_paired_region_cycle_20261104_v1";DEST=OUT/"fqie_h2_lvis_paired_region_cycle_eval_20261104_v1"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-h,c+h]
def main():
    manifest=json.loads((RUN/"manifest.json").read_text());labels=json.loads((RUN/"eval_labels.json").read_text())
    frozen=json.loads((RUN/"freeze_yesno.json").read_text());decision={x["candidate_id"]:x["strict_decision"] for x in frozen["items"]}
    label={x["candidate_id"]:x for x in labels["items"]};pairs=defaultdict(dict);rows=[]
    for x in manifest["items"]:
        l=label[x["candidate_id"]];d=decision[x["candidate_id"]];kind="target" if l["eval_label"]=="TARGET_REGION" else "absent"
        row={"probe_id":x["probe_id"],"candidate_id":x["candidate_id"],"pair_id":x["pair_id"],"source_image_id":x["source_image_id"],
          "kind":kind,"strict_decision":d,"query":x["query_for_later_audit_only"]};rows.append(row);pairs[x["pair_id"]][kind]=row
    if len(pairs)!=100 or any(set(v)!={"target","absent"} for v in pairs.values()):raise RuntimeError("pair contract mismatch")
    outcomes=Counter();detail=[]
    for pid,x in sorted(pairs.items()):
        py=x["target"]["strict_decision"]=="yes";ny=x["absent"]["strict_decision"]=="yes"
        state="TARGET_YES_ABSENT_NO" if py and not ny else "ALWAYS_YES" if py and ny else "ALWAYS_NO" if not py and not ny else "REVERSED"
        outcomes[state]+=1;detail.append({"pair_id":pid,"source_image_id":x["target"]["source_image_id"],"target_query":x["target"]["query"],
          "absent_query":x["absent"]["query"],"target_decision":x["target"]["strict_decision"],"absent_decision":x["absent"]["strict_decision"],"state":state})
    target_yes=sum(x["strict_decision"]=="yes" for x in rows if x["kind"]=="target");absent_yes=sum(x["strict_decision"]=="yes" for x in rows if x["kind"]=="absent")
    result={"status":"DEVELOPMENT_IDENTICAL_REGION_QUERY_SWITCH_DIAGNOSTIC","denominators":{"pairs":100,"calls":200},
      "target_yes":target_yes,"target_sensitivity":target_yes/100,"target_sensitivity_wilson95":wilson(target_yes,100),
      "absent_yes":absent_yes,"absent_false_activation":absent_yes/100,"absent_false_activation_wilson95":wilson(absent_yes,100),
      "paired_outcomes":dict(outcomes),"paired_correct_rate":outcomes["TARGET_YES_ABSENT_NO"]/100,
      "paired_correct_wilson95":wilson(outcomes["TARGET_YES_ABSENT_NO"],100),"pair_details":detail,
      "frozen_inputs":{"manifest_sha256":sha(RUN/"manifest.json"),"labels_sha256":sha(RUN/"eval_labels.json"),"unary_freeze_sha256":sha(RUN/"freeze_yesno.json")},
      "limitations":["LVIS VAL development cohort, not TEST or external confirmation.","One largest annotated object is selected per source image.",
        "The absent query is image-level verified absent; the target query matches the selected official GT object.",
        "Unary class discrimination does not establish full-query attribute/relation behavior, duplicate identity, or counting correctness."]}
    DEST.mkdir(parents=True,exist_ok=True);p=DEST/"evaluation.json";s=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if p.exists() and p.read_text()!=s:raise RuntimeError("locked evaluation differs")
    p.write_text(s);print(json.dumps({k:v for k,v in result.items() if k!="pair_details"},indent=2))
if __name__=="__main__":main()
