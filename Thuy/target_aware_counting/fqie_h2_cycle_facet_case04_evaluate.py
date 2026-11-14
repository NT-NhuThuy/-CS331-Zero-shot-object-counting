"""Exploratory post-freeze evaluation of Cycle unary facets on case_04."""
from __future__ import annotations
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"target_aware_counting/output"
RUN=OUT/"fqie_h2_cycle_facet_case04_20261104_v1";DEST=OUT/"fqie_h2_cycle_facet_case04_eval_20261104_v1"
FACET={0:"head_class",1:"color_attribute",2:"color_attribute",3:"spatial_relation",4:"spatial_relation",5:"color_attribute"}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    m=json.loads((RUN/"manifest.json").read_text());labels=json.loads((RUN/"eval_labels.json").read_text());f=json.loads((RUN/"freeze_yesno.json").read_text())
    lab={x["candidate_id"]:x for x in labels["items"]};dec={x["candidate_id"]:x["strict_decision"] for x in f["items"]};rows=[]
    for x in m["items"]:
        l=lab[x["candidate_id"]];actual=dec[x["candidate_id"]];rows.append({"probe_id":x["probe_id"],"region_index":l["region_index"],
          "query":l["query"],"facet":FACET[l["query_index"]],"expected":l["expected_decision"],"actual":actual,"correct":actual==l["expected_decision"]})
    by=defaultdict(list)
    for x in rows:by[x["facet"]].append(x)
    result={"status":"EXPLORATORY_ASSISTANT_LABELED_FACET_DIAGNOSTIC","overall":{"correct":sum(x["correct"] for x in rows),"total":len(rows)},
      "facets":{k:{"correct":sum(x["correct"] for x in v),"total":len(v),"errors":[x for x in v if not x["correct"]]} for k,v in by.items()},
      "rows":rows,"frozen_inputs":{"manifest_sha256":sha(RUN/"manifest.json"),"labels_sha256":sha(RUN/"eval_labels.json"),"unary_freeze_sha256":sha(RUN/"freeze_yesno.json")},
      "limitations":["Expected answers come from prior assistant visual interpretation, not independent annotation.","Only one image and three overlapping proposals are tested.",
        "Relation questions ask whether a region depicts the full relational referent; this does not prove relation grounding or subject-owner binding.",
        "Results are capability evidence only and cannot pass full H2."]}
    DEST.mkdir(parents=True,exist_ok=True);p=DEST/"evaluation.json";s=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if p.exists() and p.read_text()!=s:raise RuntimeError("locked evaluation differs")
    p.write_text(s);print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
if __name__=="__main__":main()
