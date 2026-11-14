"""Post-freeze CycleGRPO unary safety evaluation on absent111 native top-five."""
from __future__ import annotations
import hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"target_aware_counting/output"
RUN=OUT/"fqie_h2_absent111_cycle_top5_20261104_v1"
LEDGER=OUT/"fqie_h2_evidence_ledger_20261104_v1/absent111_top5_evidence.jsonl"
RULE=OUT/"fqie_h2_full_mass_rule_20261104_v1/evaluation.json"
DEST=OUT/"fqie_h2_absent111_cycle_top5_eval_20261104_v2"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wilson(k,n,z=1.959963984540054):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-h,c+h]
def summarize(rows,key):
    kept=[x for x in rows if x[key]];by=defaultdict(int)
    for x in kept:by[x["item_id"]]+=1
    return {"boxes_kept":len(kept),"boxes_total":len(rows),"box_activation_rate":len(kept)/len(rows),
      "queries_active":len(by),"queries_total":len({x['item_id'] for x in rows}),
      "top1_active":sum(x[key] and x["native_rank"]==1 for x in rows),
      "retained_count_median":sorted(by.get(q,0) for q in {x['item_id'] for x in rows})[len({x['item_id'] for x in rows})//2],
      "retained_count_max":max(by.values(),default=0)}
def main():
    m=json.loads((RUN/"manifest.json").read_text());f=json.loads((RUN/"freeze_yesno.json").read_text())
    decision={x["candidate_id"]:x["strict_decision"] for x in f["items"]}
    evidence={x["candidate_id"]:x for x in (json.loads(s) for s in LEDGER.read_text().splitlines() if s.strip())}
    t=json.loads(RULE.read_text())["rules"]["raw_generic_bonferroni95"]["thresholds"];rows=[]
    for x in m["items"]:
        e=evidence[x["candidate_id"]];alpha=e["features"]["alpha_raw_full"]>=t["raw"] and e["features"]["alpha_query_minus_generic_object"]>=t["margin"]
        cycle=decision[x["candidate_id"]]=="yes"
        rows.append({"probe_id":x["probe_id"],"candidate_id":x["candidate_id"],"item_id":x["item_id"],
          "source_image_id":x["item_id"].split("__")[1],"query":x["query_for_later_audit_only"],"native_rank":x["native_rank"],"alpha":alpha,"cycle":cycle,
          "alpha_and_cycle":alpha and cycle,"alpha_or_cycle":alpha or cycle})
    yes=[x for x in rows if x["cycle"]];rank=Counter(x["native_rank"] for x in yes)
    active_queries={x["item_id"] for x in yes};active_sources={x["source_image_id"] for x in yes}
    result={"status":"DEVELOPMENT_VERIFIED_ABSENT_TOP5_SAFETY_DIAGNOSTIC","denominators":{"boxes":len(rows),"queries":len({x['item_id'] for x in rows})},
      "cycle_false_activation_wilson95":wilson(len(yes),len(rows)),"cycle_yes_rank_counts":dict(sorted(rank.items())),
      "cycle_query_activation_rate":len(active_queries)/111,"cycle_active_query_wilson95":wilson(len(active_queries),111),
      "cycle_active_sources":len(active_sources),"cycle_yes_boxes_on_most_active_source":max(Counter(x["source_image_id"] for x in yes).values(),default=0),
      "rules":{k:summarize(rows,k) for k in ("alpha","cycle","alpha_and_cycle","alpha_or_cycle")},
      "cycle_yes_items":[x for x in rows if x["cycle"]],
      "frozen_inputs":{"manifest_sha256":sha(RUN/"manifest.json"),"unary_freeze_sha256":sha(RUN/"freeze_yesno.json"),"evidence_ledger_sha256":sha(LEDGER)},
      "limitations":["Only native top-five proposals are tested; 6,390 lower-ranked absent proposals remain unverified.",
        "FSC absent111 is development transfer, not TEST or independent confirmation.","Target sensitivity comes from a separate LVIS strict-IoU50 cohort.",
        "Binary answers have no calibrated confidence and do not resolve duplicate, part, relation, or identity."]}
    DEST.mkdir(parents=True,exist_ok=True);p=DEST/"evaluation.json";s=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if p.exists() and p.read_text()!=s:raise RuntimeError("locked evaluation differs")
    p.write_text(s);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
