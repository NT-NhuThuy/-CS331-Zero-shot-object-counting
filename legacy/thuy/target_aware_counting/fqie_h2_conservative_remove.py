"""Conservative semantic remove-only rule diagnostics from frozen evidence."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"target_aware_counting/output"
LEDGER=OUT/"fqie_h2_evidence_ledger_20261104_v1"
LABELS=OUT/"fqie_h2_alpha_lvis_full_20261104_v1/eval_labels.json"
DEST=OUT/"fqie_h2_conservative_remove_20261104_v1"


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write_once(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);data=(json.dumps(value,indent=2)+"\n").encode()
    if p.exists() and p.read_bytes()!=data:raise RuntimeError(f"locked output differs: {p}")
    p.write_bytes(data)


FEATURES={"native":"native_score","raw_alpha":"alpha_raw_full","generic_margin":"alpha_query_minus_generic_object"}
RULES={"no_op":[],"native_100":["native"],"raw_alpha_100":["raw_alpha"],
       "generic_margin_100":["generic_margin"],"raw_and_generic_100":["raw_alpha","generic_margin"],
       "native_raw_generic_100":["native","raw_alpha","generic_margin"]}


def prepare():
    lvis=rows(LEDGER/"lvis_query_switch_evidence.jsonl");labels={x["candidate_id"]:x for x in json.loads(LABELS.read_text())["items"]}
    positives=[x for x in lvis if labels[x["candidate_id"]]["eval_label"]=="TARGET_IOU50"]
    if len(positives)!=164:raise RuntimeError("strict-positive denominator changed")
    thresholds={name:min(x["features"][field] for x in positives) for name,field in FEATURES.items()}
    config={"status":"LOCKED_COMPONENT_DEVELOPMENT_RULES","component":"semantic_remove_only",
      "calibration":"minimum feature value across 164 official-IoU50 LVIS strict positives",
      "thresholds":thresholds,"rules":RULES,
      "keep_semantics":"candidate kept iff every named feature is >= its threshold; empty rule is no-op",
      "invalid_policy":"missing/nonfinite feature => UNKNOWN and keep (never remove)",
      "tie_policy":"score equal to threshold is kept","output_policy":"remove-only; boxes and IDs otherwise unchanged",
      "input_sha256":{"feature_contract":sha(LEDGER/"feature_contract.json"),"lvis_ledger":sha(LEDGER/"lvis_query_switch_evidence.jsonl"),
                      "labels":sha(LABELS)},"warning":"development component diagnostic; absent111 not read during threshold creation"}
    write_once(DEST/"locked_rules.json",config);print(json.dumps(thresholds))


def keep(row,names,thresholds):
    try:return all(float(row["features"][FEATURES[n]])>=thresholds[n] for n in names)
    except (KeyError,TypeError,ValueError):return True


def evaluate():
    config=json.loads((DEST/"locked_rules.json").read_text());lvis=rows(LEDGER/"lvis_query_switch_evidence.jsonl")
    absent=rows(LEDGER/"absent111_top5_evidence.jsonl");labels={x["candidate_id"]:x for x in json.loads(LABELS.read_text())["items"]}
    result={"status":"DEVELOPMENT_COMPONENT_DIAGNOSTIC","rules":{}}
    for name,names in config["rules"].items():
        lkeep=[x for x in lvis if keep(x,names,config["thresholds"])]
        akeep=[x for x in absent if keep(x,names,config["thresholds"])]
        target_total=sum(x["eval_label"]=="TARGET_IOU50" for x in labels.values())
        target_kept=sum(labels[x["candidate_id"]]["eval_label"]=="TARGET_IOU50" for x in lkeep)
        lvis_abs_total=sum(x["eval_label"]=="VERIFIED_ABSENT" for x in labels.values())
        lvis_abs_kept=sum(labels[x["candidate_id"]]["eval_label"]=="VERIFIED_ABSENT" for x in lkeep)
        result["rules"][name]={"features":names,"lvis_strict_target_kept":target_kept,"lvis_strict_target_total":target_total,
          "lvis_strict_target_retention":target_kept/target_total,"lvis_absent_kept":lvis_abs_kept,"lvis_absent_total":lvis_abs_total,
          "absent111_top5_kept":len(akeep),"absent111_top5_total":len(absent),
          "absent111_queries_active":len({x["item_id"] for x in akeep}),
          "absent111_top1_active":sum(x["native_rank"]==1 for x in akeep)}
    result["limitations"]=["Thresholds preserve the 164 strict IoU50 LVIS positives by construction.",
      "Present-query proposals below IoU50 are UNKNOWN; their removals are not counted as correct or incorrect.",
      "Absent111 is development transfer and top-five only, not full count mass or independent confirmation.",
      "No rule is promoted by this evaluator; all fixed controls are reported."]
    write_once(DEST/"evaluation.json",result);print(json.dumps(result["rules"]))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("prepare","evaluate"));a=p.parse_args().action
    {"prepare":prepare,"evaluate":evaluate}[a]()
