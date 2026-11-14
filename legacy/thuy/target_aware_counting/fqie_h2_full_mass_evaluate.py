"""Calibrate matched two-text rules on LVIS and evaluate all absent111 boxes."""
from __future__ import annotations
import glob,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"target_aware_counting/output"
CAL=OUT/"fqie_h2_alpha_lvis_two_text_20261104_v1";LABELS=OUT/"fqie_h2_alpha_lvis_full_20261104_v1/eval_labels.json"
FULL=OUT/"fqie_h2_alpha_absent111_full_20261104_v1";DEST=OUT/"fqie_h2_full_mass_rule_20261104_v1"


def threshold(values,retention):
    v=sorted(values,reverse=True);return v[math.ceil(retention*len(v))-1]


def main():
    labels={x["candidate_id"]:x["eval_label"] for x in json.loads(LABELS.read_text())["items"]};cal=[]
    for p in glob.glob(str(CAL/"scores/*.json")):
        d=json.load(open(p));d["label"]=labels[d["candidate_id"]];cal.append(d)
    pos=[x for x in cal if x["label"]=="TARGET_IOU50"]
    rules={
      "raw95":{"raw":threshold([x["alpha_raw_full"] for x in pos],.95)},
      "generic95":{"margin":threshold([x["alpha_query_minus_generic_object"] for x in pos],.95)},
      "raw_generic_bonferroni95":{"raw":threshold([x["alpha_raw_full"] for x in pos],.975),
                                   "margin":threshold([x["alpha_query_minus_generic_object"] for x in pos],.975)}}
    full=[]
    for p in glob.glob(str(FULL/"scores/*.json")):
        d=json.load(open(p));full.append(d)
    if sum(len(x["candidates"]) for x in full)!=6942:raise RuntimeError("full denominator")
    result={"status":"DEVELOPMENT_FULL_COUNT_MASS_DIAGNOSTIC","calibration_positive_n":len(pos),"rules":{}}
    for name,t in rules.items():
        def keep(x):return ("raw" not in t or x["alpha_raw_full"]>=t["raw"]) and ("margin" not in t or x["alpha_query_minus_generic_object"]>=t["margin"])
        cal_kept=[x for x in cal if keep(x)];per=[]
        for q in full:
            kept=sum(keep(x) for x in q["candidates"]);per.append((q["item_id"],len(q["candidates"]),kept))
        total_kept=sum(x[2] for x in per);counts=[x[2] for x in per]
        result["rules"][name]={"thresholds":t,
          "lvis_strict_target_kept":sum(x["label"]=="TARGET_IOU50" for x in cal_kept),"lvis_strict_target_total":len(pos),
          "lvis_absent_kept":sum(x["label"]=="VERIFIED_ABSENT" for x in cal_kept),"lvis_absent_total":13,
          "absent111_boxes_kept":total_kept,"absent111_boxes_total":6942,
          "absent111_box_mass_reduction":1-total_kept/6942,"absent111_queries_active":sum(x>0 for x in counts),
          "absent111_queries_zeroed":sum(x==0 for x in counts),"retained_count_median":statistics.median(counts),
          "retained_count_max":max(counts)}
    result["limitations"]=["All thresholds are calibrated on 164 LVIS strict positives; absent111 is development transfer.",
      "All 6942 frozen absent boxes are scored, so count mass is complete for this M0 cohort.",
      "Strict-positive retention does not label 129 poorly localized present-query proposals.",
      "No end-to-end present count metric or independent confirmation is included."]
    DEST.mkdir(parents=True,exist_ok=True);out=DEST/"evaluation.json";payload=json.dumps(result,indent=2)+"\n"
    if out.exists() and out.read_text()!=payload:raise RuntimeError("evaluation differs")
    out.write_text(payload);print(json.dumps(result["rules"]))


if __name__=="__main__":main()
