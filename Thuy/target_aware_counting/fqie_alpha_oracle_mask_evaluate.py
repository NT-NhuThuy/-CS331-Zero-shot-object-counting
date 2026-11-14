"""Evaluate oracle-mask target thresholds against frozen rectangle negatives."""
from __future__ import annotations
import glob,json,math,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ORACLE=ROOT/"target_aware_counting/output/fqie_h2_alpha_oracle_mask_headroom_20261104_v1"
LVIS=ROOT/"target_aware_counting/output/fqie_h2_alpha_lvis_full_20261104_v1"
ABS=ROOT/"target_aware_counting/output/fqie_h2_alpha_absent111_top5_20261104_v1"


def main():
    labels={x["candidate_id"]:x["eval_label"] for x in json.loads((LVIS/"eval_labels.json").read_text())["items"]}
    rect={}
    for p in glob.glob(str(LVIS/"scores/*.json")):
        d=json.load(open(p));rect[d["candidate_id"]]=d["views"]["full"]["target_cosine"]
    oracle=[]
    for p in glob.glob(str(ORACLE/"scores/*.json")):
        d=json.load(open(p));oracle.append((d["candidate_id"],d["oracle_mask_cosine"]))
    if len(oracle)!=164 or not all(labels[c]=="TARGET_IOU50" for c,_ in oracle):raise RuntimeError("oracle binding")
    lvis_neg=[rect[c] for c,l in labels.items() if l=="VERIFIED_ABSENT"]
    abs_rows=[json.load(open(p)) for p in glob.glob(str(ABS/"scores/*.json"))]
    values=sorted((v for _,v in oracle),reverse=True);points=[]
    for retention in (1.0,.95,.9):
        threshold=values[math.ceil(retention*len(values))-1]
        kept=[r for r in abs_rows if r["alpha_full"]>=threshold]
        points.append({"target_retention":retention,"oracle_positive_threshold":threshold,
            "LVIS_rectangle_absent_kept":sum(x>=threshold for x in lvis_neg),"LVIS_absent_total":len(lvis_neg),
            "absent111_top5_kept":len(kept),"absent111_top5_total":len(abs_rows),
            "absent111_queries_active":len({r["item_id"] for r in kept})})
    delta=[v-rect[c] for c,v in oracle]
    result={"status":"ORACLE_MASK_HEADROOM_ONLY","paired_target_delta":{
        "mean":statistics.mean(delta),"median":statistics.median(delta),"improved_fraction":sum(x>0 for x in delta)/len(delta),
        "worsened_fraction":sum(x<0 for x in delta)/len(delta)},"operating_points":points,
        "limitations":["Official GT polygons are unavailable at inference.",
            "Negatives retain rectangle masks, so this estimates only positive-mask headroom.",
            "A deployable SAM mask must be evaluated separately and may contain segmentation errors."]}
    payload=json.dumps(result,indent=2)+"\n";out=ORACLE/"evaluation.json"
    if out.exists() and out.read_text()!=payload:raise RuntimeError("evaluation differs")
    out.write_text(payload);print(json.dumps(result))


if __name__=="__main__":main()
