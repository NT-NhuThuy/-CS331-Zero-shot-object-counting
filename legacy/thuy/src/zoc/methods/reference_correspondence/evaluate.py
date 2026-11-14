"""Paired development evaluation; label files are opened only after freeze verification."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter,defaultdict
from pathlib import Path

from .config import file_sha256
from .predict import rows

METHODS = ("M0","R-global","R-match","R-allocate",
           "R-match-gamma0","R-allocate-gamma0")


def verify_freeze(path: Path, predictions: Path, manifest: Path) -> dict:
    frozen = json.loads(path.read_text())
    if not frozen.get("complete") or frozen.get("labels_opened_before_freeze") is not False:
        raise ValueError("predictions were not safely frozen")
    for name,digest in frozen["files"].items():
        if file_sha256(name) != digest:
            raise ValueError(f"frozen input changed: {name}")
    if str(predictions) not in frozen["files"] or str(manifest) not in frozen["files"]:
        raise ValueError("requested predictions/manifest are not the frozen files")
    return frozen


def _iou(a,b):
    dx = max(0.,min(a[2],b[2])-max(a[0],b[0]))
    dy = max(0.,min(a[3],b[3])-max(a[1],b[1]))
    intersection = dx*dy
    area = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/area if area > 0 else 0.


def matched_count(predicted,truth,point=False):
    edges = [[j for j,target in enumerate(truth) if
              ((box[0] <= target[0] <= box[2] and box[1] <= target[1] <= box[3])
               if point else _iou(box,target) >= .5)]
             for box in predicted]
    owners = {}
    def augment(i,seen):
        for j in edges[i]:
            if j not in seen:
                seen.add(j)
                if j not in owners or augment(owners[j],seen):
                    owners[j] = i
                    return True
        return False
    for i in range(len(predicted)):
        augment(i,set())
    return len(owners)


def load_labels(cohort: str, label_root: Path | None = None,
                absent_manifest: Path | None = None) -> dict[str,dict]:
    if cohort == "fsc_verified_absent111":
        if absent_manifest is None:
            raise ValueError("--absent-manifest is required for fsc_verified_absent111")
        if not absent_manifest.is_file():
            raise FileNotFoundError(f"verified-absent manifest not found: {absent_manifest}")
        with absent_manifest.open(encoding="utf-8-sig", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        if len(csv_rows) != 111:
            raise ValueError(f"verified-absent manifest must contain exactly 111 rows, got {len(csv_rows)}")
        result = {}
        seen_ids = set()
        for row in csv_rows:
            pair_id = row["pair_id"]
            if pair_id in seen_ids:
                raise ValueError(f"duplicate pair_id in verified-absent manifest: {pair_id}")
            seen_ids.add(pair_id)
            if int(row["presence_label"]) != 0 or int(row["count_target"]) != 0 or \
                    row["final_decision"] != "KEEP_NEGATIVE" or row["benchmark_inclusion"] != "YES":
                raise ValueError("111-pair verified-absent protocol mismatch")
            result[f"fsc_verified_absent:{pair_id}"] = row
        return result
    if label_root is None or not label_root.is_dir():
        raise ValueError("--label-root is required for non-absent cohorts")
    if cohort == "lvis_dev300":
        path = label_root / "new_semantic_failure_cohort_dev_20261025_v1/cohort/labels_posthoc.jsonl"
        return {f"lvis_dev:{r['query_id']}":r for r in rows(path)}
    if cohort == "fsc_primary_dev100":
        path = label_root / "fsc_tile_candidate_coverage_val_20261019_v1/input/target_points_posthoc.jsonl"
        return {f"fsc_val_primary:{r['query_image_id']}":r for r in rows(path)}
    raise ValueError("unknown evaluation cohort")


def truth_for(cohort: str, item: dict, labels: dict[str,dict]):
    label = labels[item["item_id"]]
    if cohort == "lvis_dev300":
        status = label["status"]
        if status not in ("present","verified_absent"):
            raise ValueError("LVIS unknown/assumed absent cannot be scored as zero")
        if str(label["source_image_id"]) != str(item["source_image_id"]):
            raise ValueError("LVIS label/source mismatch")
        gt = [[x,y,x+w,y+h] for x,y,w,h in label["gt_boxes_xywh"]] if status == "present" else []
        if int(label["gt_count"]) != len(gt):
            raise ValueError("LVIS label count/boxes mismatch")
    elif cohort == "fsc_primary_dev100":
        status,gt = "present",label["points_xy"]
        if str(label["source_image_id"]) != str(item["source_image_id"]):
            raise ValueError("FSC label/source mismatch")
    else:
        status,gt = "verified_absent",[]
    return status,gt


def validate_cohort_ids(input_rows: list[dict], labels: dict[str, dict]) -> None:
    """Require the evaluated manifest and label manifest to be the same cohort."""
    input_ids = {r["item_id"] for r in input_rows}
    label_ids = set(labels)
    if input_ids != label_ids:
        missing = sorted(input_ids - label_ids)
        extra = sorted(label_ids - input_ids)
        raise ValueError(
            "label/input cohort mismatch: "
            f"missing_labels={missing[:5]}, extra_labels={extra[:5]}"
        )


def score(method: dict, truth: list, status: str, point: bool) -> dict:
    predicted = [box["box_xyxy_original"] for box in method["final_boxes"]]
    if method["count"] != len(predicted) or \
            len({box["candidate_id"] for box in method["final_boxes"]}) != len(predicted):
        raise ValueError("count/final IDs mismatch")
    tp = matched_count(predicted,truth,point) if status == "present" else 0
    count = len(predicted)
    return {"count":count,"absolute_error":abs(count-len(truth)),
            "squared_error":(count-len(truth))**2,"exact":int(count==len(truth)),
            "tp":tp,"fp":count-tp,"fn":len(truth)-tp,
            "false_zero":int(status=="present" and count==0),
            "fallback":int(method["fallback_reason"] is not None and method["fallback_reason"]!="native_control")}


def source_bootstrap_delta(rows_by_item: list[dict], method: str, control: str,
                           repetitions: int = 2000, seed: int = 42):
    grouped = defaultdict(list)
    for row in rows_by_item:
        grouped[str(row["source_image_id"])].append(row)
    clusters = list(grouped.values())
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        draw = [item for _ in clusters for item in clusters[rng.randrange(len(clusters))]]
        values.append(sum(r["methods"][method]["absolute_error"]-
                          r["methods"][control]["absolute_error"] for r in draw)/len(draw))
    values.sort()
    return [values[int(.025*repetitions)],values[int(.975*repetitions)-1]]


def evaluate(cohort: str, manifest: Path, predictions: Path, freeze: Path,
             comparison: Path, summary_path: Path, label_root: Path | None = None,
             absent_manifest: Path | None = None):
    verify_freeze(freeze,predictions,manifest)
    input_rows,prediction_rows = rows(manifest),rows(predictions)
    if len(input_rows) != len(prediction_rows) or \
            {r["item_id"] for r in input_rows} != {r["item_id"] for r in prediction_rows}:
        raise ValueError("prediction/manifest denominator mismatch")
    by_prediction = {r["item_id"]:r for r in prediction_rows}
    if len(by_prediction) != len(prediction_rows):
        raise ValueError("duplicate prediction item")
    labels = load_labels(cohort, label_root, absent_manifest)  # after freeze checks
    validate_cohort_ids(input_rows, labels)
    scored = []
    for item in input_rows:
        pred = by_prediction[item["item_id"]]
        if str(pred["source_image_id"]) != str(item["source_image_id"]) or pred["query"] != item["query"]:
            raise ValueError("prediction/manifest identity mismatch")
        status,truth = truth_for(cohort,item,labels)
        if set(pred["methods"]) != set(METHODS):
            raise ValueError("method table missing a frozen control")
        scored.append({"item_id":item["item_id"],"source_image_id":item["source_image_id"],
                       "status":status,"gt_count":len(truth),
                       "methods":{name:score(pred["methods"][name],truth,status,
                                              point=cohort=="fsc_primary_dev100") for name in METHODS}})
    if comparison.exists() or summary_path.exists():
        raise FileExistsError("evaluation output already exists")
    fields = ["cohort","item_id","source_image_id","status","gt_count"] + [
        f"{method}_{metric}" for method in METHODS for metric in
        ("count","absolute_error","exact","tp","fp","fn","false_zero","fallback")]
    with comparison.open("x",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in scored:
            flat = {"cohort":cohort,**{k:row[k] for k in fields[1:5]}}
            for method in METHODS:
                flat.update({f"{method}_{metric}":value for metric,value in row["methods"][method].items()
                             if f"{method}_{metric}" in fields})
            writer.writerow(flat)
    result = {"cohort":cohort,"n":len(scored),"sources":len({r["source_image_id"] for r in scored}),
              "status_counts":dict(Counter(r["status"] for r in scored)),"methods":{}}
    for method in METHODS:
        metric = [r["methods"][method] for r in scored]
        absent = [r for r in scored if r["status"] == "verified_absent"]
        totals = {"mae":sum(v["absolute_error"] for v in metric)/len(metric),
                  "rmse":math.sqrt(sum(v["squared_error"] for v in metric)/len(metric)),
                  "exact":sum(v["exact"] for v in metric),
                  "tp":sum(v["tp"] for v in metric),"fp":sum(v["fp"] for v in metric),
                  "fn":sum(v["fn"] for v in metric),
                  "false_zero":sum(v["false_zero"] for v in metric),
                  "fallback":sum(v["fallback"] for v in metric),
                  "absent_positive":sum(r["methods"][method]["count"]>0 for r in absent),
                  "absent_countmass":sum(r["methods"][method]["count"] for r in absent),
                  "absent_new":sum(r["methods"]["M0"]["count"]==0 and
                                    r["methods"][method]["count"]>0 for r in absent),
                  "absent_repaired":sum(r["methods"]["M0"]["count"]>0 and
                                         r["methods"][method]["count"]==0 for r in absent)}
        if method != "M0":
            totals["mae_delta_vs_m0_ci95_source_bootstrap"] = source_bootstrap_delta(scored,method,"M0")
        result["methods"][method] = totals
    summary_path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort",choices=("lvis_dev300","fsc_primary_dev100","fsc_verified_absent111"),required=True)
    for name in ("manifest","predictions","freeze","comparison","summary"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--label-root", type=Path,
                        help="Root containing archived label artifacts for present cohorts")
    parser.add_argument("--absent-manifest", type=Path,
                        help="Explicit verified-absent CSV; required for fsc_verified_absent111")
    args = parser.parse_args()
    summary = evaluate(args.cohort,args.manifest,args.predictions,args.freeze,
                       args.comparison,args.summary,args.label_root,args.absent_manifest)
    print(json.dumps({"cohort":args.cohort,"n":summary["n"],"sources":summary["sources"]}))


if __name__ == "__main__":
    main()
