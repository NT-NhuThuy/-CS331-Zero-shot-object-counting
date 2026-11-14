"""Labels-free, resumable prediction from frozen candidate/reference/hypothesis caches."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .allocation import allocate, independent_nms
from .config import file_sha256, load_locked_config, stable_hash
from .hypothesis_builder import Hypothesis

FORBIDDEN_INFERENCE_KEYS = {"gt_count","gt_boxes","points_xy","query_status",
                            "presence_label","annotations","labels"}


def rows(path: str | Path) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def keyed(path: str | Path) -> dict[str,dict]:
    result = {}
    for row in rows(path):
        key = row["item_id"]
        if key in result:
            raise ValueError(f"duplicate item ID in {path}: {key}")
        if FORBIDDEN_INFERENCE_KEYS & row.keys():
            raise ValueError(f"label/status field in inference artifact {path}")
        result[key] = row
    return result


def _hypotheses(records: list[dict], candidate_ids: set[str], references: dict[str,dict],
                query: str) -> list[Hypothesis]:
    result = []
    for record in records:
        h = Hypothesis(**record)
        if h.candidate_id not in candidate_ids or h.reference_id not in references or \
                references[h.reference_id]["semantic_role"] != h.semantic_role or \
                references[h.reference_id].get("canonical_query",query) != query:
            raise ValueError("hypothesis candidate/reference provenance mismatch")
        result.append(h)
    return result


def _fallback(pool: dict, reason: str) -> dict:
    by_id = {c["candidate_id"]:c for c in pool["candidates"]}
    ids = pool["m0_final_ids"]
    return {"count":len(ids),"final_boxes":[{"candidate_id":cid,
             "box_xyxy_original":by_id[cid]["box_xyxy_original"],
             "reference_id":None,"semantic_role":"target","retained_support":{},
             "solver_status":"fallback_m0"} for cid in ids],
            "selected_distractors":[],"fallback_reason":reason,"solver":None}


def _decoded(pool: dict, chosen: list[Hypothesis], retained: dict,
             solver: dict | None = None) -> dict:
    by_id = {c["candidate_id"]:c for c in pool["candidates"]}
    target = [h for h in chosen if h.semantic_role == "target"]
    if len({h.candidate_id for h in chosen}) != len(chosen):
        raise ValueError("more than one hypothesis for one candidate")
    final = [{"candidate_id":h.candidate_id,"box_xyxy_original":by_id[h.candidate_id]["box_xyxy_original"],
              "reference_id":h.reference_id,"semantic_role":"target",
              "retained_support":retained.get(h.hypothesis_id,{}),
              "solver_status":solver["status"] if solver else "independent_control"}
             for h in target]
    return {"count":len(final),"final_boxes":final,
            "selected_distractors":[h.hypothesis_id for h in chosen if h.semantic_role == "competitor"],
            "fallback_reason":None,"solver":solver}


def predict_item(item: dict, pool: dict, evidence: dict,
                 references: dict[str,dict], config: dict, config_hash: str) -> dict:
    started = time.monotonic()
    if any(item[k] != pool[k] or item[k] != evidence[k]
           for k in ("item_id","source_image_id","query")):
        raise ValueError("manifest/pool/evidence identity mismatch")
    candidates = pool["candidates"]
    candidate_ids = [c["candidate_id"] for c in candidates]
    if len(set(candidate_ids)) != len(candidate_ids) or \
            not set(pool["m0_final_ids"]) <= set(candidate_ids):
        raise ValueError("duplicate/missing candidate or M0 ID")
    if evidence["feature_config_sha256"] != stable_hash(config["feature"]) or \
            evidence["numerical_config_sha256"] != stable_hash(config["numerical"]) or \
            evidence["reference_snapshot_sha256"] != config["gallery_sha256"]:
        raise ValueError("hypothesis/config/gallery hash mismatch")
    base = {"item_id":item["item_id"],"source_image_id":item["source_image_id"],
            "query":item["query"],"config_sha256":config_hash,
            "gallery_sha256":config["gallery_sha256"],"methods":{},
            "candidate_count":len(candidates),"m0_count":len(pool["m0_final_ids"]),
            "raw_pool_complete":bool(pool["raw_pool_complete"]),
            "unresolved_candidate_ids":evidence.get("unresolved_candidate_ids",[])}
    base["methods"]["M0"] = _fallback(pool,"native_control")
    reason = ("raw_pool_incomplete" if not pool["raw_pool_complete"] else
              evidence["status"] if evidence["status"] != "ok" else None)
    if reason:
        for method in ("R-global","R-match","R-allocate","R-match-gamma0","R-allocate-gamma0"):
            base["methods"][method] = _fallback(pool,reason)
        base["latency_seconds"] = time.monotonic()-started
        return base
    boxes = {c["candidate_id"]:tuple(c["box_xyxy_original"]) for c in candidates}
    numerical = config["numerical"]
    opening, retention = numerical["opening_cost"],numerical["retention_fraction"]
    groups = {"R-global":"global_hypotheses","R-match":"match_hypotheses",
              "R-match-gamma0":"gamma_zero_hypotheses"}
    decoded = {name:_hypotheses(evidence[key],set(candidate_ids),references,item["query"])
               for name,key in groups.items()}
    expected = {h.hypothesis_id for h in decoded["R-match"]}
    if expected != {h.hypothesis_id for h in decoded["R-global"]} or \
            expected != {h.hypothesis_id for h in decoded["R-match-gamma0"]}:
        raise ValueError("controls do not use identical candidate/reference alternatives")
    for method,hypotheses in decoded.items():
        chosen = independent_nms(hypotheses,boxes,opening,numerical["nms_iou"])
        base["methods"][method] = _decoded(pool,chosen,{})
    for method,source in (("R-allocate","R-match"),
                          ("R-allocate-gamma0","R-match-gamma0")):
        hypotheses = decoded[source]
        try:
            outcome = allocate(hypotheses,opening,retention,numerical["milp_relative_gap"])
        except (RuntimeError, ValueError) as error:
            base["methods"][method] = _fallback(pool,"milp_technical_error")
            base["methods"][method]["solver"] = {"status":"technical_error",
                                                    "message":str(error)}
            continue
        solver = {"status":outcome.status,"objective":outcome.objective,
                  "dual_bound":outcome.dual_bound,"relative_gap":outcome.relative_gap,
                  "node_count":outcome.node_count,"message":outcome.solver_message}
        if outcome.status != "optimal":
            base["methods"][method] = _fallback(pool,"milp_"+outcome.status)
            base["methods"][method]["solver"] = solver
        else:
            by_hid = {h.hypothesis_id:h for h in hypotheses}
            chosen = [by_hid[x] for x in outcome.selected_ids]
            for h in chosen:
                score = sum(h.atom_weights[p]*fraction for p,fraction in
                            outcome.retained_support[h.hypothesis_id].items())
                if score+1e-6 < retention*h.evidence_gain:
                    raise ArithmeticError("selected hypothesis lost required support")
            base["methods"][method] = _decoded(pool,chosen,outcome.retained_support,solver)
            if base["methods"][method]["count"] != outcome.target_count:
                raise ArithmeticError("MILP target count/box mismatch")
    base["latency_seconds"] = time.monotonic()-started
    return base


def freeze_predictions(manifest: Path, candidates: Path, hypotheses: Path,
                       gallery: Path, config: Path, predictions: Path, freeze: Path) -> dict:
    expected = keyed(manifest)
    actual = keyed(predictions)
    if set(expected) != set(actual):
        raise ValueError(f"prediction set incomplete: {len(actual)}/{len(expected)}")
    config_doc,config_hash = load_locked_config(config)
    for item_id,row in actual.items():
        if row["config_sha256"] != config_hash or row["gallery_sha256"] != config_doc["gallery_sha256"]:
            raise ValueError("prediction config/gallery mismatch")
        for method in row["methods"].values():
            if method["count"] != len(method["final_boxes"]):
                raise ValueError("prediction count/box mismatch")
    files = {str(path):file_sha256(path) for path in
             (manifest,candidates,hypotheses,gallery,config,predictions)}
    record = {"complete":True,"items":len(expected),"files":files,
              "labels_opened_before_freeze":False}
    if freeze.exists():
        existing = json.loads(freeze.read_text())
        if existing != record:
            raise ValueError("existing freeze differs from current complete artifacts")
        return existing
    with freeze.open("x") as stream:
        json.dump(record,stream,sort_keys=True,indent=2)
        stream.write("\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    for name in ("manifest","candidates","hypotheses","gallery","config","out","freeze"):
        parser.add_argument("--"+name,required=True,type=Path)
    args = parser.parse_args()
    config, config_hash = load_locked_config(args.config)
    if file_sha256(args.gallery) != config["gallery_sha256"]:
        raise ValueError("gallery snapshot changed")
    manifest,candidates,evidence = [keyed(path) for path in
                                    (args.manifest,args.candidates,args.hypotheses)]
    if set(manifest) != set(candidates) or set(manifest) != set(evidence):
        raise ValueError("manifest/candidate/hypothesis item set incomplete")
    reference_rows = rows(args.gallery)
    references = {r["reference_id"]:r for r in reference_rows}
    if len(references) != len(reference_rows):
        raise ValueError("duplicate reference ID")
    done = keyed(args.out) if args.out.exists() else {}
    if not set(done) <= set(manifest) or any(r["config_sha256"] != config_hash or
                                              r["gallery_sha256"] != config["gallery_sha256"]
                                              for r in done.values()):
        raise ValueError("resume output has unknown item or changed config")
    for item_id in manifest:
        if item_id in done:
            continue
        record = predict_item(manifest[item_id],candidates[item_id],evidence[item_id],
                              references,config,config_hash)
        with args.out.open("a") as stream:
            stream.write(json.dumps(record,sort_keys=True)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"completed":len(done)+1,"planned":len(manifest),
                          "item_id":item_id,"R-allocate_count":record["methods"]["R-allocate"]["count"]}),flush=True)
        done[item_id] = record
    freeze_predictions(args.manifest,args.candidates,args.hypotheses,
                       args.gallery,args.config,args.out,args.freeze)
    print(json.dumps({"complete":True,"items":len(done)}),flush=True)


if __name__ == "__main__":
    main()
