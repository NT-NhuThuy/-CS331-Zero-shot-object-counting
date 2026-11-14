"""Build decision-free H2 evidence ledgers from frozen native/Alpha outputs."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"target_aware_counting/output"
DEST=OUT/"fqie_h2_evidence_ledger_20261104_v1"


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_once(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.read_bytes()!=data:raise RuntimeError(f"locked output differs: {p}")
    p.write_bytes(data)
def jsonl(rows):return "".join(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n" for x in rows).encode()


def build_lvis():
    base=OUT/"fqie_h2_alpha_lvis_full_20261104_v1";text=OUT/"fqie_h2_alpha_text_controls_20261104_v1"
    manifest=json.loads((base/"inference_manifest.json").read_text());bf=json.loads((base/"freeze_scores.json").read_text())
    tf=json.loads((text/"freeze_scores.json").read_text());rows=[]
    for x in manifest["items"]:
        rel="scores/"+x["candidate_id"].replace(":","_")+".json";bp=base/rel;tp=text/rel
        if sha(bp)!=bf["score_file_hashes"][rel] or sha(tp)!=tf["score_file_hashes"][rel]:raise RuntimeError("score freeze mismatch")
        b=json.loads(bp.read_text());t=json.loads(tp.read_text());s=t["scores"]
        row={"candidate_id":x["candidate_id"],"source_image_id":x["source_image_id"],
             "query_image_id":x["query_image_id"],"query":x["query"],"box_xyxy":x["box_xyxy"],
             "features":{"native_score":x["native_score"],"alpha_raw_full":b["views"]["full"]["target_cosine"],
                 "alpha_raw_context":b["views"]["context"]["target_cosine"],
                 "alpha_query_minus_generic_object":s["target"]-s["generic_object"],
                 "alpha_query_minus_background":s["target"]-s["background"],
                 "alpha_full_positive_pixels":b["views"]["full"]["alpha_pixels"]},
             "caption_evidence":{"status":"NOT_AVAILABLE_FOR_THIS_CANDIDATE"},
             "feature_decision":"NOT_MADE"}
        if not all(math.isfinite(v) for v in row["features"].values()):raise RuntimeError("nonfinite feature")
        rows.append(row)
    return rows,{"manifest":sha(base/"inference_manifest.json"),"alpha_freeze":sha(base/"freeze_scores.json"),
                 "text_control_freeze":sha(text/"freeze_scores.json")}


def build_absent111():
    base=OUT/"fqie_h2_alpha_absent111_top5_20261104_v1";text=OUT/"fqie_h2_alpha_absent111_text_controls_20261104_v1"
    manifest=json.loads((base/"inference_manifest.json").read_text());bf=json.loads((base/"freeze_scores.json").read_text())
    tf=json.loads((text/"freeze_scores.json").read_text());rows=[]
    for x in manifest["items"]:
        rel="scores/"+x["candidate_id"].replace(":","_")+".json";bp=base/rel;tp=text/rel
        if sha(bp)!=bf["score_file_hashes"][rel] or sha(tp)!=tf["score_file_hashes"][rel]:raise RuntimeError("score freeze mismatch")
        b=json.loads(bp.read_text());t=json.loads(tp.read_text());s=t["scores"]
        row={"candidate_id":x["candidate_id"],"item_id":x["item_id"],"source_image_id":x["source_image_id"],
             "query":x["query"],"native_rank":x["native_rank"],"box_xyxy":x["box_xyxy"],
             "features":{"native_score":x["native_score"],"alpha_raw_full":b["alpha_full"],
                 "alpha_query_minus_generic_object":s["target"]-s["generic_object"],
                 "alpha_query_minus_background":s["target"]-s["background"],
                 "alpha_full_positive_pixels":b["alpha_pixels"]},
             "caption_evidence":{"status":"NOT_AVAILABLE_FOR_THIS_CANDIDATE"},
             "feature_decision":"NOT_MADE"}
        if not all(math.isfinite(v) for v in row["features"].values()):raise RuntimeError("nonfinite feature")
        rows.append(row)
    return rows,{"manifest":sha(base/"inference_manifest.json"),"alpha_freeze":sha(base/"freeze_scores.json"),
                 "text_control_freeze":sha(text/"freeze_scores.json")}


def main():
    lvis,lh=build_lvis();absent,ah=build_absent111()
    if len(lvis)!=306 or len(absent)!=552:raise RuntimeError("denominator mismatch")
    write_once(DEST/"lvis_query_switch_evidence.jsonl",jsonl(lvis));write_once(DEST/"absent111_top5_evidence.jsonl",jsonl(absent))
    contract={"status":"FROZEN_DECISION_FREE_EVIDENCE","created_date":"2026-11-04",
      "feature_semantics":{"native_score":"frozen CountGD++ native score; not probability",
       "alpha_raw_full":"official Alpha-CLIP cosine on synchronized full view",
       "alpha_query_minus_generic_object":"same visual embedding: query cosine minus generic-object cosine",
       "alpha_query_minus_background":"same visual embedding: query cosine minus background cosine",
       "alpha_full_positive_pixels":"post-transform rectangle-alpha support; diagnostic validity feature"},
      "invalid_unknown_policy":"missing caption remains NOT_AVAILABLE; no score is converted to confidence or label",
      "input_hashes":{"lvis":lh,"absent111":ah},"denominators":{"lvis":len(lvis),"absent111_top5":len(absent)},
      "selector":"NOT_IMPLEMENTED_GATE_NOT_PASSED"}
    write_once(DEST/"feature_contract.json",(json.dumps(contract,indent=2)+"\n").encode())
    print(json.dumps(contract["denominators"]))


if __name__=="__main__":main()
