"""Query-free CycleGRPO captions for candidates removed by the fixed H2 rule."""
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"target_aware_counting/output"
BASE=OUT/"fqie_h2_alpha_lvis_full_20261104_v1";SCORES=OUT/"fqie_h2_alpha_lvis_two_text_20261104_v1"
RULES=OUT/"fqie_h2_full_mass_rule_20261104_v1/evaluation.json";OLD=OUT/"cyclegrpo_capability_probe_20261027_v1"
DEST=OUT/"fqie_h2_removed_caption_20261104_v2";sys.path.insert(0,str(OLD))
from fqie_alpha_smoke import sha,write_once


def prepare():
    from PIL import Image
    manifest=json.loads((BASE/"inference_manifest.json").read_text())["items"]
    thresholds=json.loads(RULES.read_text())["rules"]["raw_generic_bonferroni95"]["thresholds"]
    items=[]
    for x in manifest:
        d=json.loads((SCORES/"scores"/(x["candidate_id"].replace(":","_")+".json")).read_text())
        keep=d["alpha_raw_full"]>=thresholds["raw"] and d["alpha_query_minus_generic_object"]>=thresholds["margin"]
        if not keep:
            width,height=Image.open(x["image_path"]).size;box=x["box_xyxy"]
            clipped=[max(0,min(width,box[0])),max(0,min(height,box[1])),
                     max(0,min(width,box[2])),max(0,min(height,box[3]))]
            if clipped[2]<=clipped[0] or clipped[3]<=clipped[1]:raise RuntimeError("empty clipped box")
            items.append({"probe_id":f"R{len(items):03d}","candidate_id":x["candidate_id"],
                "image_path":x["image_path"],"image_sha256":x["image_sha256"],
                "original_box_xyxy":box,"box_xyxy":clipped,"box_clipped_to_image":clipped!=box,
                "query_for_later_audit_only":x["query"]})
    if len(items)!=40:raise RuntimeError(f"removed denominator changed: {len(items)}")
    write_once(DEST/"manifest.json",{"status":"LOCKED_BEFORE_CAPTION_INFERENCE","items":items,"max_regions":40,
        "selection":"removed by frozen raw+generic Bonferroni95 rule","caption_prompt":"native CycleGRPO query-free region caption",
        "box_policy":"clip detector box to source image; retain original box and clip flag",
        "max_new_tokens":96,"input_query_sent_to_caption_model":False,"runtime_lock_sha256":sha(OLD/"runtime_lock.json")})
    print(json.dumps({"items":len(items),"manifest_sha256":sha(DEST/"manifest.json")}))


def preflight():
    import importlib.util
    from runtime_gate import inspect_lock
    report=inspect_lock(OLD/"runtime_lock.json")
    if importlib.util.find_spec("iopath") is None:
        report["blockers"].append("runtime Python dependency missing: iopath")
    if report["blockers"]:raise RuntimeError(report["blockers"])
    write_once(DEST/"preflight.json",{"state":report["state"],"runtime_lock_sha256":sha(OLD/"runtime_lock.json")})
    print(json.dumps({"preflight":report["state"]}))


def freeze():
    """Validate and hash model outputs before any evaluation labels are joined."""
    m=json.loads((DEST/"manifest.json").read_text());rows=[]
    if m["status"]!="LOCKED_BEFORE_CAPTION_INFERENCE" or len(m["items"])!=m["max_regions"]:
        raise RuntimeError("unexpected locked manifest")
    for x in m["items"]:
        pid=x["probe_id"];ep=DEST/"encodings"/(pid+".json");cp=DEST/"captions"/(pid+".json")
        if not ep.exists() or not cp.exists():raise RuntimeError(f"incomplete output: {pid}")
        enc=json.loads(ep.read_text());cap=json.loads(cp.read_text())
        for d in (enc,cap):
            for key in ("probe_id","candidate_id","image_sha256"):
                if d.get(key)!=x[key]:raise RuntimeError(f"binding mismatch {pid}:{key}")
        prompt=cap.get("prompt_text","");query=x["query_for_later_audit_only"].strip(" .").lower()
        if query and query in prompt.lower():raise RuntimeError(f"query leakage: {pid}")
        if not isinstance(cap.get("raw_response"),str) or not cap["raw_response"].strip():
            raise RuntimeError(f"empty caption: {pid}")
        numeric=(cap.get("input_tokens"),cap.get("output_tokens"),cap.get("generation_seconds"))
        if not all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in numeric):
            raise RuntimeError(f"invalid generation metadata: {pid}")
        rows.append({"probe_id":pid,"candidate_id":x["candidate_id"],"encoding_sha256":sha(ep),
                     "caption_sha256":sha(cp),"output_tokens":cap["output_tokens"],
                     "token_ceiling_reached":cap["output_tokens"]>=m["max_new_tokens"]})
    payload={"status":"FROZEN_BEFORE_LABEL_JOIN","manifest_sha256":sha(DEST/"manifest.json"),
             "runtime_lock_sha256":m["runtime_lock_sha256"],"items":rows,
             "query_free_inference":True,"semantic_correctness":"unverified",
             "token_ceiling_policy":"ceiling outputs remain usable text but truncation status is UNKNOWN"}
    write_once(DEST/"freeze_outputs.json",payload)
    print(json.dumps({"frozen":len(rows),"freeze_sha256":sha(DEST/"freeze_outputs.json"),
                      "token_ceiling_reached":sum(r["token_ceiling_reached"] for r in rows)}))


def freeze_verify():
    """Freeze exact unary responses and their strict three-way parse before label join."""
    m=json.loads((DEST/"manifest.json").read_text());rows=[]
    caption_freeze=DEST/"freeze_outputs.json"
    if m.get("verify_requires_caption_freeze",True) and not caption_freeze.exists():
        raise RuntimeError("caption freeze missing")
    for x in m["items"]:
        p=DEST/"yesno"/(x["probe_id"]+".json")
        if not p.exists():raise RuntimeError(f"missing verify output: {x['probe_id']}")
        d=json.loads(p.read_text())
        for key in ("probe_id","candidate_id","image_sha256"):
            if d.get(key)!=x[key]:raise RuntimeError(f"verify binding mismatch: {x['probe_id']}:{key}")
        if d.get("query")!=x["query_for_later_audit_only"]:raise RuntimeError("query binding mismatch")
        tail=d.get("raw_response","").split("</think>")[-1].strip().lower()
        decision=tail if tail in {"yes","no","uncertain"} else "UNKNOWN"
        rows.append({"probe_id":x["probe_id"],"candidate_id":x["candidate_id"],
                     "response_sha256":sha(p),"strict_decision":decision})
    payload={"status":"FROZEN_BEFORE_LABEL_JOIN","manifest_sha256":sha(DEST/"manifest.json"),
             "caption_freeze_sha256":sha(caption_freeze) if caption_freeze.exists() else None,"items":rows,
             "parser":"lowercase exact final text after optional </think>: yes|no|uncertain; else UNKNOWN",
             "semantic_correctness":"unverified"}
    write_once(DEST/"freeze_yesno.json",payload)
    print(json.dumps({"frozen":len(rows),"freeze_sha256":sha(DEST/"freeze_yesno.json"),
                      "decisions":{v:sum(r["strict_decision"]==v for r in rows) for v in ("yes","no","uncertain","UNKNOWN")}}))


def gpu(stage, min_free_gib):
    import torch
    from run_native_smoke import gpu_gate,load_vq,load_qwen
    from native_bridge import encode_rectangle_region_native,generate_native_message
    from capability_parser import native_caption_message,unary_query_yesno_message,clean_jpeg_data_url
    lock=json.loads((OLD/"runtime_lock.json").read_text());m=json.loads((DEST/"manifest.json").read_text())
    if json.loads((DEST/"preflight.json").read_text())["runtime_lock_sha256"]!=sha(OLD/"runtime_lock.json"):raise RuntimeError("preflight changed")
    torch.manual_seed(0)
    if min_free_gib > 0:
        gpu_gate(min_free_gib)
    elif not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    start=time.monotonic()
    if stage=="encode":
        model=load_vq(lock)
        for x in m["items"]:
            p=DEST/"encodings"/(x["probe_id"]+".json")
            if p.exists():continue
            r=encode_rectangle_region_native(x["image_path"],x["box_xyxy"],model)
            write_once(p,{"probe_id":x["probe_id"],"candidate_id":x["candidate_id"],"image_sha256":x["image_sha256"],**r})
            print(json.dumps({"encoded":x["probe_id"]}),flush=True)
    elif stage=="caption":
        if not all((DEST/"encodings"/(x["probe_id"]+".json")).exists() for x in m["items"]):raise RuntimeError("encode incomplete")
        model,processor=load_qwen(lock)
        for x in m["items"]:
            p=DEST/"captions"/(x["probe_id"]+".json")
            if p.exists():continue
            enc=json.loads((DEST/"encodings"/(x["probe_id"]+".json")).read_text())
            message=native_caption_message(enc["mask_tokens"],clean_jpeg_data_url(x["image_path"]));prompt=message[0]["content"][1]["text"]
            if x["query_for_later_audit_only"].strip(" .").lower() in prompt.lower():raise RuntimeError("query leakage")
            r=generate_native_message(model,processor,message,96)
            write_once(p,{"probe_id":x["probe_id"],"candidate_id":x["candidate_id"],"image_sha256":x["image_sha256"],
                "prompt_text":prompt,**r});print(json.dumps({"captioned":x["probe_id"],"seconds":r["generation_seconds"]}),flush=True)
    else:
        if m.get("verify_requires_caption_freeze",True) and not (DEST/"freeze_outputs.json").exists():
            raise RuntimeError("freeze query-free captions first")
        if not all((DEST/"encodings"/(x["probe_id"]+".json")).exists() for x in m["items"]):raise RuntimeError("encode incomplete")
        model,processor=load_qwen(lock)
        for x in m["items"]:
            p=DEST/"yesno"/(x["probe_id"]+".json")
            if p.exists():continue
            enc=json.loads((DEST/"encodings"/(x["probe_id"]+".json")).read_text())
            message=unary_query_yesno_message(enc["mask_tokens"],x["query_for_later_audit_only"],
                                               clean_jpeg_data_url(x["image_path"]))
            prompt=message[0]["content"][1]["text"]
            r=generate_native_message(model,processor,message,8)
            write_once(p,{"probe_id":x["probe_id"],"candidate_id":x["candidate_id"],"image_sha256":x["image_sha256"],
                          "query":x["query_for_later_audit_only"],"prompt_text":prompt,**r})
            print(json.dumps({"verified":x["probe_id"],"seconds":r["generation_seconds"]}),flush=True)
    print(json.dumps({"stage":stage,"wall_seconds":time.monotonic()-start,"peak_vram_gib":torch.cuda.max_memory_allocated()/1024**3}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","preflight","encode","caption","verify","freeze","freeze-verify"))
    p.add_argument("--min-free-gib",type=float,default=18.0,
                   help="free-VRAM guard; set 0 to attempt allocation and let CUDA OOM stop the stage")
    args=p.parse_args();a=args.stage
    {"prepare":prepare,"preflight":preflight,"encode":lambda:gpu("encode",args.min_free_gib),
     "caption":lambda:gpu("caption",args.min_free_gib),"verify":lambda:gpu("verify",args.min_free_gib),
     "freeze":freeze,"freeze-verify":freeze_verify}[a]()
