"""Frozen CycleGRPO unary safety probe on one native-top candidate per absent111 query."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"target_aware_counting/output"
sys.path.insert(0,str(ROOT/"target_aware_counting"))
import fqie_h2_removed_caption as core
SRC=OUT/"fqie_h2_alpha_absent111_full_20261104_v1/manifest.json"
OLD=OUT/"cyclegrpo_capability_probe_20261027_v1"
DEST=OUT/"fqie_h2_absent111_cycle_top1_20261104_v1";core.DEST=DEST

def prepare():
    src=json.loads(SRC.read_text());items=[]
    for q in src["items"]:
        c=max(q["candidates"],key=lambda x:(x["native_score"],-x["slot"]));w,h=Image.open(q["image_path"]).size;b=c["box_xyxy"]
        clip=[max(0,min(w,b[0])),max(0,min(h,b[1])),max(0,min(w,b[2])),max(0,min(h,b[3]))]
        if clip[2]<=clip[0] or clip[3]<=clip[1]:raise RuntimeError("empty clipped top1")
        items.append({"probe_id":f"A{len(items):03d}","candidate_id":c["candidate_id"],"item_id":q["item_id"],
          "image_path":q["image_path"],"image_sha256":q["image_sha256"],"original_box_xyxy":b,"box_xyxy":clip,
          "box_clipped_to_image":clip!=b,"query_for_later_audit_only":q["query"],"native_score":c["native_score"]})
    if len(items)!=111:raise RuntimeError(f"absent denominator changed: {len(items)}")
    core.write_once(DEST/"manifest.json",{"status":"LOCKED_BEFORE_CAPTION_INFERENCE","items":items,"max_regions":111,
      "selection":"highest frozen native-score M0 candidate per verified-absent FSC query",
      "source_manifest_sha256":core.sha(SRC),"input_query_sent_to_caption_model":False,
      "verify_requires_caption_freeze":False,"caption_stage":"not_run_for_this_unary_safety_probe",
      "box_policy":"clip detector box to source image; retain original box and clip flag","max_new_tokens":8,
      "runtime_lock_sha256":core.sha(OLD/"runtime_lock.json")})
    print(json.dumps({"items":len(items),"manifest_sha256":core.sha(DEST/"manifest.json")}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","preflight","encode","verify","freeze-verify"));p.add_argument("--min-free-gib",type=float,default=18.0);a=p.parse_args()
    {"prepare":prepare,"preflight":core.preflight,"encode":lambda:core.gpu("encode",a.min_free_gib),
     "verify":lambda:core.gpu("verify",a.min_free_gib),"freeze-verify":core.freeze_verify}[a.stage]()
