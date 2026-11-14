"""Frozen CycleGRPO unary probe on every strict IoU50 LVIS target proposal."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"target_aware_counting/output"
sys.path.insert(0,str(ROOT/"target_aware_counting"));import fqie_h2_removed_caption as core
BASE=OUT/"fqie_h2_alpha_lvis_full_20261104_v1";OLD=OUT/"cyclegrpo_capability_probe_20261027_v1"
DEST=OUT/"fqie_h2_target164_cycle_unary_20261104_v1";core.DEST=DEST
def prepare():
    labels={x["candidate_id"]:x for x in json.loads((BASE/"eval_labels.json").read_text())["items"]}
    candidates={x["candidate_id"]:x for x in json.loads((BASE/"inference_manifest.json").read_text())["items"]};items=[]
    for cid in sorted(c for c,x in labels.items() if x["eval_label"]=="TARGET_IOU50"):
        x=candidates[cid];w,h=Image.open(x["image_path"]).size;b=x["box_xyxy"]
        clip=[max(0,min(w,b[0])),max(0,min(h,b[1])),max(0,min(w,b[2])),max(0,min(h,b[3]))]
        if clip[2]<=clip[0] or clip[3]<=clip[1]:raise RuntimeError("empty clipped target")
        items.append({"probe_id":f"T{len(items):03d}","candidate_id":cid,"image_path":x["image_path"],
          "image_sha256":x["image_sha256"],"original_box_xyxy":b,"box_xyxy":clip,
          "box_clipped_to_image":clip!=b,"query_for_later_audit_only":x["query"]})
    if len(items)!=164:raise RuntimeError(f"target denominator changed: {len(items)}")
    core.write_once(DEST/"manifest.json",{"status":"LOCKED_BEFORE_CAPTION_INFERENCE","items":items,"max_regions":164,
      "selection":"all frozen LVIS strict TARGET_IOU50 proposals","labels_sha256":core.sha(BASE/"eval_labels.json"),
      "verify_requires_caption_freeze":False,"caption_stage":"not_run_for_this_unary_sensitivity_probe",
      "input_query_sent_to_caption_model":False,"box_policy":"clip detector box to source image; retain original box and clip flag",
      "max_new_tokens":8,"runtime_lock_sha256":core.sha(OLD/"runtime_lock.json")})
    print(json.dumps({"items":len(items),"manifest_sha256":core.sha(DEST/"manifest.json")}))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","preflight","encode","verify","freeze-verify"));p.add_argument("--min-free-gib",type=float,default=18.0);a=p.parse_args()
    {"prepare":prepare,"preflight":core.preflight,"encode":lambda:core.gpu("encode",a.min_free_gib),
     "verify":lambda:core.gpu("verify",a.min_free_gib),"freeze-verify":core.freeze_verify}[a.stage]()
