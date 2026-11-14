"""Frozen CycleGRPO unary evidence for all 306 LVIS M0 proposals."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"target_aware_counting/output"
sys.path.insert(0,str(ROOT/"target_aware_counting"));import fqie_h2_removed_caption as core
BASE=OUT/"fqie_h2_alpha_lvis_full_20261104_v1";OLD=OUT/"cyclegrpo_capability_probe_20261027_v1"
DEST=OUT/"fqie_h2_lvis306_cycle_unary_20261104_v1";core.DEST=DEST
def prepare():
    source=json.loads((BASE/"inference_manifest.json").read_text());items=[]
    for i,x in enumerate(source["items"]):
        w,h=Image.open(x["image_path"]).size;b=x["box_xyxy"];clip=[max(0,min(w,b[0])),max(0,min(h,b[1])),max(0,min(w,b[2])),max(0,min(h,b[3]))]
        if clip[2]<=clip[0] or clip[3]<=clip[1]:raise RuntimeError("empty clipped proposal")
        items.append({"probe_id":f"V{i:03d}","candidate_id":x["candidate_id"],"image_path":x["image_path"],"image_sha256":x["image_sha256"],
          "original_box_xyxy":b,"box_xyxy":clip,"box_clipped_to_image":clip!=b,"query_for_later_audit_only":x["query"]})
    if len(items)!=306:raise RuntimeError(f"proposal denominator changed: {len(items)}")
    core.write_once(DEST/"manifest.json",{"status":"LOCKED_BEFORE_UNARY_INFERENCE","items":items,"max_regions":306,
      "selection":"all frozen no-pseudo M0 proposals in 274-pair LVIS query-switch development cohort",
      "source_manifest_sha256":core.sha(BASE/"inference_manifest.json"),"verify_requires_caption_freeze":False,"caption_stage":"not_run",
      "input_query_sent_to_caption_model":False,"max_new_tokens":8,"runtime_lock_sha256":core.sha(OLD/"runtime_lock.json")})
    print(json.dumps({"items":len(items),"manifest_sha256":core.sha(DEST/"manifest.json")}))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","preflight","encode","verify","freeze-verify"));p.add_argument("--min-free-gib",type=float,default=18.0);a=p.parse_args()
    {"prepare":prepare,"preflight":core.preflight,"encode":lambda:core.gpu("encode",a.min_free_gib),
     "verify":lambda:core.gpu("verify",a.min_free_gib),"freeze-verify":core.freeze_verify}[a.stage]()
