"""CycleGRPO evidence for source-disjoint cross-fit Alpha removals."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"target_aware_counting/output"
sys.path.insert(0,str(ROOT/"target_aware_counting"))
import fqie_h2_removed_caption as core

BASE=OUT/"fqie_h2_alpha_lvis_full_20261104_v1"
CROSS=OUT/"fqie_h2_alpha_source_crossfit_20261104_v1/manifest.json"
OLD=OUT/"cyclegrpo_capability_probe_20261027_v1"
DEST=OUT/"fqie_h2_crossfit_cyclegrpo_20261104_v1"
core.DEST=DEST

def prepare():
    from PIL import Image
    candidates={x["candidate_id"]:x for x in json.loads((BASE/"inference_manifest.json").read_text())["items"]}
    cross=json.loads(CROSS.read_text());removed=[x for x in cross["decisions"] if not x["keep"]]
    items=[]
    for d in removed:
        x=candidates[d["candidate_id"]];width,height=Image.open(x["image_path"]).size;box=x["box_xyxy"]
        clipped=[max(0,min(width,box[0])),max(0,min(height,box[1])),
                 max(0,min(width,box[2])),max(0,min(height,box[3]))]
        if clipped[2]<=clipped[0] or clipped[3]<=clipped[1]:raise RuntimeError("empty clipped box")
        items.append({"probe_id":f"C{len(items):03d}","candidate_id":x["candidate_id"],
          "image_path":x["image_path"],"image_sha256":x["image_sha256"],"original_box_xyxy":box,
          "box_xyxy":clipped,"box_clipped_to_image":clipped!=box,
          "query_for_later_audit_only":x["query"],"crossfit_fold":d["fold"]})
    if len(items)!=61:raise RuntimeError(f"cross-fit removed denominator changed: {len(items)}")
    core.write_once(DEST/"manifest.json",{"status":"LOCKED_BEFORE_CAPTION_INFERENCE","items":items,"max_regions":61,
      "selection":"removed by frozen five-fold source-disjoint Alpha cross-fit decisions",
      "crossfit_manifest_sha256":core.sha(CROSS),"caption_prompt":"native CycleGRPO query-free region caption",
      "box_policy":"clip detector box to source image; retain original box and clip flag","max_new_tokens":96,
      "input_query_sent_to_caption_model":False,"runtime_lock_sha256":core.sha(OLD/"runtime_lock.json")})
    print(json.dumps({"items":len(items),"manifest_sha256":core.sha(DEST/"manifest.json")}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","preflight","encode","caption","verify","freeze","freeze-verify"))
    p.add_argument("--min-free-gib",type=float,default=18.0);a=p.parse_args()
    {"prepare":prepare,"preflight":core.preflight,"encode":lambda:core.gpu("encode",a.min_free_gib),
     "caption":lambda:core.gpu("caption",a.min_free_gib),"verify":lambda:core.gpu("verify",a.min_free_gib),
     "freeze":core.freeze,"freeze-verify":core.freeze_verify}[a.stage]()
