"""CycleGRPO unary query-switch on identical LVIS regions: target vs verified absent."""
from __future__ import annotations
import argparse,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"target_aware_counting/output"
sys.path.insert(0,str(ROOT/"target_aware_counting"));import fqie_h2_removed_caption as core
SRC=OUT/"lvis_query_switch_val_20261010_205455/input_274/query_pairs_coco.json"
IMAGE_ROOT=ROOT/"data/lvis_v1_val_source/selected_images";OLD=OUT/"cyclegrpo_capability_probe_20261027_v1"
DEST=OUT/"fqie_h2_lvis_paired_region_cycle_20261104_v1";core.DEST=DEST
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
    doc=json.loads(SRC.read_text());images={x["id"]:x for x in doc["images"]};anns=defaultdict(list)
    for a in doc["annotations"]:anns[a["image_id"]].append(a)
    absent={x["source_image_id"]:x for x in images.values() if x["query_status"]=="verified_absent"}
    candidates=defaultdict(list)
    for iid,im in images.items():
        if im["query_status"]!="present":continue
        for a in anns[iid]:candidates[im["source_image_id"]].append((a,im))
    if len(absent)!=100 or set(absent)!=set(candidates):raise RuntimeError("source pairing denominator changed")
    items=[];labels=[]
    for pair_idx,sid in enumerate(sorted(absent)):
        # Score-blind deterministic choice: largest official box, then annotation ID.
        a,pos=max(candidates[sid],key=lambda z:(z[0]["area"],-z[0]["lvis_annotation_id"]));neg=absent[sid]
        x,y,w,h=a["bbox"];box=[x,y,x+w,y+h];path=IMAGE_ROOT/pos["file_name"]
        if not path.is_file():raise RuntimeError(f"missing image {path}")
        clipped=[max(0,min(pos["width"],box[0])),max(0,min(pos["height"],box[1])),
                 max(0,min(pos["width"],box[2])),max(0,min(pos["height"],box[3]))]
        if clipped[2]<=clipped[0] or clipped[3]<=clipped[1]:raise RuntimeError("empty clipped GT box")
        image_sha=file_sha(path);pair_id=f"L{pair_idx:03d}"
        for suffix,query,label in (("P",pos["query_caption"],"TARGET_REGION"),("N",neg["query_caption"],"VERIFIED_ABSENT_REGION")):
            probe_id=pair_id+suffix;cid=f"lvis_paired:{sid}:{a['lvis_annotation_id']}:{suffix}"
            items.append({"probe_id":probe_id,"candidate_id":cid,"pair_id":pair_id,"source_image_id":sid,
              "image_path":str(path.resolve()),"image_sha256":image_sha,"original_box_xyxy":box,"box_xyxy":clipped,
              "box_clipped_to_image":clipped!=box,"query_for_later_audit_only":query})
            labels.append({"probe_id":probe_id,"candidate_id":cid,"pair_id":pair_id,"source_image_id":sid,
              "eval_label":label,"lvis_annotation_id":a["lvis_annotation_id"]})
    manifest={"status":"LOCKED_BEFORE_UNARY_INFERENCE","items":items,"max_regions":200,
      "selection":"one largest official GT object per source; identical region paired with its true query and one official verified-absent query",
      "selection_reads_model_scores":False,"source_query_coco_sha256":core.sha(SRC),"verify_requires_caption_freeze":False,
      "caption_stage":"not_run_for_paired_unary_query_switch","input_query_sent_to_caption_model":False,"max_new_tokens":8,
      "runtime_lock_sha256":core.sha(OLD/"runtime_lock.json")}
    core.write_once(DEST/"manifest.json",manifest);core.write_once(DEST/"eval_labels.json",{"status":"LOCKED_SEPARATELY_BEFORE_INFERENCE","items":labels})
    print(json.dumps({"pairs":100,"items":len(items),"manifest_sha256":core.sha(DEST/"manifest.json"),"labels_sha256":core.sha(DEST/"eval_labels.json")}))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","preflight","encode","verify","freeze-verify"));p.add_argument("--min-free-gib",type=float,default=18.0);a=p.parse_args()
    {"prepare":prepare,"preflight":core.preflight,"encode":lambda:core.gpu("encode",a.min_free_gib),
     "verify":lambda:core.gpu("verify",a.min_free_gib),"freeze-verify":core.freeze_verify}[a.stage]()
