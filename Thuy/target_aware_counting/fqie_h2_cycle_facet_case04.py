"""Exploratory Cycle unary facet probe on the locked pickup case_04."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"target_aware_counting/output"
sys.path.insert(0,str(ROOT/"target_aware_counting"));import fqie_h2_removed_caption as core
OLD=OUT/"cyclegrpo_capability_probe_20261027_v1";DEST=OUT/"fqie_h2_cycle_facet_case04_20261104_v1";core.DEST=DEST
QUERIES=["pickup truck","green pickup truck","white pickup truck","green pickup truck in front of a white pickup truck","white pickup truck behind a green pickup truck","red pickup truck"]
# Prior assistant visual interpretation only; not independent GT.
EXPECTED=[[1,1,0,1,0,0],[1,1,0,1,0,0],[1,0,1,0,1,0]]
def prepare():
    case=next(json.loads(s) for s in (OLD/"cases_input.jsonl").read_text().splitlines() if '"case_id":"case_04"' in s);items=[];labels=[]
    for ri,r in enumerate(case["regions"]):
        for qi,q in enumerate(QUERIES):
            pid=f"R{ri}Q{qi}";cid=f"case04:{r['proposal_id']}:{qi}"
            items.append({"probe_id":pid,"candidate_id":cid,"image_path":case["image_path"],"image_sha256":case["image_sha256"],
              "original_box_xyxy":r["bbox_xyxy"],"box_xyxy":r["bbox_xyxy"],"box_clipped_to_image":False,"query_for_later_audit_only":q})
            labels.append({"probe_id":pid,"candidate_id":cid,"region_index":ri,"query_index":qi,"query":q,
              "expected_decision":"yes" if EXPECTED[ri][qi] else "no","label_provenance":"prior_assistant_visual_interpretation_exploratory"})
    core.write_once(DEST/"manifest.json",{"status":"LOCKED_BEFORE_UNARY_INFERENCE","items":items,"max_regions":18,
      "selection":"three locked case_04 regions x six fixed facet queries","verify_requires_caption_freeze":False,
      "caption_stage":"not_run","input_query_sent_to_caption_model":False,"max_new_tokens":8,
      "runtime_lock_sha256":core.sha(OLD/"runtime_lock.json")})
    core.write_once(DEST/"eval_labels.json",{"status":"EXPLORATORY_LABELS_LOCKED_BEFORE_INFERENCE","items":labels,
      "warning":"assistant visual labels are not independent confirmation"})
    print(json.dumps({"items":18,"manifest_sha256":core.sha(DEST/"manifest.json"),"labels_sha256":core.sha(DEST/"eval_labels.json")}))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","preflight","encode","verify","freeze-verify"));p.add_argument("--min-free-gib",type=float,default=18.0);a=p.parse_args()
    {"prepare":prepare,"preflight":core.preflight,"encode":lambda:core.gpu("encode",a.min_free_gib),
     "verify":lambda:core.gpu("verify",a.min_free_gib),"freeze-verify":core.freeze_verify}[a.stage]()
