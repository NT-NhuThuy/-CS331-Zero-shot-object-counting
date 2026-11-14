"""Numerically matched target+generic Alpha calibration on frozen LVIS boxes."""
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"target_aware_counting"))
from fqie_alpha_smoke import alpha_mask,cosine,load_model,normalized_text,sha,synchronized_transform,write_once
BASE=ROOT/"target_aware_counting/output/fqie_h2_alpha_lvis_full_20261104_v1"
DEST=ROOT/"target_aware_counting/output/fqie_h2_alpha_lvis_two_text_20261104_v1"
BASE_SHA="523637bfa91db228c19513644fe9b9fba94ba5855321533f2e58d9b9fcd774c9"


def prepare():
    if sha(BASE/"inference_manifest.json")!=BASE_SHA:raise RuntimeError("base changed")
    items=json.loads((BASE/"inference_manifest.json").read_text())["items"]
    write_once(DEST/"manifest.json",{"status":"LOCKED_NUMERIC_PARITY_CALIBRATION","base_sha256":BASE_SHA,"items":items,
        "text_batch_order":["target","generic_object"],"view":"full_letterbox","inference_reads_labels":False})
    print(json.dumps({"items":len(items),"manifest_sha256":sha(DEST/"manifest.json")}))


def run():
    import torch
    from PIL import Image
    m=json.loads((DEST/"manifest.json").read_text());torch.manual_seed(0);model,ac=load_model();res=int(model.visual.input_resolution);cache={};start=time.monotonic()
    with torch.inference_mode():
        for x in m["items"]:
            out=DEST/"scores"/(x["candidate_id"].replace(":","_")+".json")
            if out.exists():continue
            image=Image.open(x["image_path"]).convert("RGB");q=x["query"]
            if q not in cache:cache[q]=model.encode_text(ac.tokenize([normalized_text(q),"a photo of an object."],truncate=True).cuda())
            rt,at,pixels=synchronized_transform(image,alpha_mask(image.size,x["box_xyxy"]),res)
            visual=model.visual(rt.unsqueeze(0).to(model.dtype).cuda(),at.unsqueeze(0).to(model.dtype).cuda())
            raw=cosine(visual,cache[q][0:1]);generic=cosine(visual,cache[q][1:2])
            write_once(out,{"candidate_id":x["candidate_id"],"alpha_raw_full":raw,"alpha_generic_object":generic,
                "alpha_query_minus_generic_object":raw-generic,"alpha_pixels":pixels,"decision":"NOT_MADE"})
    print(json.dumps({"items":len(m["items"]),"wall_seconds":time.monotonic()-start,
        "peak_vram_gib":torch.cuda.max_memory_allocated()/1024**3}))


def freeze():
    m=json.loads((DEST/"manifest.json").read_text());hashes={}
    for x in m["items"]:
        p=DEST/"scores"/(x["candidate_id"].replace(":","_")+".json");d=json.loads(p.read_text())
        if d["candidate_id"]!=x["candidate_id"] or d["alpha_pixels"]<=0 or not all(math.isfinite(d[k]) for k in ("alpha_raw_full","alpha_generic_object","alpha_query_minus_generic_object")):raise RuntimeError("invalid")
        hashes[str(p.relative_to(DEST))]=sha(p)
    write_once(DEST/"freeze_scores.json",{"status":"COMPLETE_MATCHED_TWO_TEXT_RAW","manifest_sha256":sha(DEST/"manifest.json"),"score_file_hashes":hashes})
    print(json.dumps({"frozen":len(hashes)}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("prepare","run","freeze"));a=p.parse_args().action
    {"prepare":prepare,"run":run,"freeze":freeze}[a]()
