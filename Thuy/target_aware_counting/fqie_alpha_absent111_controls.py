"""Same-query Alpha-CLIP region controls on frozen absent111 top-5 boxes."""
from __future__ import annotations

import argparse,json,math,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"target_aware_counting"))
from fqie_alpha_smoke import alpha_mask,cosine,load_model,normalized_text,sha,synchronized_transform,write_once
from fqie_alpha_region_controls import far_box,box_iou

BASE=ROOT/"target_aware_counting/output/fqie_h2_alpha_absent111_top5_20261104_v1"
DEST=ROOT/"target_aware_counting/output/fqie_h2_alpha_absent111_controls_20261104_v1"
BASE_SHA="48cbe6edb2d75de3842784a6e0300dbae377fed3162ae8eda97320c4677b5bcf"


def prepare():
    if sha(BASE/"inference_manifest.json")!=BASE_SHA:raise RuntimeError("base manifest changed")
    items=json.loads((BASE/"inference_manifest.json").read_text())["items"]
    write_once(DEST/"manifest.json",{"status":"LOCKED_BEFORE_CONTROL_INFERENCE","base_manifest_sha256":BASE_SHA,
        "items":items,"controls":["all_content","complement_rectangle","far_same_size_rectangle"],
        "view":"full_letterbox_identical_to_correct_region","inference_reads_labels":False})
    print(json.dumps({"items":len(items),"manifest_sha256":sha(DEST/"manifest.json")}))


def run():
    import torch
    from PIL import Image,ImageChops
    m=json.loads((DEST/"manifest.json").read_text())
    if len(m["items"])!=552 or m["base_manifest_sha256"]!=BASE_SHA:raise RuntimeError("manifest mismatch")
    torch.manual_seed(0);model,ac=load_model();res=int(model.visual.input_resolution);cache={};start=time.monotonic()
    with torch.inference_mode():
        for item in m["items"]:
            out=DEST/"scores"/(item["candidate_id"].replace(":","_")+".json")
            if out.exists():
                if json.loads(out.read_text())["candidate_id"]!=item["candidate_id"]:raise RuntimeError("resume mismatch")
                continue
            image=Image.open(item["image_path"]).convert("RGB")
            if sha(Path(item["image_path"]))!=item["image_sha256"]:raise RuntimeError("image changed")
            q=item["query"]
            if q not in cache:cache[q]=model.encode_text(ac.tokenize([normalized_text(q)],truncate=True).cuda())
            correct=alpha_mask(image.size,item["box_xyxy"]);allm=Image.new("L",image.size,255)
            complement=ImageChops.invert(correct);shiftbox=far_box(image.size,item["box_xyxy"]);shift=alpha_mask(image.size,shiftbox)
            values={}
            for name,mask in (("all_content",allm),("complement_rectangle",complement),("far_same_size_rectangle",shift)):
                rt,at,pixels=synchronized_transform(image,mask,res)
                visual=model.visual(rt.unsqueeze(0).to(model.dtype).cuda(),at.unsqueeze(0).to(model.dtype).cuda())
                values[name]={"alpha_pixels":pixels,"target_cosine":cosine(visual,cache[q])}
            write_once(out,{"candidate_id":item["candidate_id"],"item_id":item["item_id"],"controls":values,
                "shifted_box_xyxy":shiftbox,"shifted_vs_correct_iou":box_iou(shiftbox,item["box_xyxy"]),"decision":"NOT_MADE"})
        print(json.dumps({"candidates":len(m["items"]),"visual_forwards":3*len(m["items"]),
            "wall_seconds":time.monotonic()-start,"peak_vram_gib":torch.cuda.max_memory_allocated()/1024**3}))


def freeze():
    m=json.loads((DEST/"manifest.json").read_text());hashes={}
    for item in m["items"]:
        p=DEST/"scores"/(item["candidate_id"].replace(":","_")+".json");d=json.loads(p.read_text())
        if d["candidate_id"]!=item["candidate_id"]:raise RuntimeError("binding mismatch")
        if not all(math.isfinite(v["target_cosine"]) and v["alpha_pixels"]>0 for v in d["controls"].values()):raise RuntimeError("invalid")
        hashes[str(p.relative_to(DEST))]=sha(p)
    write_once(DEST/"freeze_scores.json",{"status":"COMPLETE_RAW_CONTROLS_NO_DECISION",
        "manifest_sha256":sha(DEST/"manifest.json"),"score_file_hashes":hashes});print(json.dumps({"frozen":len(hashes)}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("prepare","run","freeze"));a=p.parse_args().action
    {"prepare":prepare,"run":run,"freeze":freeze}[a]()
