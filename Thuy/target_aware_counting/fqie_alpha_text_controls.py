"""Frozen Alpha-CLIP text-control ablation on LVIS candidates."""
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"target_aware_counting"))
from fqie_alpha_smoke import alpha_mask,cosine,load_model,normalized_text,sha,synchronized_transform,write_once
BASE=ROOT/"target_aware_counting/output/fqie_h2_alpha_lvis_full_20261104_v1"
DEST=ROOT/"target_aware_counting/output/fqie_h2_alpha_text_controls_20261104_v1"
BASE_SHA="523637bfa91db228c19513644fe9b9fba94ba5855321533f2e58d9b9fcd774c9"


def texts(q):
    clean=q.strip().replace("_"," ").rstrip(" .")
    return {"target":f"a photo of {clean}.","generic_object":"a photo of an object.",
            "background":"a photo of the background.","without_target":f"a photo without {clean}.",
            "different_target":f"a photo of something other than {clean}."}


def prepare():
    if sha(BASE/"inference_manifest.json")!=BASE_SHA:raise RuntimeError("base changed")
    items=json.loads((BASE/"inference_manifest.json").read_text())["items"]
    write_once(DEST/"manifest.json",{"status":"LOCKED_BEFORE_TEXT_CONTROL_INFERENCE","base_sha256":BASE_SHA,
        "items":items,"text_controls":["generic_object","background","without_target","different_target"],
        "visual_input":"correct_rectangle_full_letterbox","inference_reads_labels":False})
    print(json.dumps({"items":len(items),"manifest_sha256":sha(DEST/"manifest.json")}))


def run():
    import torch
    from PIL import Image
    m=json.loads((DEST/"manifest.json").read_text());torch.manual_seed(0);model,ac=load_model();res=int(model.visual.input_resolution)
    cache={};start=time.monotonic()
    with torch.inference_mode():
        for x in m["items"]:
            out=DEST/"scores"/(x["candidate_id"].replace(":","_")+".json")
            if out.exists():continue
            image=Image.open(x["image_path"]).convert("RGB");q=x["query"];prompts=texts(q)
            if q not in cache:
                names=list(prompts);cache[q]=(names,model.encode_text(ac.tokenize([prompts[n] for n in names],truncate=True).cuda()))
            names,emb=cache[q];rt,at,pixels=synchronized_transform(image,alpha_mask(image.size,x["box_xyxy"]),res)
            visual=model.visual(rt.unsqueeze(0).to(model.dtype).cuda(),at.unsqueeze(0).to(model.dtype).cuda())
            scores={n:cosine(visual,emb[i:i+1]) for i,n in enumerate(names)}
            write_once(out,{"candidate_id":x["candidate_id"],"query":q,"prompts":prompts,"scores":scores,
                "alpha_pixels":pixels,"decision":"NOT_MADE"})
    print(json.dumps({"candidates":len(m["items"]),"wall_seconds":time.monotonic()-start,
        "peak_vram_gib":torch.cuda.max_memory_allocated()/1024**3}))


def freeze():
    m=json.loads((DEST/"manifest.json").read_text());hashes={}
    for x in m["items"]:
        p=DEST/"scores"/(x["candidate_id"].replace(":","_")+".json");d=json.loads(p.read_text())
        if d["candidate_id"]!=x["candidate_id"] or d["alpha_pixels"]<=0 or not all(math.isfinite(v) for v in d["scores"].values()):raise RuntimeError("invalid")
        hashes[str(p.relative_to(DEST))]=sha(p)
    write_once(DEST/"freeze_scores.json",{"status":"COMPLETE_RAW_TEXT_CONTROLS_NO_DECISION",
        "manifest_sha256":sha(DEST/"manifest.json"),"score_file_hashes":hashes});print(json.dumps({"frozen":len(hashes)}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("prepare","run","freeze"));a=p.parse_args().action
    {"prepare":prepare,"run":run,"freeze":freeze}[a]()
