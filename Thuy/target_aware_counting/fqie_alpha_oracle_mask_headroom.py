"""Oracle LVIS mask diagnostic: estimates mask-quality headroom only."""
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"target_aware_counting"))
from fqie_alpha_smoke import cosine,load_model,normalized_text,sha,synchronized_transform,write_once

BASE=ROOT/"target_aware_counting/output/fqie_h2_alpha_lvis_full_20261104_v1"
QUERY_GT=ROOT/"target_aware_counting/output/lvis_query_switch_val_20261010_205455/input_274/query_pairs_coco.json"
LVIS=ROOT/"data/lvis_v1_val_source/lvis_v1_val.json"
DEST=ROOT/"target_aware_counting/output/fqie_h2_alpha_oracle_mask_headroom_20261104_v1"
LVIS_SHA="2bf946b92c3037f53c172d80017f5b74ea035f00a21b20e0766b3b638b2363f9"


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb>inter else 0


def xyxy(b):return [b[0],b[1],b[0]+b[2],b[1]+b[3]]


def prepare():
    if sha(LVIS)!=LVIS_SHA:raise RuntimeError("LVIS source changed")
    manifest=json.loads((BASE/"inference_manifest.json").read_text())["items"]
    labels={x["candidate_id"]:x for x in json.loads((BASE/"eval_labels.json").read_text())["items"]}
    qgt=json.loads(QUERY_GT.read_text());anns={}
    for a in qgt["annotations"]:anns.setdefault(a["image_id"],[]).append(a)
    lvis={a["id"]:a for a in json.loads(LVIS.read_text())["annotations"]}
    items=[]
    for x in manifest:
        if labels[x["candidate_id"]]["eval_label"]!="TARGET_IOU50":continue
        matches=anns[x["query_image_id"]];best=max(matches,key=lambda a:iou(x["box_xyxy"],xyxy(a["bbox"])))
        overlap=iou(x["box_xyxy"],xyxy(best["bbox"]))
        if overlap<.5:raise RuntimeError("positive/match mismatch")
        source=lvis[best["lvis_annotation_id"]]
        items.append({**x,"lvis_annotation_id":best["lvis_annotation_id"],"matched_iou":overlap,
                      "segmentation":source["segmentation"]})
    if len(items)!=164:raise RuntimeError("unexpected positive count")
    write_once(DEST/"manifest.json",{"status":"LOCKED_ORACLE_DIAGNOSTIC","items":items,"lvis_sha256":LVIS_SHA,
        "warning":"uses official GT polygons unavailable to inference; headroom only"})
    print(json.dumps({"items":len(items),"manifest_sha256":sha(DEST/"manifest.json")}))


def run():
    import torch
    from PIL import Image,ImageDraw
    m=json.loads((DEST/"manifest.json").read_text());torch.manual_seed(0);model,ac=load_model();res=int(model.visual.input_resolution)
    cache={};start=time.monotonic()
    with torch.inference_mode():
        for x in m["items"]:
            out=DEST/"scores"/(x["candidate_id"].replace(":","_")+".json")
            if out.exists():continue
            image=Image.open(x["image_path"]).convert("RGB");mask=Image.new("L",image.size,0);draw=ImageDraw.Draw(mask)
            for polygon in x["segmentation"]:
                if len(polygon)>=6:draw.polygon([(polygon[i],polygon[i+1]) for i in range(0,len(polygon),2)],fill=255)
            q=x["query"]
            if q not in cache:cache[q]=model.encode_text(ac.tokenize([normalized_text(q)],truncate=True).cuda())
            rt,at,pixels=synchronized_transform(image,mask,res)
            if pixels==0:raise RuntimeError("oracle mask vanished")
            visual=model.visual(rt.unsqueeze(0).to(model.dtype).cuda(),at.unsqueeze(0).to(model.dtype).cuda())
            write_once(out,{"candidate_id":x["candidate_id"],"lvis_annotation_id":x["lvis_annotation_id"],
                "oracle_alpha_pixels":pixels,"oracle_mask_cosine":cosine(visual,cache[q]),"decision":"NOT_MADE"})
    print(json.dumps({"items":len(m["items"]),"wall_seconds":time.monotonic()-start,
        "peak_vram_gib":torch.cuda.max_memory_allocated()/1024**3}))


def freeze():
    m=json.loads((DEST/"manifest.json").read_text());hashes={}
    for x in m["items"]:
        p=DEST/"scores"/(x["candidate_id"].replace(":","_")+".json");d=json.loads(p.read_text())
        if d["candidate_id"]!=x["candidate_id"] or d["oracle_alpha_pixels"]<=0 or not math.isfinite(d["oracle_mask_cosine"]):raise RuntimeError("invalid")
        hashes[str(p.relative_to(DEST))]=sha(p)
    write_once(DEST/"freeze_scores.json",{"status":"COMPLETE_ORACLE_RAW_NO_DECISION","manifest_sha256":sha(DEST/"manifest.json"),"score_file_hashes":hashes})
    print(json.dumps({"frozen":len(hashes)}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("prepare","run","freeze"));a=p.parse_args().action
    {"prepare":prepare,"run":run,"freeze":freeze}[a]()
