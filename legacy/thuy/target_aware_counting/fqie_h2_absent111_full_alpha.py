"""Full-box absent111 Alpha raw/generic evidence, resumable per query."""
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"target_aware_counting"))
from fqie_alpha_smoke import alpha_mask,cosine,load_model,normalized_text,sha,synchronized_transform,write_once

META=ROOT/"target_aware_counting/output/sink_target_selection_20261029_v1/fsc_verified_absent111.jsonl"
PRED=ROOT/"target_aware_counting/output/spatial_reseeding_fsc111_20261102_v1/predictions_absent111_m0.jsonl"
DEST=ROOT/"target_aware_counting/output/fqie_h2_alpha_absent111_full_20261104_v1"
META_SHA="161ada4aedaa81b028eccb7faa96329f746e39f1e081255b8cc1af9e8c61955e"
PRED_SHA="8ace7bae36346750e211303d983f554d66cd7f6c711eeb8540d3d23146c35573"


def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def safe(s):return "".join(c if c.isalnum() or c in "_-" else "_" for c in s)


def prepare():
    if sha(META)!=META_SHA or sha(PRED)!=PRED_SHA:raise RuntimeError("source changed")
    meta={x["item_id"]:x for x in rows(META)};items=[];total=0
    for r in rows(PRED):
        m=meta[r["item_id"]]
        if len(r["final_boxes"])!=len(r["final_scores"]) or r["count"]!=len(r["final_boxes"]):raise RuntimeError("binding")
        boxes=[]
        for score,box in zip(r["final_scores"],r["final_boxes"]):
            boxes.append({"candidate_id":box["source_id"],"native_score":score,"box_xyxy":box["bbox_xyxy_float"],"slot":box["slot"]})
        path=Path(m["image_path"]);total+=len(boxes)
        items.append({"item_id":r["item_id"],"query_id":r["query_id"],"source_image_id":r["source_image_id"],
            "query":r["query"],"image_path":str(path),"image_sha256":sha(path),"candidates":boxes})
    if len(items)!=111 or total!=6942:raise RuntimeError("denominator")
    write_once(DEST/"manifest.json",{"status":"LOCKED_BEFORE_FULL_INFERENCE","items":items,"total_candidates":total,
        "meta_sha256":META_SHA,"predictions_sha256":PRED_SHA,"view":"full_letterbox",
        "texts":["target","generic_object"],"inference_reads_labels":False})
    print(json.dumps({"items":len(items),"candidates":total,"manifest_sha256":sha(DEST/"manifest.json")}))


def run():
    import torch
    from PIL import Image
    m=json.loads((DEST/"manifest.json").read_text());torch.manual_seed(0);model,ac=load_model();res=int(model.visual.input_resolution);start=time.monotonic();done=0
    with torch.inference_mode():
        for item in m["items"]:
            out=DEST/"scores"/(safe(item["query_id"])+".json")
            if out.exists():
                d=json.loads(out.read_text());done+=len(d["candidates"]);continue
            image=Image.open(item["image_path"]).convert("RGB")
            if sha(Path(item["image_path"]))!=item["image_sha256"]:raise RuntimeError("image changed")
            text=model.encode_text(ac.tokenize([normalized_text(item["query"]),"a photo of an object."],truncate=True).cuda())
            values=[]
            for c in item["candidates"]:
                rt,at,pixels=synchronized_transform(image,alpha_mask(image.size,c["box_xyxy"]),res)
                visual=model.visual(rt.unsqueeze(0).to(model.dtype).cuda(),at.unsqueeze(0).to(model.dtype).cuda())
                raw=cosine(visual,text[0:1]);generic=cosine(visual,text[1:2])
                values.append({**c,"alpha_raw_full":raw,"alpha_generic_object":generic,
                    "alpha_query_minus_generic_object":raw-generic,"alpha_pixels":pixels})
            write_once(out,{"item_id":item["item_id"],"query_id":item["query_id"],"source_image_id":item["source_image_id"],
                "query":item["query"],"candidates":values,"decision":"NOT_MADE"});done+=len(values)
            print(json.dumps({"complete_query":item["query_id"],"candidates_done":done}),flush=True)
    print(json.dumps({"candidates":done,"wall_seconds":time.monotonic()-start,
        "peak_vram_gib":torch.cuda.max_memory_allocated()/1024**3}))


def freeze():
    m=json.loads((DEST/"manifest.json").read_text());hashes={};total=0
    for item in m["items"]:
        p=DEST/"scores"/(safe(item["query_id"])+".json");d=json.loads(p.read_text())
        if d["item_id"]!=item["item_id"] or len(d["candidates"])!=len(item["candidates"]):raise RuntimeError("binding")
        for x in d["candidates"]:
            if x["alpha_pixels"]<=0 or not all(math.isfinite(x[k]) for k in ("alpha_raw_full","alpha_generic_object","alpha_query_minus_generic_object")):raise RuntimeError("invalid")
        total+=len(d["candidates"]);hashes[str(p.relative_to(DEST))]=sha(p)
    if total!=6942:raise RuntimeError("incomplete")
    write_once(DEST/"freeze_scores.json",{"status":"COMPLETE_FULL_RAW_NO_DECISION","manifest_sha256":sha(DEST/"manifest.json"),
        "candidates":total,"score_file_hashes":hashes});print(json.dumps({"frozen_queries":len(hashes),"candidates":total}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("prepare","run","freeze"));a=p.parse_args().action
    {"prepare":prepare,"run":run,"freeze":freeze}[a]()
