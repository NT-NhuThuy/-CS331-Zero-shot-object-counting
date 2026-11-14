"""Extract DINOv2 register patch atoms once per source image/reference.

This requires an explicit local official repo and checkpoint. It never
downloads weights or chooses a different backbone automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import file_sha256, stable_hash
from .predict import rows
from .spatial_features import (extract_image_atoms, extract_reference_atoms,
                               load_atom_cache, load_local_dinov2_registers,
                               save_atom_cache)


def _cache(path: Path, image_sha: str, feature: dict, extractor) -> None:
    if path.exists():
        load_atom_cache(path,image_sha,feature)
        return
    save_atom_cache(path,extractor(),image_sha,feature)


def prepare(manifests: list[Path], gallery: Path, feature_config: Path,
            official_repo: Path, checkpoint: Path, checkpoint_sha256: str,
            cache_dir: Path, item_index: Path, gallery_out: Path,
            device: str) -> dict:
    if item_index.exists() or gallery_out.exists():
        raise FileExistsError("feature index or featured gallery already exists")
    feature = json.loads(feature_config.read_text())
    if feature.get("backbone") != "dinov2_vitb14_reg" or feature.get("patch_size") != 14 or \
            feature.get("layer") != "x_norm_patchtokens" or \
            feature.get("view_policy") not in ("full","four_core_tiles") or \
            int(feature.get("long_edge_pixels",0)) <= 0:
        raise ValueError("feature protocol is not locked to DINOv2 registers patch tokens")
    items = [item for manifest in manifests for item in rows(manifest)]
    if len({r["item_id"] for r in items}) != len(items):
        raise ValueError("duplicate source item across feature manifests")
    gallery_rows = rows(gallery)
    model = load_local_dinov2_registers(official_repo,checkpoint,checkpoint_sha256,device)
    cache_dir.mkdir(parents=True,exist_ok=True)
    by_image = {}
    index = []
    feature_hash = stable_hash(feature)
    for item in items:
        image_path = Path(item["image_path"])
        image_sha = file_sha256(image_path)
        key = (str(image_path.resolve()),image_sha)
        if key not in by_image:
            cache_path = cache_dir/f"source_{image_sha[:20]}_{feature_hash[:12]}.npz"
            _cache(cache_path,image_sha,feature,
                   lambda:extract_image_atoms(image_path,model,feature))
            by_image[key] = cache_path
        index.append({"item_id":item["item_id"],"source_image_id":item["source_image_id"],
                      "query":item["query"],"image_sha256":image_sha,
                      "atom_cache_path":str(by_image[key])})
    featured = []
    for record in gallery_rows:
        row = dict(record)
        if row["status"] == "valid":
            if not row.get("weak_roi_xyxy") or file_sha256(row["image_path"]) != row["image_sha256"]:
                raise ValueError("valid reference ROI/image hash missing or changed")
            cache_path = cache_dir/f"reference_{row['reference_id']}_{feature_hash[:12]}.npz"
            _cache(cache_path,row["image_sha256"],feature,
                   lambda r=row:extract_reference_atoms(r["image_path"],
                                                         tuple(r["weak_roi_xyxy"]),model,feature))
            row["atom_cache_path"] = str(cache_path)
        featured.append(row)
    with item_index.open("x") as stream:
        for row in index:
            stream.write(json.dumps(row,sort_keys=True)+"\n")
    with gallery_out.open("x") as stream:
        for row in featured:
            stream.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
    return {"items":len(index),"source_images":len(by_image),
            "valid_references":sum(r["status"] == "valid" for r in featured),
            "feature_config_sha256":feature_hash,"item_index_sha256":file_sha256(item_index),
            "featured_gallery_sha256":file_sha256(gallery_out)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",action="append",required=True,type=Path)
    for name in ("gallery","feature-config","official-repo","checkpoint",
                 "cache-dir","item-index","gallery-out"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--checkpoint-sha256",required=True)
    parser.add_argument("--device",required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest,args.gallery,args.feature_config,args.official_repo,
                             args.checkpoint,args.checkpoint_sha256,args.cache_dir,args.item_index,
                             args.gallery_out,args.device),sort_keys=True))


if __name__ == "__main__":
    main()
