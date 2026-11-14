"""Build a fixed local reference snapshot with weak CountGD M0 ROI provenance.

The provider manifest is an archived, preordered retrieval result. This module
never searches the web during prediction or silently substitutes benchmark art.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .config import file_sha256, stable_hash


def _rows(path: str | Path) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _term(value: str) -> str:
    return " ".join(value.strip().split()).rstrip(" .").casefold()


def perceptual_dhash(path: str | Path) -> int:
    with Image.open(path) as opened:
        image = opened.convert("L").resize((9,8),Image.Resampling.LANCZOS)
        pixels = np.asarray(image,dtype=np.int16)
    bits = pixels[:,1:] > pixels[:,:-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def _eval_fingerprints(manifests: list[str | Path]) -> tuple[set[str],list[int]]:
    paths = {Path(row["image_path"]) for manifest in manifests for row in _rows(manifest)}
    return ({file_sha256(path) for path in paths},
            [perceptual_dhash(path) for path in paths])


def _read_query_bank(path: str | Path) -> dict[str,dict]:
    bank = {}
    for item in _rows(path):
        if {"gt_count","gt_boxes","query_status","presence_label","annotations","labels"} & item.keys():
            raise ValueError("query bank contains evaluation labels/status")
        query = item["canonical_query"]
        if query in bank or not query.strip():
            raise ValueError("duplicate/empty query-bank key")
        for competitor in item.get("competitors",[]):
            if competitor.get("relation") != "exclusive_same_level" or \
                    not competitor.get("query_only_source"):
                raise ValueError("competitor lacks query-only same-level exclusion provenance")
            if _term(competitor["term"]) in {_term(v) for v in item.get("aliases",[])} or \
                    _term(competitor["term"]) == _term(query):
                raise ValueError("synonym cannot be an excluding competitor")
        bank[query] = item
    return bank


def build_reference_bank(query_bank_path: str | Path, retrieval_config_path: str | Path,
                         output_path: str | Path, m0_predictions_path: str | Path | None = None) -> dict:
    """Validate archived local retrieval, exact/near leakage, and M0 one-box ROIs."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    queries = _read_query_bank(query_bank_path)
    config = json.loads(Path(retrieval_config_path).read_text())
    required = {"provider_manifest","provider_manifest_sha256","evaluation_manifests",
                "max_rank_per_term","min_valid_target","min_valid_competitor",
                "near_duplicate_hamming_max","broken_image_policy"}
    if not required <= set(config) or config["broken_image_policy"] != "invalidate":
        raise ValueError("retrieval configuration is not fully declared")
    if config["max_rank_per_term"] < 1 or config["min_valid_target"] < 1 or \
            config["min_valid_competitor"] < 0 or not (0 <= config["near_duplicate_hamming_max"] <= 64):
        raise ValueError("invalid retrieval count/duplicate policy")
    if file_sha256(config["provider_manifest"]) != config["provider_manifest_sha256"]:
        raise ValueError("provider manifest changed")
    eval_hashes, eval_dhashes = _eval_fingerprints(config["evaluation_manifests"])
    pred = {}
    if m0_predictions_path is not None:
        for row in _rows(m0_predictions_path):
            if row["variant"] != "M0" or row["item_id"] in pred:
                raise ValueError("invalid/duplicate M0 reference prediction")
            pred[row["item_id"]] = row
    source_rows = _rows(config["provider_manifest"])
    seen_provider = set()
    output = []
    for item in source_rows:
        if {"gt_count","gt_boxes","query_status","presence_label","annotations","labels"} & item.keys():
            raise ValueError("retrieval source contains evaluation labels/status")
        query, role, term = item["canonical_query"], item["semantic_role"], item["retrieval_term"]
        if query not in queries or role not in ("target","competitor"):
            raise ValueError("provider row has unknown query/semantic role")
        allowed = ({query,*queries[query].get("aliases",[])} if role == "target" else
                   {c["term"] for c in queries[query].get("competitors",[])})
        if _term(term) not in {_term(value) for value in allowed}:
            raise ValueError("provider term is not authorized by query-only bank")
        rank = int(item["retrieval_rank"])
        key = (query,role,term,rank)
        if key in seen_provider:
            raise ValueError("provider rank duplicated")
        seen_provider.add(key)
        if rank < 1 or rank > config["max_rank_per_term"]:
            continue
        for field in ("source_url","retrieved_at","license","provider","image_path"):
            if field not in item:
                raise ValueError(f"provider provenance missing {field}")
        path = Path(item["image_path"])
        record = dict(item)
        record["reference_id"] = stable_hash([query,role,term,rank,str(path)])[:20]
        record["image_sha256"] = None
        record["weak_roi_xyxy"] = None
        record["roi_source"] = None
        record["status"] = "valid"
        try:
            digest = file_sha256(path)
            dhash = perceptual_dhash(path)
            with Image.open(path) as opened:
                opened.verify()
            record["image_sha256"] = digest
            if digest in eval_hashes:
                record["status"] = "evaluation_exact_duplicate"
            elif any((dhash ^ other).bit_count() <= config["near_duplicate_hamming_max"]
                     for other in eval_dhashes):
                record["status"] = "evaluation_near_duplicate"
        except (OSError,ValueError):
            record["status"] = "broken_image"
        m0_id = f"reference:{record['reference_id']}"
        record["reference_m0_item_id"] = m0_id
        if record["status"] == "valid":
            if m0_predictions_path is None or m0_id not in pred:
                record["status"] = "roi_pending"
            else:
                prediction = pred[m0_id]
                if prediction["count"] != len(prediction["final_boxes"]):
                    raise ValueError("reference M0 count/box mismatch")
                if prediction["count"] != 1:
                    record["status"] = "roi_not_unique"
                else:
                    record["weak_roi_xyxy"] = prediction["final_boxes"][0]["bbox_xyxy_float"]
                    record["roi_source"] = "frozen_CountGD_M0_single_final_box_F2"
        output.append(record)
    if len({r["reference_id"] for r in output}) != len(output):
        raise ValueError("reference IDs collide")
    output.sort(key=lambda r:(r["canonical_query"],r["semantic_role"],
                              r["retrieval_term"],int(r["retrieval_rank"])))
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open("x") as stream:
        for item in output:
            stream.write(json.dumps(item,sort_keys=True,ensure_ascii=False)+"\n")
    coverage = {}
    for query in queries:
        counts = defaultdict(int)
        for record in output:
            if record["canonical_query"] == query and record["status"] == "valid":
                counts[record["semantic_role"]] += 1
        coverage[query] = {"target":counts["target"],"competitor":counts["competitor"],
                           "reference_available":counts["target"] >= config["min_valid_target"] and
                           counts["competitor"] >= config["min_valid_competitor"]}
    return {"queries":len(queries),"references":len(output),"valid":sum(r["status"] == "valid" for r in output),
            "coverage":coverage,"snapshot_sha256":file_sha256(output_path)}


def make_reference_m0_manifest(references_path: str | Path, output_path: str | Path) -> int:
    """Feed archived images to the existing CountGD M0 runner before finalizing ROIs."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    rows = []
    for item in _rows(references_path):
        if item["status"] != "roi_pending":
            continue
        rows.append({"item_id":item["reference_m0_item_id"],
                     "query_id":item["reference_id"],
                     "source_image_id":item["reference_id"],
                     "image_path":item["image_path"],
                     "query":item["retrieval_term"].strip(" .")+" ."})
    with output_path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row,sort_keys=True)+"\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode",required=True)
    build = sub.add_parser("build")
    build.add_argument("--query-bank",required=True,type=Path)
    build.add_argument("--retrieval-config",required=True,type=Path)
    build.add_argument("--out",required=True,type=Path)
    build.add_argument("--m0-predictions",type=Path)
    manifest = sub.add_parser("reference-m0-manifest")
    manifest.add_argument("--references",required=True,type=Path)
    manifest.add_argument("--out",required=True,type=Path)
    args = parser.parse_args()
    if args.mode == "build":
        print(json.dumps(build_reference_bank(args.query_bank,args.retrieval_config,args.out,
                                              args.m0_predictions),sort_keys=True))
    else:
        print(json.dumps({"items":make_reference_m0_manifest(args.references,args.out)}))


if __name__ == "__main__":
    main()
