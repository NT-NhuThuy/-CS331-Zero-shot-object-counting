"""One online method: CountGD++ native -> tiles -> router -> Qwen -> identity -> count.

This module requires local checkpoints (not shipped). Selection is explicit:
single-pass diagnostic or pinned query-only pseudo=3. Neither certifies V14
parity, SOTA or a paper result without its own raw-model validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from countlab.method.schema import parse_input
from countlab.method.router import select_for_verification
from countlab.method.pipeline import run_method
from .countgd_native import NativeCountGD, PROTOCOL_ID
from .countgd_official import OfficialCountGD, PROTOCOL_ID as OFFICIAL_PROTOCOL_ID
from .recovery import recover_from_native
from .qwen_unary import QwenRegionUnary, verify_identity_pairs


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for chunk in iter(lambda: fd.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def execute_online(*, image_path: Path, query: str, image_id: str, query_id: str,
                   config: dict, backend: Any | None = None, verifier: Any | None = None) -> tuple[dict, dict]:
    """Return (GT-free frozen-input manifest, unified output).

    Injection is for CPU contract tests; production provider loads real native
    CountGD++ and Qwen weights.  Identity edges remain UNKNOWN unless produced
    by an independently verified pairwise backend. That is an explicit gap.
    """
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    protocol_id = config.get("protocol_id")
    if protocol_id not in (PROTOCOL_ID, OFFICIAL_PROTOCOL_ID):
        raise ValueError("CountGD++ protocol id must be explicit")
    official = protocol_id == OFFICIAL_PROTOCOL_ID
    if official:
        from .countgd_official import validate_protocol
        from countlab.method.schema import _validate_no_gt
        _validate_no_gt(config)
        validate_protocol(config.get("official_countgd", {}))
        if backend is not None and getattr(backend, "protocol_id", None) != OFFICIAL_PROTOCOL_ID:
            raise ValueError("injected backend does not match official protocol")
        if backend is not None and getattr(backend, "config", {}).get("protocol_sha256") != config["official_countgd"]["protocol_sha256"]:
            raise ValueError("official backend protocol hash differs from requested config")
    provider = backend if backend is not None else (
        OfficialCountGD(config["official_countgd"]) if official else NativeCountGD(config["native_countgd"]))
    base = provider.infer(image_path, query, role="baseline", view_id="full", image_id=image_id)
    recovery_config = config["recovery"]
    rescues = []
    if len(base) <= int(recovery_config["trigger_baseline_le"]):
        rescues = recover_from_native(image_path, query, provider,
                                      grids=tuple(recovery_config["grids"]),
                                      overlap=float(recovery_config["overlap"]),
                                      max_tiles=int(recovery_config["max_tiles"]), image_id=image_id)
    manifest = {"schema":"countlab_candidate_pool_v1", "image_id":image_id, "query_id":query_id,
                "query_text":query, "baseline_count":len(base), "baseline":base,
                "rescue":rescues, "identity_edges":[],
                "metadata": {"backend_protocol":protocol_id,
                             "image_sha256":file_sha(image_path),
                             "backend_provenance":getattr(provider,"provenance", {"injected_test_double":True}),
                             "protocol_limits": ("Query-only pseudo=3 full-image core; HOLD for V14 parity" if official else
                                                 "NO pseudo exemplars; identity edges not yet produced online")}}
    if official:
        manifest["metadata"]["detector_runs"] = list(provider.runs)
    # Optional exact-key *FSC exploratory* adapter from Thư. Never silently
    # transfer its source-specific frozen node features onto LVIS/other data.
    thu_config=config.get('thu_frozen_features')
    if thu_config:
        from .thu_features import attach_features
        map_path=Path(thu_config['candidate_map_path'])
        mapping=json.loads(map_path.read_text(encoding='utf-8'))
        cal_path=thu_config.get('calibration_path')
        calibration=json.loads(Path(cal_path).read_text(encoding='utf-8')) if cal_path else None
        manifest=attach_features(manifest,Path(thu_config['npz_path']),mapping,calibration)
    baseline, rescue, _ = parse_input(manifest)
    chosen = select_for_verification(baseline+rescue, budget=config.get("verification_budget"))
    if chosen and verifier is None:
        qcfg = config.get("qwen")
        if qcfg is None:
            raise RuntimeError("Qwen verifier is required for selected candidates; use verification_budget=0 for baseline-only")
        verifier = QwenRegionUnary(**qcfg)
    for role in ("baseline", "rescue"):
        for item in manifest[role]:
            if item["candidate_id"] not in chosen:
                continue
            p = next(p for p in baseline+rescue if p.candidate_id == item["candidate_id"])
            sem, whole = verifier.verdict(image_path, query, p)
            if sem not in ("YES","NO","UNKNOWN") or whole not in ("YES","NO","UNKNOWN"):
                raise ValueError("verifier returned invalid verdict")
            item["evidence"] = {"semantic":sem, "whole_instance":whole}
            item["evidence_refs"].append(f"Qwen_region:{item['candidate_id']}")
    # Optional Lộc Qwen pairwise identity evidence, required to form SAME edges.
    identity_cfg=config.get('identity',{})
    if identity_cfg.get('enabled',False):
        if verifier is None:
            raise RuntimeError('pairwise verification needs a Qwen verifier backend')
        parsed_base,parsed_rescue,_=parse_input(manifest)
        unary={p.candidate_id:(p.semantic,p.whole) for p in parsed_rescue}
        manifest['identity_edges']=verify_identity_pairs(image_path,parsed_rescue,unary,verifier,
                           max_pairs=int(identity_cfg['max_pairs']),
                           min_overlap=float(identity_cfg.get('min_overlap',.1)))
    manifest["metadata"]["verifier"] = {"model_id":getattr(verifier,"model_id", "none"),
                                           "revision":getattr(verifier,"revision", None),
                                           "calls":getattr(verifier,"calls", 0),
                                           "unknown_or_failed":getattr(verifier,"fallbacks", 0)}
    # UNKNOWN edges never create a SAME relation; identity prompts are not oracle labels.
    parse_input(manifest)
    result = run_method(manifest, budget=config.get("verification_budget"),
                        abstain_on_conflict=bool(config.get("abstain_on_conflict",True)),
                        policy=config["method_policy"])
    result["limitations"].append("Official query-only pseudo=3 adapter: HOLD for V14 parity" if official else
                                  "Online native adapter uses no pseudo-exemplars and does not reproduce V14")
    if not identity_cfg.get("enabled", False):
        result["limitations"].append("No online pairwise physical identity verifier is attached")
    else:
        result["limitations"].append("Pairwise Qwen edges are uncalibrated and not GT-validated")
    contract_test = official and provider.provenance.get("evidence_level") == "SOFTWARE_CONTRACT_TEST"
    result["evidence_trace"].append({"stage":"software_contract_test" if contract_test else "actual_inference",
                                      "native_protocol":protocol_id,
                                      "verifier_calls":getattr(verifier,"calls",0),
                                      "tile_proposals":len(rescues)})
    return manifest, result


def main():
    ap = argparse.ArgumentParser(description="UNICOUNT native online inference (requires local CountGD++/Qwen weights)")
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--query", required=True)
    ap.add_argument("--query-id", required=True)
    ap.add_argument("--image-id", required=True)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    if a.out_dir.exists():
        raise FileExistsError("create-only run directory required")
    cfg = json.loads(a.config.read_text())
    manifest, output = execute_online(image_path=a.image, query=a.query, image_id=a.image_id,
                                      query_id=a.query_id, config=cfg)
    a.out_dir.mkdir(parents=True, exist_ok=False)
    for name, data in (("frozen_candidates.json",manifest),("prediction.json",output)):
        with (a.out_dir/name).open("x",encoding="utf-8") as fd:
            json.dump(data,fd,ensure_ascii=False,indent=2,sort_keys=True)
            fd.write("\n")
    print(json.dumps({"count":output["count"],"status":output["count_status"],
                      "source":manifest["metadata"]["backend_protocol"],"output":str(a.out_dir)},ensure_ascii=False))

if __name__ == "__main__":
    main()
