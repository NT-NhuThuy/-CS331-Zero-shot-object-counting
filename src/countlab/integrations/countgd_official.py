"""Pinned query-only CountGD++ two-pass bridge. NOT a V14 parity certificate.

The positive pseudo-exemplar selector is loaded from the pinned upstream
engine as a single function, without its GT-reading evaluator or imports.
PrACo's synthetic positive/negative adapter is a DIFFERENT operating point.
No torch import, download or model execution happens during preflight.
"""
from __future__ import annotations

import ast
import copy
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

from countlab.method.schema import _validate_no_gt
from .countgd_native import normalized_box_to_xyxy, _tolist

PROTOCOL_ID = "CGD_OFFICIAL_QUERY_ONLY_PSEUDO3_V1"
SOURCE_REPO = "https://github.com/niki-amini-naieni/CountGDPlusPlus"
SOURCE_COMMIT = "2ecbc4331d6730d017f52a4a97bbcbae7d5d714a"
SOURCE_BLOBS = {
    "engine_inference.py": "32e2cecc2648975d3252985a18b86eed332d4f62",
    "cfg_app.py": "d9ce134c0a2c3f9d864b5c39f9407b4068e4a685",
    "datasets/transforms_app.py": "e4ac7ac8afde6e7a3d3d9746dbd87d2bde2e6879",
    "util/misc.py": "4a3d759dea39fe8ad01f04d2bb19842b995f53e3",
    "models/GroundingDINO/groundingdino_app.py": "710c95f0e4d6cbebbd5dcc29c1c72ecec3369586",
    "models/GroundingDINO/ops/src/cuda/ms_deform_attn_cuda.cu": "d6d583647cce987196d5ad1968a8a365a379e774",
}
OPERATING_POINT = {
    "num_exemplars": 0, "num_pseudo_exemplars": 3,
    "caption_template": "{query} .", "resize_short": 800, "resize_max": 1333,
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "box_threshold": 0.23, "text_threshold": 0.0, "seed": 42,
    "precision": "float32", "amp": False, "crop": False, "tt_norm": False,
    "batch_size": 1, "use_synth_exemplars": False, "negative_prompts": [],
    "query_handling": "verbatim_no_singularization", "nms": False,
    "pseudo_selection": "upstream_get_pseudo_exemplars_max_sigmoid_torch_sort_descending",
    "final_filter": "max_sigmoid_gt_box_threshold_and_all_content_tokens_gt_text_threshold",
}
LIBRARIES = ("torch", "torchvision", "transformers", "timm", "numpy", "Pillow")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_protocol(config: dict) -> dict:
    """A changed operating point requires a new adapter/protocol, not a rename."""
    if not isinstance(config, dict) or not isinstance(config.get("protocol"), dict):
        raise ValueError("official config/protocol must be objects")
    _validate_no_gt(config)
    p = config.get("protocol", {})
    if (p.get("adapter") != PROTOCOL_ID or p.get("source_repo") != SOURCE_REPO
            or p.get("source_commit") != SOURCE_COMMIT
            or p.get("source_git_blobs") != SOURCE_BLOBS
            or p.get("operating_point") != OPERATING_POINT):
        raise ValueError("official protocol/operating point/source mismatch")
    if config.get("protocol_sha256") != canonical_sha(p):
        raise ValueError("protocol_sha256 mismatch; freeze the complete protocol first")
    for field in ("checkpoint_sha256", "cuda_extension_sha256"):
        if not _digest(p.get(field)):
            raise ValueError(f"{field}: exact frozen asset SHA256 required")
    bert = p.get("bert_files_sha256", {})
    if (not isinstance(bert, dict) or not {"config.json", "vocab.txt"} <= bert.keys()
            or not any(n.endswith((".bin", ".safetensors")) for n in bert)):
        raise ValueError("BERT config, tokenizer and weights must be hash-locked")
    for name, value in bert.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name or not _digest(value):
            raise ValueError("invalid BERT asset path/hash")
    versions = p.get("library_versions", {})
    if set(versions) != set(LIBRARIES) or any(not isinstance(v, str) or not v or v.startswith("FILL") for v in versions.values()):
        raise ValueError("exact library_versions must be frozen before loading")
    if not isinstance(p.get("cuda_runtime"), str) or not p["cuda_runtime"] or p["cuda_runtime"].startswith("FILL"):
        raise ValueError("exact cuda_runtime must be frozen")
    flags = p.get("runtime_flags", {})
    if set(flags) != {"matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark", "deterministic_algorithms"} or any(type(v) is not bool for v in flags.values()):
        raise ValueError("explicit runtime_flags must be frozen")
    if config.get("device") != "cuda:0":
        raise ValueError("this operating point requires explicit device=cuda:0")
    return p


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                   stderr=subprocess.PIPE, encoding="utf-8").strip()


def inspect_assets(config: dict) -> dict:
    """Fail closed on incomplete locks; never load torch or modify upstream."""
    if not isinstance(config, dict):
        return {"status": "BLOCKED", "protocol_id": PROTOCOL_ID,
                "blockers": ["official config must be an object"], "verified": [],
                "parity_status": "HOLD for V14 parity"}
    blockers: list[str] = []
    verified: list[str] = []
    try:
        p = validate_protocol(config)
    except (ValueError, TypeError) as e:
        blockers.append(str(e))
        p = None
    paths = {}
    for field in ("repo", "checkpoint", "bert_dir"):
        value = config.get(field)
        path = Path(value) if isinstance(value, str) and value else None
        valid = path is not None and (path.is_file() if field == "checkpoint" else path.is_dir())
        if not valid:
            blockers.append(f"{field}: local asset missing")
        else:
            paths[field] = path.resolve()
    if p is not None:
        repo = paths.get("repo")
        if repo:
            try:
                if _git(repo, "rev-parse", "HEAD") != p["source_commit"]:
                    blockers.append("upstream source commit mismatch")
                if _git(repo, "status", "--porcelain", "--untracked-files=no"):
                    blockers.append("upstream has tracked modifications; no silent CUDA/source patch")
                for name, expected in p["source_git_blobs"].items():
                    data = (repo / name).read_bytes()
                    actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                    if actual != expected:
                        blockers.append(f"upstream source bytes mismatch: {name}")
                verified.append("upstream commit and selected source files inspected")
            except (OSError, subprocess.SubprocessError) as e:
                blockers.append(f"upstream unavailable: {type(e).__name__}")
        checkpoint = paths.get("checkpoint")
        if checkpoint:
            if file_sha(checkpoint) != p["checkpoint_sha256"]:
                blockers.append("checkpoint SHA256 mismatch")
            else:
                verified.append("checkpoint SHA256 matches lock")
        bert = paths.get("bert_dir")
        if bert:
            actual_files = {q.relative_to(bert).as_posix() for q in bert.rglob("*") if q.is_file()}
            if actual_files != set(p["bert_files_sha256"]):
                blockers.append("BERT directory file inventory differs from lock")
            for name, expected in p["bert_files_sha256"].items():
                target = (bert / name).resolve()
                if not target.is_relative_to(bert) or not target.is_file() or file_sha(target) != expected:
                    blockers.append(f"BERT asset missing/hash mismatch: {name}")
        for package, expected in p["library_versions"].items():
            try:
                if importlib.metadata.version(package) != expected:
                    blockers.append(f"library version mismatch: {package}")
            except importlib.metadata.PackageNotFoundError:
                blockers.append(f"Python dependency missing: {package}")
        try:
            spec = importlib.util.find_spec("MultiScaleDeformableAttention")
            if spec is None or not spec.origin or file_sha(Path(spec.origin)) != p["cuda_extension_sha256"]:
                blockers.append("CUDA extension missing/hash mismatch")
        except (ImportError, ValueError, OSError):
            blockers.append("CUDA extension unavailable")
    return {"status": "BLOCKED" if blockers else "READY_FOR_GPU_SMOKE_NOT_MODEL_PARITY",
            "protocol_id": PROTOCOL_ID, "blockers": blockers, "verified": verified,
            "parity_status": "HOLD for V14 parity",
            "note": "Source/asset checks do not establish GPU or V14 count/box parity"}


def validate_prediction(raw: dict) -> tuple[list, list, int]:
    _validate_no_gt(raw)
    try:
        boxes = _tolist(raw["boxes_cxcywh_normalized"])
        probabilities = _tolist(raw["token_probabilities"])
        ids = _tolist(raw["input_ids"])
    except (KeyError, TypeError) as e:
        raise ValueError("raw detector boxes/token probabilities required; count-only is outside v1") from e
    if len(boxes) != len(probabilities) or 1012 not in ids or ids.index(1012) <= 1:
        raise ValueError("unaligned prediction or missing BERT content/period separator")
    for box, scores in zip(boxes, probabilities):
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError("nonfinite/invalid model box")
        if len(scores) < len(ids) or not all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in scores):
            raise ValueError("nonfinite/invalid token probabilities")
    return boxes, probabilities, ids.index(1012)


def candidates_from_prediction(raw: dict, *, width: int, height: int, namespace: str,
                               role: str, view_id: str, offset=(0., 0.)) -> list[dict]:
    if role not in ("baseline", "rescue") or not view_id:
        raise ValueError("invalid role/view")
    if (not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0
            or len(offset) != 2 or not all(math.isfinite(float(v)) and v >= 0 for v in offset)):
        raise ValueError("invalid image size/offset")
    boxes, probabilities, end = validate_prediction(raw)
    out = []
    for index, (box, scores) in enumerate(zip(boxes, probabilities)):
        confidence = max(map(float, scores))
        if confidence <= OPERATING_POINT["box_threshold"] or not all(float(v) > OPERATING_POINT["text_threshold"] for v in scores[1:end]):
            continue
        if box[2] <= 0 or box[3] <= 0:
            raise ValueError("selected detection has no physical box; cannot invent a v1 proposal")
        region = normalized_box_to_xyxy(box, width, height, offset)
        if region is None:
            raise ValueError("selected detection clips to an empty box; count parity cannot be assumed")
        cid = "cgd-pseudo3:" + canonical_sha([namespace, role, view_id, list(offset), index])
        out.append({"candidate_id": cid, "view_id": view_id, "box_xyxy": region,
                    "confidence": confidence, "native_query_index": index,
                    "view_provenance": {"original_size": [width, height], "offset": list(offset),
                                        "raw_box_cxcywh_normalized": list(box)},
                    "evidence_refs": [f"{PROTOCOL_ID}:{namespace}:forward2:{index}"],
                    "evidence": {"semantic": "UNKNOWN", "whole_instance": "UNKNOWN"}})
    return out


class OfficialCountGD:
    """Two detector calls, even for zero first-stage proposals. No count replay."""
    protocol_id = PROTOCOL_ID

    def __init__(self, config: dict, *, engine: Any = None):
        self.config = copy.deepcopy(config)
        p = validate_protocol(self.config)
        if engine is None:
            readiness = inspect_assets(self.config)
            if readiness["blockers"]:
                raise RuntimeError("Official CountGD++ assets BLOCKED: " + "; ".join(readiness["blockers"]))
            engine = _TorchOfficialEngine(self.config)
        self.engine = engine
        self.runs: list[dict] = []
        self.provenance = {
            "adapter": PROTOCOL_ID, "protocol": p,
            "protocol_sha256": self.config["protocol_sha256"],
            "adapter_source_sha256": file_sha(Path(__file__)),
            "evidence_level": "SOFTWARE_CONTRACT_TEST" if not isinstance(engine, _TorchOfficialEngine) else "GPU_PATH_UNVALIDATED",
            "parity_status": "HOLD for V14 parity",
            "runtime": getattr(engine, "runtime", {"kind": "injected_test_double"}),
            "asset_validation": "VERIFIED" if isinstance(engine, _TorchOfficialEngine) else "NOT_CHECKED_TEST_DOUBLE",
        }

    def infer(self, image_path: Path, query: str, *, role="baseline", view_id="full",
              offset=(0., 0.), image_id="") -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("nonempty query required")
        from PIL import Image
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        caption = OPERATING_POINT["caption_template"].format(query=query)
        namespace = canonical_sha([self.config["protocol_sha256"], file_sha(image_path), image_id, query])
        start = time.perf_counter()
        scope = self.engine.seeded() if hasattr(self.engine, "seeded") else nullcontext()
        with scope:
            self.engine.prepare(image_path)
            first = self.engine.forward(caption, [])
            first_boxes, _, _ = validate_prediction(first)
            pseudo = _tolist(self.engine.select_pseudo(first, num_exemplars=3))
            resized_width, resized_height = self.engine.resized_size
            if len(pseudo) > 3 or any(len(b) != 4 or not all(math.isfinite(float(v)) for v in b)
                                      or not 0 <= b[0] < b[2] <= resized_width
                                      or not 0 <= b[1] < b[3] <= resized_height for b in pseudo):
                raise ValueError("invalid pseudo-exemplar selection")
            second = self.engine.forward(caption, pseudo)
            candidates = candidates_from_prediction(second, width=width, height=height,
                                                    namespace=namespace, role=role, view_id=view_id, offset=offset)
            profile = self.engine.finish_profile() if hasattr(self.engine, "finish_profile") else {"peak_vram_bytes": None}
        self.runs.append({"image_sha256": file_sha(image_path), "view_id": view_id,
                          "original_size": [width, height],
                          "resized_size": list(self.engine.resized_size),
                          "caption_sha256": hashlib.sha256(caption.encode()).hexdigest(),
                          "first_stage_raw_sha256": canonical_sha(first),
                          "second_stage_raw_sha256": canonical_sha(second),
                          "first_stage_queries": len(first_boxes), "initial_exemplars": 0,
                          "pseudo_exemplars_resized_xyxy": pseudo,
                          "second_stage_queries": len(second["boxes_cxcywh_normalized"]),
                          "detector_calls": 2, "output_candidates": len(candidates),
                          "elapsed_sec": time.perf_counter() - start, **profile})
        return candidates


class _TorchOfficialEngine:
    """Reuse the audited loader; execute the upstream positive pseudo selector.

    No import of engine_inference itself: that evaluator reads annotations.
    This bridge passes only an image, caption and self-generated boxes.
    """
    def __init__(self, config: dict):
        import torch
        import sys
        self.torch = torch
        self.device = torch.device(config["device"])
        if not torch.cuda.is_available() or torch.version.cuda != config["protocol"]["cuda_runtime"]:
            raise RuntimeError("CUDA unavailable/runtime differs from frozen lock")
        actual_flags = {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        if actual_flags != config["protocol"]["runtime_flags"]:
            raise RuntimeError("runtime flags differ from lock; refusing to change caller settings")
        original = Path(__file__).resolve().parents[3] / "legacy/thuy/src"
        sys.path.insert(0, str(original))
        from zoc.backends.countgd_grounding.backend import CountGDBackend
        # Restore caller RNG state: loading must not change another stage's seed.
        with self.seeded():
            self.backend = CountGDBackend({"repo": str(Path(config["repo"]).resolve()),
                "checkpoint": str(Path(config["checkpoint"]).resolve()),
                "bert_dir": str(Path(config["bert_dir"]).resolve()), "device": config["device"],
                "resize_short": 800, "resize_max": 1333, "feature_levels": [0, 1],
                "grid_size": 7, "context_scale": 2.0, "text_threshold": 0.0})
        repo = Path(config["repo"]).resolve()
        for name, module in list(sys.modules.items()):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in ("models", "datasets", "util", "groundingdino")):
                location = getattr(module, "__file__", None)
                if location and not Path(location).resolve().is_relative_to(repo):
                    raise RuntimeError("upstream module namespace collision; use a fresh process")
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in self.backend.model.parameters()):
            raise RuntimeError("model precision differs from float32 lock")
        source = Path(config["repo"]) / "engine_inference.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_pseudo_exemplars"]
        if len(functions) != 1:
            raise RuntimeError("pinned upstream pseudo selector missing")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        self.selector = namespace["get_pseudo_exemplars"]
        self.runtime = {"versions": config["protocol"]["library_versions"], "cuda": torch.version.cuda,
                        "device": str(self.device), "gpu_name": torch.cuda.get_device_name(self.device),
                        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "source_sha256": {n: file_sha(Path(config["repo"]) / n) for n in SOURCE_BLOBS},
                        "loader_source_sha256": file_sha(original / "zoc/backends/countgd_grounding/backend.py")}

    @contextmanager
    def seeded(self):
        """Pin this inference's RNGs and restore the caller, including failures."""
        import numpy as np
        import random
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            with self.torch.random.fork_rng(devices=[self.device.index]):
                self.torch.random.default_generator.manual_seed(42)
                self.torch.cuda.default_generators[self.device.index].manual_seed(42)
                random.seed(42); np.random.seed(42)
                yield
        finally:
            random.setstate(py_state); np.random.set_state(np_state)

    def prepare(self, path: Path):
        torch = self.torch
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.tensor, self.original_size = self.backend._input(str(path))
        self.resized_size = (self.tensor.shape[2], self.tensor.shape[1])
        self.nested = self.backend.nested_fn([self.tensor])

    def forward(self, caption: str, exemplars: list) -> dict:
        torch = self.torch
        self.caption = caption
        boxes = torch.tensor(exemplars, dtype=torch.float32, device=self.device).reshape(-1, 4)
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            self.output = self.backend.model(self.nested, self.nested, [boxes], [], [], captions=[caption])
        token = self.output.get("token")
        if token is None:
            raise RuntimeError("pinned official detector token output required")
        return {"boxes_cxcywh_normalized": _tolist(self.output["pred_boxes"][0]),
                "token_probabilities": _tolist(self.output["pred_logits"][0].sigmoid()),
                "input_ids": _tolist(token["input_ids"][0])}

    def select_pseudo(self, raw: dict, *, num_exemplars: int):
        torch = self.torch
        with torch.cuda.device(self.device), torch.inference_mode():
            result = self.selector(self.output, [(self.tensor.shape[1], self.tensor.shape[2])],
                                   [torch.tensor([0], device=self.device)], [[self.caption]],
                                   [self.caption], 0.23, num_exemplars=num_exemplars)
        return _tolist(result[0])

    def finish_profile(self):
        self.torch.cuda.synchronize(self.device)
        return {"peak_vram_bytes": self.torch.cuda.max_memory_allocated(self.device)}
