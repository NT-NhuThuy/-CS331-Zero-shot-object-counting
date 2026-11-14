"""Explicit configuration contract; numerical calibration is never implicit."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


@dataclass(frozen=True)
class NumericalConfig:
    gamma: float
    rho: float
    opening_cost: float
    retention_fraction: float
    nms_iou: float
    match_max_iterations: int
    match_tolerance: float
    milp_relative_gap: float = 0.0
    max_atoms_per_side: int | None = None

    def __post_init__(self) -> None:
        if self.gamma < 0 or self.rho <= 0 or self.opening_cost <= 0:
            raise ValueError("gamma must be nonnegative; rho and opening cost must be positive")
        if not (0 < self.retention_fraction <= 1) or not (0 <= self.nms_iou <= 1):
            raise ValueError("retention fraction or NMS IoU outside [0,1]")
        if self.match_max_iterations < 1 or self.match_tolerance <= 0 or self.milp_relative_gap < 0:
            raise ValueError("invalid solver budget/tolerance")
        if self.max_atoms_per_side is not None and self.max_atoms_per_side < 1:
            raise ValueError("max_atoms_per_side must be positive when set")


def load_locked_config(path: str | Path) -> tuple[dict, str]:
    path = Path(path)
    document = json.loads(path.read_text())
    if document.get("status") != "protocol_locked":
        raise ValueError("calibration/gallery/feature policy has not been locked")
    for field in ("gallery_snapshot", "gallery_sha256", "model_checkpoint", "model_sha256",
                  "feature", "numerical", "candidate_pool", "source_split",
                  "reference_policy"):
        if field not in document:
            raise ValueError(f"missing locked config field {field}")
    for path_field, hash_field in (("gallery_snapshot", "gallery_sha256"),
                                   ("model_checkpoint", "model_sha256")):
        if file_sha256(document[path_field]) != document[hash_field]:
            raise ValueError(f"{path_field} hash changed")
    NumericalConfig(**document["numerical"])
    feature = document["feature"]
    if feature.get("backbone") != "dinov2_vitb14_reg" or feature.get("patch_size") != 14:
        raise ValueError("locked DINOv2 registers backbone/patch size changed")
    if feature.get("layer") != "x_norm_patchtokens" or not feature.get("long_edge_pixels") or \
            feature.get("view_policy") not in ("full", "four_core_tiles"):
        raise ValueError("feature layer, resolution or view policy is unbound")
    reference_policy = document["reference_policy"]
    if int(reference_policy.get("min_valid_target",0)) < 1 or \
            int(reference_policy.get("min_valid_competitor",-1)) < 0:
        raise ValueError("reference coverage policy is unbound")
    return document, stable_hash(document)
