"""Shared source-image DINO patch atoms with explicit full/core-tile geometry."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import file_sha256, stable_hash

DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ViewSpec:
    view_id: str
    source_rect: tuple[float, float, float, float]
    source_hw: tuple[int, int]
    core_index: int | None = None


@dataclass(frozen=True)
class Atom:
    atom_id: str
    footprint_xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float]
    view_id: str
    feature: tuple[float, ...]


def core_rect(source_hw: tuple[int, int], index: int) -> tuple[float, float, float, float]:
    h, w = source_hw
    if index not in range(4) or h <= 0 or w <= 0:
        raise ValueError("invalid four-core index or source size")
    xm, ym = w / 2, h / 2
    return ((0, 0, xm, ym), (xm, 0, w, ym),
            (0, ym, xm, h), (xm, ym, w, h))[index]


def four_core_views(source_hw: tuple[int, int], context_fraction: float) -> list[ViewSpec]:
    if not (0 <= context_fraction <= 1):
        raise ValueError("context fraction outside [0,1]")
    h, w = source_hw
    views = []
    for index in range(4):
        x0, y0, x1, y1 = core_rect(source_hw, index)
        dx, dy = (x1 - x0) * context_fraction, (y1 - y0) * context_fraction
        views.append(ViewSpec(f"tile{index}", (max(0, x0-dx), max(0, y0-dy),
                                                 min(w, x1+dx), min(h, y1+dy)), source_hw, index))
    return views


def full_view(source_hw: tuple[int, int]) -> ViewSpec:
    h, w = source_hw
    return ViewSpec("full", (0, 0, w, h), source_hw)


def prepare_letterbox(image: Image.Image, long_edge_pixels: int, patch_size: int = 14):
    if long_edge_pixels <= 0 or patch_size <= 0:
        raise ValueError("resolution/patch size must be positive")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("empty image")
    scale = long_edge_pixels / max(width, height)
    new_w, new_h = max(1, round(width*scale)), max(1, round(height*scale))
    new_w, new_h = min(new_w, long_edge_pixels), min(new_h, long_edge_pixels)
    canvas_w = math.ceil(new_w/patch_size)*patch_size
    canvas_h = math.ceil(new_h/patch_size)*patch_size
    resized = image.convert("RGB").resize((new_w, new_h), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (canvas_w, canvas_h),
                       tuple(round(255*c) for c in DINOV2_MEAN))
    canvas.paste(resized, (0, 0))
    pixels = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float().div_(255)
    mean = torch.tensor(DINOV2_MEAN)[:, None, None]
    std = torch.tensor(DINOV2_STD)[:, None, None]
    pixels = (pixels-mean)/std
    gh, gw = canvas_h//patch_size, canvas_w//patch_size
    valid = np.zeros((gh, gw), dtype=bool)
    for row in range(gh):
        for col in range(gw):
            valid[row, col] = ((col+0.5)*patch_size < new_w and
                               (row+0.5)*patch_size < new_h)
    return pixels, (new_h, new_w), valid


def atoms_from_grid(view: ViewSpec, grid: np.ndarray, content_hw: tuple[int, int],
                    valid: np.ndarray) -> list[Atom]:
    if grid.ndim != 3 or valid.shape != grid.shape[:2]:
        raise ValueError("patch grid/validity mismatch")
    gh, gw, _ = grid.shape
    patch_size = 14  # locked DINOv2 ViT-B/14 backbone
    content_h, content_w = content_hw
    if min(gh, gw, content_h, content_w) <= 0:
        raise ValueError("empty feature grid")
    x0, y0, x1, y1 = view.source_rect
    if x1 <= x0 or y1 <= y0:
        raise ValueError("invalid view source rectangle")
    atoms = []
    for row in range(gh):
        for col in range(gw):
            if not valid[row, col]:
                continue
            px0, py0 = col*patch_size, row*patch_size
            px1, py1 = min((col+1)*patch_size, content_w), min((row+1)*patch_size, content_h)
            if px1 <= px0 or py1 <= py0:
                continue
            rect = (x0+(x1-x0)*px0/content_w, y0+(y1-y0)*py0/content_h,
                    x0+(x1-x0)*px1/content_w, y0+(y1-y0)*py1/content_h)
            center = ((rect[0]+rect[2])/2, (rect[1]+rect[3])/2)
            if view.core_index is not None:
                cx0, cy0, cx1, cy1 = core_rect(view.source_hw, view.core_index)
                if not (cx0 <= center[0] < cx1 or (center[0] == cx1 and cx1 == view.source_hw[1])) or \
                        not (cy0 <= center[1] < cy1 or (center[1] == cy1 and cy1 == view.source_hw[0])):
                    continue
                # Context pixels inform the feature, but only this disjoint core
                # owns evidence mass in original-image coordinates.
                rect = (max(rect[0],cx0),max(rect[1],cy0),
                        min(rect[2],cx1),min(rect[3],cy1))
                if rect[2] <= rect[0] or rect[3] <= rect[1]:
                    continue
                center = ((rect[0]+rect[2])/2,(rect[1]+rect[3])/2)
            feature = np.asarray(grid[row, col], dtype=np.float64)
            if not np.isfinite(feature).all() or np.linalg.norm(feature) == 0:
                raise ValueError("invalid DINO patch feature")
            feature = feature/np.linalg.norm(feature)
            atoms.append(Atom(f"{view.view_id}:{row}:{col}", rect, center,
                              view.view_id, tuple(float(v) for v in feature)))
    return atoms


def candidate_atoms(atoms: list[Atom], box: tuple[float, float, float, float],
                    min_intersection_fraction: float = 0.0) -> list[Atom]:
    """All proposals query this one atom table, so shared pixels retain shared IDs."""
    if not (0 <= min_intersection_fraction <= 1):
        raise ValueError("intersection fraction outside [0,1]")
    chosen = []
    for atom in atoms:
        a = atom.footprint_xyxy
        area = (a[2]-a[0])*(a[3]-a[1])
        overlap = max(0, min(a[2], box[2])-max(a[0], box[0])) * \
                  max(0, min(a[3], box[3])-max(a[1], box[1]))
        if overlap > 0 and overlap/area >= min_intersection_fraction:
            chosen.append(atom)
    return chosen


def load_local_dinov2_registers(repo: str | Path, checkpoint: str | Path,
                                expected_sha256: str, device: str = "cpu"):
    """Use the official local hub entrypoint; no automatic download or model switch."""
    repo, checkpoint = Path(repo), Path(checkpoint)
    if not (repo / "hubconf.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("local DINOv2 official repo/checkpoint missing")
    if file_sha256(checkpoint) != expected_sha256:
        raise ValueError("DINOv2 checkpoint hash changed")
    model = torch.hub.load(str(repo), "dinov2_vitb14_reg", source="local", pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def patch_grid(model, pixels: torch.Tensor, patch_size: int = 14) -> np.ndarray:
    """Official DINOv2 forward_features should expose normalized patch tokens."""
    if pixels.ndim != 3 or pixels.shape[1] % patch_size or pixels.shape[2] % patch_size:
        raise ValueError("image tensor is not patch aligned")
    output = model.forward_features(pixels.unsqueeze(0).to(next(model.parameters()).device))
    if not isinstance(output, dict) or "x_norm_patchtokens" not in output:
        raise RuntimeError("DINOv2 registers API lacks x_norm_patchtokens")
    patches = output["x_norm_patchtokens"]
    gh, gw = pixels.shape[1]//patch_size, pixels.shape[2]//patch_size
    if patches.shape[:2] != (1, gh*gw):
        raise RuntimeError("CLS/register token removal or patch grid shape mismatch")
    return patches[0].detach().float().cpu().reshape(gh, gw, -1).numpy()


def extract_image_atoms(image_path: str | Path, model, feature_config: dict) -> list[Atom]:
    """Extract one shared full grid or four disjoint core grids for a source image."""
    if feature_config.get("patch_size") != 14 or feature_config.get("backbone") != "dinov2_vitb14_reg":
        raise ValueError("feature config differs from the locked backbone")
    long_edge = int(feature_config["long_edge_pixels"])
    policy = feature_config["view_policy"]
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        source_hw = (image.height,image.width)
        if policy == "full":
            views = [full_view(source_hw)]
        elif policy == "four_core_tiles":
            views = four_core_views(source_hw,float(feature_config["tile_context_fraction"]))
        else:
            raise ValueError("unknown feature view policy")
        atoms = []
        for view in views:
            x0,y0,x1,y1 = view.source_rect
            rect = (math.floor(x0),math.floor(y0),math.ceil(x1),math.ceil(y1))
            exact_view = ViewSpec(view.view_id,rect,source_hw,view.core_index)
            pixels,content_hw,valid = prepare_letterbox(image.crop(rect),long_edge,14)
            atoms.extend(atoms_from_grid(exact_view,patch_grid(model,pixels),content_hw,valid))
    if not atoms or len({a.atom_id for a in atoms}) != len(atoms):
        raise ValueError("shared atom extraction produced empty/duplicate grid")
    return atoms


def extract_reference_atoms(image_path: str | Path, weak_roi_xyxy: tuple[float,float,float,float],
                            model, feature_config: dict) -> list[Atom]:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        x0,y0,x1,y1 = weak_roi_xyxy
        rect = (max(0,math.floor(x0)),max(0,math.floor(y0)),
                min(image.width,math.ceil(x1)),min(image.height,math.ceil(y1)))
        if rect[2] <= rect[0] or rect[3] <= rect[1]:
            raise ValueError("weak reference ROI has no area")
        view = ViewSpec("weak_roi",rect,(image.height,image.width))
        pixels,content_hw,valid = prepare_letterbox(image.crop(rect),
                                                     int(feature_config["long_edge_pixels"]),14)
        atoms = atoms_from_grid(view,patch_grid(model,pixels),content_hw,valid)
    if not atoms:
        raise ValueError("weak reference ROI has no valid patches")
    return atoms


def save_atom_cache(path: str | Path, atoms: list[Atom], image_sha256: str,
                    feature_config: dict) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not atoms:
        raise ValueError("cannot cache empty atom grid")
    data = np.asarray([a.feature for a in atoms],dtype=np.float32)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("invalid atom features")
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream,features=data,
                                ids=np.asarray([a.atom_id for a in atoms]),
                                rects=np.asarray([a.footprint_xyxy for a in atoms],dtype=np.float64),
                                centers=np.asarray([a.center_xy for a in atoms],dtype=np.float64),
                                views=np.asarray([a.view_id for a in atoms]),
                                image_sha256=np.asarray(image_sha256),
                                feature_config_sha256=np.asarray(stable_hash(feature_config)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return file_sha256(path)


def load_atom_cache(path: str | Path, image_sha256: str, feature_config: dict) -> list[Atom]:
    with np.load(path,allow_pickle=False) as data:
        if str(data["image_sha256"]) != image_sha256 or \
                str(data["feature_config_sha256"]) != stable_hash(feature_config):
            raise ValueError("atom cache image/config hash mismatch")
        features,ids,rects,centers,views = [data[k].copy() for k in
                                            ("features","ids","rects","centers","views")]
    n = len(ids)
    if features.ndim != 2 or features.shape[0] != n or features.shape[1] == 0 or \
            rects.shape != (n,4) or centers.shape != (n,2) or views.shape != (n,) or \
            not np.isfinite(features).all() or not np.isfinite(rects).all() or \
            not np.isfinite(centers).all():
        raise ValueError("corrupt atom cache array shapes/values")
    if len({str(value) for value in ids}) != len(ids):
        raise ValueError("duplicate cached atom ID")
    return [Atom(str(atom_id),tuple(float(x) for x in rect),
                 tuple(float(x) for x in center),str(view),tuple(float(x) for x in feature))
            for atom_id,rect,center,view,feature in zip(ids,rects,centers,views,features)]
