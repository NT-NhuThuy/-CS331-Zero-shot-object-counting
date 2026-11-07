from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{lineno}: invalid JSON") from e


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm_query(value):
    value = " ".join(str(value).strip().lower().split())
    if not value or any(c in value for c in ".?!\n\r"):
        raise ValueError(f"Use a nonempty object phrase without sentence separators: {value!r}")
    return value


def load_config(path):
    path = Path(path).expanduser().resolve()
    cfg = read_json(path)
    for key in ("repo", "data_root", "images_dir", "checkpoint", "bert_dir"):
        if not cfg.get(key):
            raise ValueError(f"Missing config key: {key}")
        p = Path(cfg[key]).expanduser()
        cfg[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    for key in ("class_aliases",):
        if cfg.get(key):
            p = Path(cfg[key]).expanduser()
            cfg[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    cfg.setdefault("device", "cuda")
    cfg.setdefault("seed", 42)
    cfg.setdefault("resize_short", 800)
    cfg.setdefault("resize_max", 1333)
    cfg.setdefault("grid_size", 7)
    cfg.setdefault("context_scale", 1.5)
    cfg.setdefault("baseline_threshold", 0.23)
    cfg.setdefault("text_threshold", 0.0)
    cfg.setdefault("pool_threshold", 0.23)
    cfg.setdefault("grounding_threshold", 0.5)
    cfg.setdefault("point_radius_relative", 0.01)
    cfg.setdefault("roi_chunk", 64)
    cfg.setdefault("model", {"dim": 256, "heads": 8, "layers": 2, "dropout": 0.1})
    for key in ("baseline_threshold", "pool_threshold", "grounding_threshold", "text_threshold"):
        if not 0 <= cfg[key] <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    if cfg["resize_short"] <= 0 or cfg["resize_max"] < cfg["resize_short"]:
        raise ValueError("Invalid resize sizes")
    if cfg["grid_size"] < 1 or cfg["context_scale"] < 1 or cfg["roi_chunk"] < 1:
        raise ValueError("Invalid ROI configuration")
    if cfg["point_radius_relative"] <= 0:
        raise ValueError("Matching radius must be positive")
    return cfg


def atomic_torch_save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_tensor_file(path):
    import torch
    # Our own files contain tensors and primitive types only.
    return torch.load(path, map_location="cpu", weights_only=True)
