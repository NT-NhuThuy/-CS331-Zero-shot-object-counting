from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
from .io import read_json, norm_query, digest, file_hash


class FSC147:
    """Official 384/VarV2 coordinates: points=(x,y), exemplar corners=(x,y).

    Annotation H/W are original sizes, NOT coordinate bounds for VarV2 files.
    Neither points nor boxes may be scaled using annotation H/W or ratio_h/w.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        root = Path(cfg["data_root"])
        self.images_dir = Path(cfg["images_dir"])
        paths = [root / "annotation_FSC147_384.json",
                 root / "Train_Test_Val_FSC_147.json", root / "ImageClasses_FSC147.txt"]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        self.annotations = read_json(paths[0])
        all_splits = read_json(paths[1])
        self.splits = {s: all_splits[s] for s in ("train", "val", "test")}
        aliases = read_json(cfg["class_aliases"]) if cfg.get("class_aliases") else {}
        aliases = {norm_query(k): norm_query(v) for k, v in aliases.items()}
        self.classes = {}
        for line in paths[2].read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image, query = line.split(maxsplit=1)
            query = norm_query(query)
            query = aliases.get(query, query)
            if image in self.classes:
                raise ValueError(f"Duplicate class entry: {image}")
            self.classes[image] = query
        self.image_split = {}
        for split, names in self.splits.items():
            for name in names:
                if name in self.image_split:
                    raise ValueError(f"Image repeated/across splits: {name}")
                if Path(name).name != name:
                    raise ValueError(f"Image name must be a basename: {name}")
                if name not in self.classes or name not in self.annotations:
                    raise ValueError(f"Missing annotation/class: {name}")
                self.image_split[name] = split
        self.class_sets = {s: set(self.classes[n] for n in names)
                           for s, names in self.splits.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = self.class_sets[a] & self.class_sets[b]
            if overlap:
                raise ValueError(f"Zero-shot class overlap {a}/{b}: {sorted(overlap)}")
        self.fingerprint = digest({"metadata": [file_hash(p) for p in paths],
                                   "aliases": aliases, "coordinate_schema": "fsc384-xy-box-xy-point"})

    def image_path(self, name):
        if name not in self.image_split:
            raise ValueError(f"Unknown image: {name}")
        return self.images_dir / name

    def size(self, name):
        with Image.open(self.image_path(name)) as img:
            return img.size  # actual width, height

    def points(self, name):
        points = np.asarray(self.annotations[name]["points"], dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(points).all():
            raise ValueError(f"Nonfinite points: {name}")
        return points

    def boxes(self, name):
        corners = np.asarray(self.annotations[name]["box_examples_coordinates"], dtype=np.float64)
        if corners.size == 0:
            return np.empty((0, 4))
        if corners.ndim != 3 or corners.shape[1:] != (4, 2) or not np.isfinite(corners).all():
            raise ValueError(f"Expected 4 (x,y) exemplar corners: {name}")
        xy = corners
        return np.concatenate([xy.min(axis=1), xy.max(axis=1)], axis=1)

    def validate_image(self, name):
        w, h = self.size(name)
        pts, boxes = self.points(name), self.boxes(name)
        if len(pts) == 0:
            raise ValueError(f"FSC positive class has no points: {name}")
        # Coordinates on the right/bottom boundary are permitted; large mismatches fail.
        eps = 1e-6  # e.g. official 1989.jpg has x=408.00000000000006 at width=408
        if ((pts < -eps).any() or (pts[:, 0] > w + eps).any() or (pts[:, 1] > h + eps).any()):
            raise ValueError(f"Points outside actual image {name} ({w}x{h}); check data variant")
        if len(boxes) and ((boxes < -eps).any() or (boxes[:, [0, 2]] > w + eps).any()
                           or (boxes[:, [1, 3]] > h + eps).any()
                           or (boxes[:, 2:] <= boxes[:, :2]).any()):
            raise ValueError(f"Invalid exemplar coordinates: {name}; boxes use (x,y) corners")
        return w, h

    def audit(self, limit=None):
        result = {"fingerprint": self.fingerprint, "coordinate_schema": "boxes xy; points xy; actual image size",
                  "splits": {s: {"images": len(v), "classes": len(self.class_sets[s])}
                             for s, v in self.splits.items()}}
        checked = 0
        for split in ("train", "val", "test"):
            names = self.splits[split][:limit] if limit else self.splits[split]
            for name in names:
                self.validate_image(name)
                checked += 1
        result["images_checked"] = checked
        return result
