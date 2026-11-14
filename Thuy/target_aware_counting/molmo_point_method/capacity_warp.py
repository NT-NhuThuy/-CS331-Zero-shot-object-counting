"""Invertible separable image warp driven by native decoder address collisions.

All coordinates are pixel-edge coordinates.  A point at an image boundary stays
at that boundary.  The demand is only a proposal proxy, never an object count.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Warp:
    x_edges: np.ndarray
    y_edges: np.ndarray
    width: int
    height: int

    def forward(self, xy):
        p = np.asarray(xy, dtype=float).reshape(-1, 2)
        return np.stack((np.interp(p[..., 0], self.x_edges, np.linspace(0, self.width, len(self.x_edges))),
                         np.interp(p[..., 1], self.y_edges, np.linspace(0, self.height, len(self.y_edges)))), axis=-1)

    def inverse(self, xy):
        p = np.asarray(xy, dtype=float).reshape(-1, 2)
        return np.stack((np.interp(p[..., 0], np.linspace(0, self.width, len(self.x_edges)), self.x_edges),
                         np.interp(p[..., 1], np.linspace(0, self.height, len(self.y_edges)), self.y_edges)), axis=-1)

    def render(self, image: Image.Image) -> Image.Image:
        if image.size != (self.width, self.height):
            raise ValueError("warp and image sizes differ")
        if np.allclose(self.x_edges, np.linspace(0, self.width, len(self.x_edges))) and \
                np.allclose(self.y_edges, np.linspace(0, self.height, len(self.y_edges))):
            return image.copy()
        # PIL mesh uses source quads; each output cell maps to a source rectangle.
        # Piecewise linear separable maps are exact at grid edges.
        nx, ny = len(self.x_edges)-1, len(self.y_edges)-1
        xout = np.linspace(0, self.width, nx+1)
        yout = np.linspace(0, self.height, ny+1)
        mesh = []
        for j in range(ny):
            for i in range(nx):
                box = tuple(round(v) for v in (xout[i], yout[j], xout[i+1], yout[j+1]))
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                quad = (self.x_edges[i], self.y_edges[j], self.x_edges[i], self.y_edges[j+1],
                        self.x_edges[i+1], self.y_edges[j+1], self.x_edges[i+1], self.y_edges[j])
                mesh.append((box, quad))
        return image.transform(image.size, Image.Transform.MESH, mesh,
                               resample=Image.Resampling.BICUBIC)


def native_addresses(xy, metadata: dict):
    """Return native (patch token, subpatch) addresses per point.

    A point can be represented by several crops; all are retained.  The
    native repeat mask forbids reusing a patch/subpatch address regardless
    of the finer 3x3 location token.
    """
    mapping = np.asarray(metadata["subpatch_mapping"][0])
    pooling = np.asarray(metadata["token_pooling"])
    width, height = map(float, metadata["image_sizes"][0])
    if mapping.ndim != 2 or pooling.ndim != 2 or width <= 0 or height <= 0:
        raise ValueError("invalid native processor mapping")
    out = []
    for x, y in xy:
        if not (0 <= x <= width and 0 <= y <= height):
            out.append(None)
            continue
        ix = min(int(x / width * mapping.shape[1]), mapping.shape[1]-1)
        iy = min(int(y / height * mapping.shape[0]), mapping.shape[0]-1)
        vit = int(mapping[iy, ix])
        if vit < 0:
            out.append(None)
            continue
        addresses = np.argwhere(pooling == vit)
        out.append(tuple((int(patch), int(subpatch))
                         for patch, subpatch in addresses) or None)
    return out


def collision_groups(xy, metadata: dict):
    addresses = native_addresses(xy, metadata)
    # Shared address is a demand proxy, not an unavoidable collision: another
    # crop may still give one of the points a distinct address.
    # Return each unordered pair once.  The earlier star-shaped groups counted
    # the same pair repeatedly when three or more proposals shared an address.
    return [[i, j] for i, address in enumerate(addresses) if address is not None
            for j in range(i+1, len(addresses))
            if addresses[j] is not None and set(address) & set(addresses[j])]


def assignment_capacity(xy, metadata: dict) -> dict:
    """Maximum one-to-one proposal/address matching across all processor crops.

    Alternating reachability from unmatched proposals identifies a deficient
    proposal subset for demand. This remains a GT-free proxy: duplicate or part
    proposals can make a deficient subset without extra physical instances.
    """
    choices = [tuple(dict.fromkeys(x or ())) for x in native_addresses(xy, metadata)]
    owner = {}

    def augment(i, seen):
        for address in choices[i]:
            if address in seen:
                continue
            seen.add(address)
            previous = owner.get(address)
            if previous is None or augment(previous, seen):
                owner[address] = i
                return True
        return False

    for i in range(len(choices)):
        augment(i, set())
    matched = set(owner.values())
    unmatched = sorted(set(range(len(choices))) - matched)
    deficient = set(unmatched)
    queue = list(unmatched)
    for i in queue:
        for address in choices[i]:
            j = owner.get(address)
            if j is not None and j not in deficient:
                deficient.add(j)
                queue.append(j)
    return {"proposal_count": len(choices), "matched_count": len(matched),
            "unmatched_count": len(unmatched), "unmatched_indices": unmatched,
            "deficient_indices": sorted(deficient),
            "unmapped_indices": [i for i, x in enumerate(choices) if not x]}


def _axis_edges(length: int, demand: np.ndarray, floor: float, min_slope: float,
                max_slope: float) -> np.ndarray:
    if not (0 < floor <= 1 and 0 < min_slope <= 1 <= max_slope):
        raise ValueError("invalid warp bounds")
    if not np.any(demand) or np.ptp(demand) < 1e-12:
        return np.linspace(0, length, len(demand)+1)
    density = floor + (1-floor) * demand / max(float(demand.mean()), 1e-12)
    # Bounded water filling preserves integral and derivative constraints.
    for _ in range(100):
        density = np.clip(density, min_slope, max_slope)
        delta = len(density) - density.sum()
        free = (density < max_slope-1e-10) if delta > 0 else (density > min_slope+1e-10)
        if abs(delta) < 1e-9 or not free.any():
            break
        density[free] += delta / free.sum()
    if abs(density.sum() - len(density)) > 1e-7:
        raise ValueError("could not satisfy warp slope bounds")
    # ``density`` is the desired forward derivative on uniform source bins.
    # Warp stores source locations corresponding to uniform output edges, so
    # invert the integrated forward map.  Accumulating density directly as
    # source edges would compress high-demand regions instead of expanding them.
    source = np.linspace(0, length, len(density)+1)
    warped = np.r_[0., np.cumsum(density)] * length / len(density)
    return np.interp(np.linspace(0, length, len(density)+1), warped, source)


def build_warp(width: int, height: int, points_xy, metadata: dict, *,
               bins: int = 64, floor: float = 0.5, min_slope: float = 0.5,
               max_slope: float = 2.0) -> tuple[Warp, dict]:
    if min(width, height, bins) < 2 or bins > min(width, height):
        raise ValueError("bins must be between 2 and the shortest image side")
    points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise ValueError("nonfinite proposal center")
    capacity = assignment_capacity(points, metadata)
    demand = np.zeros((bins, bins), dtype=float)
    involved = capacity["deficient_indices"]
    for i in involved:
        x, y = points[i]
        if 0 <= x <= width and 0 <= y <= height:
            demand[min(int(y / height * bins), bins-1),
                   min(int(x / width * bins), bins-1)] += 1
    x_edges = _axis_edges(width, demand.sum(axis=0), floor, min_slope, max_slope)
    y_edges = _axis_edges(height, demand.sum(axis=1), floor, min_slope, max_slope)
    warp = Warp(x_edges, y_edges, width, height)
    x_forward_slopes = (width / bins) / np.diff(x_edges)
    y_forward_slopes = (height / bins) / np.diff(y_edges)
    return warp, {"proposal_count": len(points), "assignment_capacity": capacity,
                  "demand_proposals": len(involved),
                  "y_slope_minmax": [float(v) for v in (y_forward_slopes.min(), y_forward_slopes.max())],
                  "x_slope_minmax": [float(v) for v in (x_forward_slopes.min(), x_forward_slopes.max())]}
