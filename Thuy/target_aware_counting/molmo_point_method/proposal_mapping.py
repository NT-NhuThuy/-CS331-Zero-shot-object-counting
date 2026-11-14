"""Map GT-free CountGD proposal centers to native patch-token addresses."""
from __future__ import annotations

import numpy as np


def proposal_patch_indices(candidates: list[dict], metadata: dict, *,
                           max_candidates: int = 256) -> dict:
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if len(metadata["subpatch_mapping"]) != 1 or len(metadata["image_sizes"]) != 1:
        raise ValueError("single-image mapping required")
    mapping = np.asarray(metadata["subpatch_mapping"][0])
    pooling = np.asarray(metadata["token_pooling"])
    width, height = metadata["image_sizes"][0]
    if mapping.ndim != 2 or pooling.ndim != 2 or width <= 0 or height <= 0:
        raise ValueError("invalid native mapping geometry")
    selected = candidates[:max_candidates]
    addresses = []
    unmapped = []
    for candidate in selected:
        x0, y0, x1, y1 = candidate["box_xyxy_original"]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            unmapped.append(candidate["candidate_id"])
            continue
        x, y = (x0+x1)/2, (y0+y1)/2
        ix = min(int(x/width*mapping.shape[1]), mapping.shape[1]-1)
        iy = min(int(y/height*mapping.shape[0]), mapping.shape[0]-1)
        vit_id = int(mapping[iy, ix])
        patch_ids = np.flatnonzero(np.any(pooling == vit_id, axis=1)).tolist()
        if not patch_ids:
            unmapped.append(candidate["candidate_id"])
        for patch_id in patch_ids:
            addresses.append({"candidate_id": candidate["candidate_id"],
                              "patch_index": int(patch_id), "vit_patch_id": vit_id})
    ordered = list(dict.fromkeys(r["patch_index"] for r in addresses))
    return {"addresses": addresses,
            "patch_indices": sorted(ordered),
            "ordered_patch_indices": ordered,
            "candidate_total": len(candidates), "candidate_considered": len(selected),
            "candidate_unmapped_ids": unmapped,
            "candidate_capped": len(candidates)-len(selected)}
