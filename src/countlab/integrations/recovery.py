"""GT-blind multi-view source. Outputs *candidates*, never labels or counts.

A native-image grid is a NEW protocol, not Lộc V7.5 or FQIE V14 parity.
Only claim parity after matching exact crop grids, thresholds and NMS from
those versions on raw source predictions.
"""
from __future__ import annotations
from pathlib import Path
import tempfile
from typing import Iterable


def tile_bounds(width: int, height: int, *, grids: tuple[int, ...]=(2,), overlap: float=.20,
                max_tiles: int=64) -> list[tuple[str, tuple[int,int,int,int]]]:
    if width < 2 or height < 2 or not 0 <= overlap < .5:
        raise ValueError("invalid image dimensions or overlap")
    boxes = []
    for grid in grids:
        if not isinstance(grid, int) or not 1 <= grid <= 12:
            raise ValueError("grid must be in [1,12]")
        for row in range(grid):
            for col in range(grid):
                cell_w, cell_h = width/grid, height/grid
                x0 = max(0, round((col - overlap) * cell_w))
                y0 = max(0, round((row - overlap) * cell_h))
                x1 = min(width, round((col + 1 + overlap) * cell_w))
                y1 = min(height, round((row + 1 + overlap) * cell_h))
                if x1 <= x0 or y1 <= y0:
                    raise ValueError("invalid tile region")
                boxes.append((f"g{grid}_r{row}_c{col}", (x0, y0, x1, y1)))
    if len(boxes) > max_tiles:
        raise ValueError("tile budget exceeded; change frozen protocol")
    return boxes


def recover_from_native(image_path: Path, query: str, provider, *, grids: tuple[int,...]=(2,),
                        overlap: float=.2, max_tiles: int=64, image_id: str="") -> list[dict]:
    """Call real CountGD++ backend for each crop, restore ORIGINAL coordinates."""
    from PIL import Image
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        bounds = tile_bounds(*rgb.size, grids=grids, overlap=overlap, max_tiles=max_tiles)
        with tempfile.TemporaryDirectory(prefix="countlab_tiles_") as folder:
            result: list[dict] = []
            for view_id, (x0, y0, x1, y1) in bounds:
                path = Path(folder) / f"{view_id}.png"
                rgb.crop((x0,y0,x1,y1)).save(path)
                result.extend(provider.infer(path, query, role="rescue", view_id=view_id,
                                             offset=(x0,y0), image_id=image_id))
    return result
