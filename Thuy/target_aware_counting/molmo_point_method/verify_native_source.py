"""Check the pinned official MolmoPoint grammar without loading checkpoint weights."""
from __future__ import annotations

import sys
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .head_scoring import score_step


def main() -> None:
    source = Path(__file__).resolve().parent / "vendor_source_188130f"
    model_class = get_class_from_dynamic_module(
        "modeling_molmo_point.MolmoPointForConditionalGeneration",
        str(source), local_files_only=True)
    module = sys.modules[model_class.__module__]
    bounds = module.GeneratedTokenBounds(
        vocab_size=10, n_patches=3, n_subpatches=2, n_locations=9,
        no_more_points_class=True)
    mask = module.MolmoPointLogitProcessor(
        bounds, prevent_repeats=True, force_patch_sorted=True,
        force_subpatch_sorted=True)
    def allowed(prefix):
        scores = mask(torch.tensor([prefix]), torch.zeros(1, bounds.location_end))
        return torch.isfinite(scores[0])
    initial = allowed([1])
    assert initial[bounds.patch_start:bounds.patch_end].all()
    assert not initial[bounds.subpatch_start:bounds.location_end].any()
    after_patch = allowed([1, bounds.patch_start])
    assert after_patch[bounds.subpatch_start:bounds.subpatch_end].all()
    assert not after_patch[:bounds.subpatch_start].any()
    after_stop = allowed([1, bounds.no_more_points_token_id])
    assert not after_stop[bounds.patch_start:bounds.location_end].any()
    assert after_stop[:bounds.patch_start].all()
    points = module.extract_image_points(
        "<POINT_0><POINT_3><POINT_9> 1",
        np.array([[0, 1], [2, 3]]), [np.array([[0, 1], [2, 3]])],
        no_more_points_class=True, location="3x3", image_sizes=[(100, 100)])
    assert len(points) == 1 and points[0][:2] == [1, 0]
    assert abs(points[0][2]-24.75) < 1e-6 and abs(points[0][3]-24.75) < 1e-6
    raw = torch.full((1, 1, bounds.location_end), -100000.0)
    raw[0, 0, :bounds.patch_start] = 0
    raw[0, 0, bounds.patch_start+1] = 2
    output = SimpleNamespace(logits=raw,
                             patch_logits=torch.tensor([[[0.0, 1.0, -1.0, -2.0]]]))
    options = score_step(output, torch.tensor([[1]]), mask, bounds,
                         max_language=10, max_spatial=4, eos_token_ids=(2,))
    assert abs(sum(math.exp(option.log_probability) for option in options)-1) < 1e-6
    assert any(option.token_id == bounds.no_more_points_token_id for option in options)
    print("native_source_mask_contract=pass")


if __name__ == "__main__":
    main()
