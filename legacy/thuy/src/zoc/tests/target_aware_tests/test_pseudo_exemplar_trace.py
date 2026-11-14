from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

# Import the selector from the existing CountGD++ source. Running this test
# requires the same Python environment as CountGD++ because engine_inference.py
# imports the model's normal dependencies.
ROOT = Path(__file__).resolve().parents[1]
COUNTGD = ROOT / "CountGDPlusPlus"
sys.path.insert(0, str(COUNTGD))

try:
    from engine_inference import (  # noqa: E402
        _pseudo_exemplar_selection_metadata,
        get_pseudo_exemplars,
    )
except ModuleNotFoundError as exc:
    pytest.skip(f"CountGD++ integration source is not included: {exc}", allow_module_level=True)


def test_selection_trace_matches_native_rule() -> None:
    # 6 decoder queries, 3 text-token scores each.
    logits = torch.tensor(
        [[[0.0, 0.0, 1.0], [4.0, -2.0, -3.0], [1.0, 1.5, 0.0],
          [-2.0, -2.0, -2.0], [3.0, 2.0, 1.0], [2.0, 0.0, -1.0]]],
        dtype=torch.float32,
    )
    boxes = torch.tensor(
        [[[0.50, 0.50, 0.10, 0.10], [0.10, 0.10, 0.10, 0.10],
          [0.20, 0.20, 0.10, 0.10], [0.30, 0.30, 0.10, 0.10],
          [0.70, 0.70, 0.10, 0.10], [0.90, 0.90, 0.10, 0.10]]],
        dtype=torch.float32,
    )
    outputs = {"pred_logits": logits, "pred_boxes": boxes}
    image_sizes = [torch.tensor([100, 200])]

    pseudo, meta = _pseudo_exemplar_selection_metadata(
        outputs, image_sizes, box_threshold=0.70, num_exemplars=3
    )

    scores = logits.sigmoid().max(dim=-1).values[0]
    expected = torch.argsort(scores[scores > 0.70], descending=True)
    original_indices = torch.nonzero(scores > 0.70, as_tuple=False).flatten()
    expected = original_indices[expected][:3]

    assert torch.equal(meta[0]["selected_indices"], expected)
    assert torch.equal(meta[0]["selected_indices_pre_batch_truncation"], expected)
    assert pseudo[0].shape == (3, 4)
    assert torch.allclose(
        meta[0]["selected_scores"], scores[expected], atol=1e-6, rtol=1e-6
    )

    # Public backward-compatible helper must return the same boxes.
    legacy = get_pseudo_exemplars(
        outputs,
        image_sizes,
        [torch.tensor([0])],
        [["x"]],
        ["x ."],
        0.70,
        num_exemplars=3,
    )
    assert torch.allclose(legacy[0], pseudo[0])


def test_empty_selection_is_device_safe() -> None:
    outputs = {
        "pred_logits": torch.full((1, 4, 3), -10.0),
        "pred_boxes": torch.rand(1, 4, 4),
    }
    pseudo, meta = _pseudo_exemplar_selection_metadata(
        outputs, [torch.tensor([100, 100])], box_threshold=0.99, num_exemplars=3
    )
    assert pseudo[0].shape == (0, 4)
    assert meta[0]["selected_indices"].numel() == 0
    assert meta[0]["num_exemplars_selected"] == 0
