from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sp1_alignment.capture import TextCrossAttentionCapture, summarize_text_attention  # noqa: E402


class ToyDecoderLayer(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.ca_text = nn.MultiheadAttention(dim, num_heads=2, batch_first=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, tgt: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        out, _ = self.ca_text(tgt, text, text, need_weights=False)
        return self.proj(out)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyDecoderLayer(), ToyDecoderLayer()])
        self.class_head = nn.Linear(8, 5)

    def forward(self, tgt: torch.Tensor, text: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        hs = []
        x = tgt
        for layer in self.layers:
            x = layer(x, text)
            hs.append(x)
        logits = self.class_head(x.transpose(0, 1))
        return {"pred_logits": logits, "hs": hs}


def main() -> None:
    torch.manual_seed(19)
    baseline = ToyModel()
    instrumented = copy.deepcopy(baseline)
    tgt = torch.randn(4, 1, 8)
    text = torch.randn(8, 1, 8)

    with torch.no_grad():
        expected = baseline(tgt, text)

    capture = TextCrossAttentionCapture(instrumented)
    capture.start()
    with torch.no_grad():
        actual = instrumented(tgt, text)

    exact_logits = torch.equal(expected["pred_logits"], actual["pred_logits"])
    exact_hs = all(torch.equal(a, b) for a, b in zip(expected["hs"], actual["hs"]))
    layers = capture.get_last()

    print(f"candidate output exact match: {exact_logits}")
    print(f"hidden-state output exact match: {exact_hs}")
    print(f"captured layers: {len(layers)}")
    print(f"attention shapes: {[tuple(x.shape) for x in layers]}")

    ids = torch.tensor([[101, 11, 12, 1008, 1008, 1008, 1012, 102]])
    summary = summarize_text_attention(layers, ids)
    total = (
        summary["semantic_mass"]
        + summary["exemplar_mass"]
        + summary["special_mass"]
        + summary["other_mass"]
    )
    normalized = torch.allclose(total, torch.ones_like(total), atol=1e-6)
    print(f"attention mass sums to 1: {normalized}")

    if not (exact_logits and exact_hs and len(layers) == 2 and normalized):
        raise SystemExit(1)

    capture.detach()
    print("SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
