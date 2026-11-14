from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Iterator, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class TokenRoleMasks:
    """Token-role masks for the CountGD++ inference prompt."""

    semantic: torch.Tensor
    exemplar: torch.Tensor
    special: torch.Tensor
    padding_or_other: torch.Tensor


class TextCrossAttentionCapture:
    """Capture CountGD++ decoder text cross-attention without changing model math.

    CountGD++ uses nn.MultiheadAttention modules named ``ca_text`` inside
    the decoder layers. The wrapper temporarily replaces each module's
    ``forward`` method so the original attention output is returned exactly
    as before while its attention weights are copied to CPU for inspection.

    This avoids modifying the upstream transformer implementation.
    """

    def __init__(self, model: nn.Module, *, module_suffix: str = ".ca_text") -> None:
        self.model = model
        self.module_suffix = module_suffix
        self._original_forwards: dict[str, object] = {}
        self._captures: list[tuple[str, torch.Tensor]] = []
        self._attached = False
        self._recording = False

    @property
    def module_names(self) -> list[str]:
        return list(self._original_forwards.keys())

    def attach(self) -> None:
        if self._attached:
            return

        found = 0
        for name, module in self.model.named_modules():
            if not name.endswith(self.module_suffix):
                continue
            if not isinstance(module, nn.MultiheadAttention):
                continue

            original_forward = module.forward
            self._original_forwards[name] = original_forward

            def wrapped_forward(
                self_module: nn.MultiheadAttention,
                *args,
                _original_forward=original_forward,
                _capture_name=name,
                **kwargs,
            ):
                # IMPORTANT: the model's actual output must follow exactly the
                # original call path. Forcing need_weights=True can make PyTorch
                # choose a different MultiheadAttention execution path, which is
                # enough to change floating-point outputs.
                result = _original_forward(*args, **kwargs)

                if self._recording:
                    capture_kwargs = dict(kwargs)
                    capture_kwargs["need_weights"] = True
                    capture_kwargs["average_attn_weights"] = False
                    capture_result = _original_forward(*args, **capture_kwargs)
                    if (
                        isinstance(capture_result, tuple)
                        and len(capture_result) == 2
                        and torch.is_tensor(capture_result[1])
                    ):
                        self._captures.append(
                            (_capture_name, capture_result[1].detach().float().cpu())
                        )

                return result

            module._sp1_capture_owner = self
            module.forward = MethodType(wrapped_forward, module)
            found += 1

        if found == 0:
            raise RuntimeError(
                "No CountGD++ text-cross-attention modules named '*.ca_text' were found."
            )
        self._attached = True

    def detach(self) -> None:
        if not self._attached:
            return
        module_map = dict(self.model.named_modules())
        for name, original_forward in self._original_forwards.items():
            module = module_map.get(name)
            if module is None:
                continue
            module.forward = original_forward  # type: ignore[method-assign]
            if hasattr(module, "_sp1_capture_owner"):
                delattr(module, "_sp1_capture_owner")
        self._original_forwards.clear()
        self._captures.clear()
        self._recording = False
        self._attached = False

    def start(self) -> None:
        if not self._attached:
            self.attach()
        self._captures.clear()
        self._recording = True

    def stop(self) -> None:
        self._recording = False

    def get_last(self, *, strict: bool = True) -> list[torch.Tensor]:
        """Return the attention tensors in decoder-layer execution order.

        Each tensor has shape ``[batch, heads, num_queries, text_tokens]``.
        """
        if not self._captures:
            raise RuntimeError(
                "No text attention was captured. Call start() immediately before model(...)."
            )

        ordered = sorted(self._captures, key=lambda item: self.module_names.index(item[0]))
        tensors = [tensor for _, tensor in ordered]

        if strict and len(tensors) != len(self.module_names):
            raise RuntimeError(
                f"Expected {len(self.module_names)} attention tensors, got {len(tensors)}."
            )
        return tensors

    @contextmanager
    def capture(self) -> Iterator["TextCrossAttentionCapture"]:
        self.start()
        try:
            yield self
        finally:
            # Keep the captured tensors available after the context exits.
            self.stop()


def build_token_role_masks(
    input_ids: torch.Tensor,
    *,
    cls_id: int = 101,
    sep_id: int = 102,
    star_id: int = 1008,
    dot_id: int = 1012,
) -> TokenRoleMasks:
    """Build semantic/exemplar/special masks from CountGD++ token IDs.

    The validated CountGD++ prompt pattern is typically::

        [CLS] semantic tokens * * * . [SEP]

    One known validation sample has no ``*`` tokens; in that case all tokens
    between [CLS] and the first ``.`` are treated as semantic tokens.
    """
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [B,T], got {tuple(input_ids.shape)}")

    ids = input_ids.to(dtype=torch.long)
    batch, tokens = ids.shape
    semantic = torch.zeros((batch, tokens), dtype=torch.bool, device=ids.device)
    exemplar = ids.eq(star_id)
    special = ids.eq(cls_id) | ids.eq(sep_id) | ids.eq(dot_id)

    for b in range(batch):
        row = ids[b]
        dot_positions = torch.nonzero(row.eq(dot_id), as_tuple=False).flatten()
        first_dot = int(dot_positions[0]) if dot_positions.numel() else tokens
        cls_positions = torch.nonzero(row.eq(cls_id), as_tuple=False).flatten()
        start = int(cls_positions[0]) + 1 if cls_positions.numel() else 0

        star_positions = torch.nonzero(row.eq(star_id), as_tuple=False).flatten()
        end = int(star_positions[0]) if star_positions.numel() else first_dot
        if end > start:
            semantic[b, start:end] = True

    padding_or_other = ~(semantic | exemplar | special)
    return TokenRoleMasks(
        semantic=semantic,
        exemplar=exemplar,
        special=special,
        padding_or_other=padding_or_other,
    )


def _masked_sum(attention: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # attention: [B,H,N,T]
    # mask:      [B,T]
    return (attention * mask[:, None, None, :].to(attention.dtype)).sum(dim=-1).mean(dim=1)


def _masked_max(attention: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Return max over head and token positions selected by mask.
    expanded = mask[:, None, None, :].expand_as(attention)
    values = attention.masked_fill(~expanded, float("-inf"))
    return values.amax(dim=-1).amax(dim=1)


def summarize_text_attention(
    attention_layers: Sequence[torch.Tensor],
    input_ids: torch.Tensor,
) -> dict[str, torch.Tensor | list[int]]:
    """Summarize per-layer text cross-attention without storing full maps.

    Returns per-candidate tensors with shape ``[L,B,N]`` plus global token-role
    statistics. This is intentionally compact enough for a 1286-image run.
    """
    if not attention_layers:
        raise ValueError("attention_layers is empty")

    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    masks = build_token_role_masks(input_ids)
    batch, token_count = input_ids.shape[-2], input_ids.shape[-1]
    semantic_mass = []
    exemplar_mass = []
    special_mass = []
    other_mass = []
    semantic_peak = []
    exemplar_peak = []
    argmax_role = []

    for layer_index, attention in enumerate(attention_layers):
        if attention.ndim != 4:
            raise ValueError(
                f"layer {layer_index}: attention must be [B,H,N,T], got {tuple(attention.shape)}"
            )
        if attention.shape[0] != batch or attention.shape[-1] != token_count:
            raise ValueError(
                f"layer {layer_index}: attention/input mismatch: "
                f"{tuple(attention.shape)} vs input_ids={tuple(input_ids.shape)}"
            )

        semantic_mass.append(_masked_sum(attention, masks.semantic))
        exemplar_mass.append(_masked_sum(attention, masks.exemplar))
        special_mass.append(_masked_sum(attention, masks.special))
        other_mass.append(_masked_sum(attention, masks.padding_or_other))
        semantic_peak.append(_masked_max(attention, masks.semantic))
        exemplar_peak.append(_masked_max(attention, masks.exemplar))

        mean_heads = attention.mean(dim=1)
        argmax_token = mean_heads.argmax(dim=-1)
        role = torch.full_like(argmax_token, 3, dtype=torch.long)  # other/padding
        role = torch.where(torch.gather(masks.special, 1, argmax_token), torch.tensor(2, device=role.device), role)
        role = torch.where(torch.gather(masks.exemplar, 1, argmax_token), torch.tensor(1, device=role.device), role)
        role = torch.where(torch.gather(masks.semantic, 1, argmax_token), torch.tensor(0, device=role.device), role)
        argmax_role.append(role)

    return {
        "semantic_mass": torch.stack(semantic_mass, dim=0),
        "exemplar_mass": torch.stack(exemplar_mass, dim=0),
        "special_mass": torch.stack(special_mass, dim=0),
        "other_mass": torch.stack(other_mass, dim=0),
        "semantic_peak": torch.stack(semantic_peak, dim=0),
        "exemplar_peak": torch.stack(exemplar_peak, dim=0),
        "argmax_role": torch.stack(argmax_role, dim=0),
        "num_layers": len(attention_layers),
    }


def flatten_capture_for_save(
    capture: TextCrossAttentionCapture,
    input_ids: torch.Tensor,
) -> dict[str, torch.Tensor | list[int]]:
    """Convenience helper for inference code."""
    return summarize_text_attention(capture.get_last(), input_ids)
