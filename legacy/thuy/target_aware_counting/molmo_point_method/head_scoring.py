"""Exact factorization of a MolmoPoint spatial step before wrapper argmax.

At a patch decision, the LM chooses PATCH_TOKEN against ordinary vocabulary.
The spatial head then chooses one patch conditional on that decision. The HF
generation wrapper exposes only the argmax patch; this module restores the
conditional distribution without adding CountGD scores to model probability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Option:
    token_id: int
    log_probability: float
    kind: str


def _log_softmax_valid(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    if not torch.isfinite(values).any():
        raise ValueError("native grammar left no valid continuation")
    return torch.log_softmax(values, dim=-1)


def score_step(output, prefix: torch.Tensor, processor, bounds, *,
               max_language: int = 8, max_spatial: int = 16,
               forced_patch_indices: tuple[int, ...] = (),
               eos_token_ids: tuple[int, ...] = ()) -> list[Option]:
    """Return scored choices at one native grammar step for a batch of one.

    `output` is the official forward output; `processor` is the *same* native
    `MolmoPointLogitProcessor` used by generation. Scores are log probabilities
    after its masks. A no-more-points patch is retained even if outside top-k.
    """
    if prefix.ndim != 2 or prefix.shape[0] != 1 or output.logits.shape[0] != 1:
        raise ValueError("reference scorer requires batch size one")
    if max_language < 1 or max_spatial < 1:
        raise ValueError("invalid option limits")
    raw = output.logits[0, -1].detach().float()
    if raw.numel() != bounds.location_end:
        raise ValueError("expanded vocabulary does not match native bounds")
    last = int(prefix[0, -1])
    patch_stage = (last < bounds.patch_start or last >= bounds.subpatch_end or
                   last == getattr(bounds, "no_more_points_token_id", -1))
    if patch_stage and output.patch_logits is None:
        masked = processor(prefix, raw.clone()[None, :])[0].float()
        logp = _log_softmax_valid(masked)
        ids = torch.topk(logp, min(max_language, int(torch.isfinite(masked).sum()))).indices.tolist()
        ids.extend(i for i in eos_token_ids if 0 <= i < len(logp) and
                   math.isfinite(float(logp[i])) and i not in ids)
        return [Option(i, float(logp[i]), "language") for i in ids]
    if patch_stage and output.patch_logits is not None:
        patch = output.patch_logits[0, -1].detach().float()
        if patch.numel() != bounds.patch_end - bounds.patch_start:
            raise ValueError("patch head/bounds mismatch")
        native_patch = raw[bounds.patch_start:bounds.patch_end]
        viable_native = native_patch > -10000
        if not viable_native.any():
            raise ValueError("wrapper has no patch-token logit")
        patch_token_logit = native_patch.max()
        reconstructed = raw.clone()
        reconstructed[bounds.patch_start:bounds.patch_end] = patch_token_logit
        masked = processor(prefix, reconstructed[None, :])[0].float()
        allowed_patch = torch.isfinite(masked[bounds.patch_start:bounds.patch_end])
        valid_patch = allowed_patch & (patch > -10000)
        # The original wrapper uses -100000 as a sentinel. It must never be
        # treated as a tiny but valid spatial probability.
        base = masked[:bounds.patch_start]
        if valid_patch.any():
            base = torch.cat((base, patch_token_logit.view(1)))
        base_logp = _log_softmax_valid(base)
        choices = []
        language = torch.topk(base_logp[:bounds.patch_start],
                              min(max_language, bounds.patch_start))
        for prob, idx in zip(language.values.tolist(), language.indices.tolist()):
            if math.isfinite(prob):
                choices.append(Option(idx, prob, "language"))
        for idx in eos_token_ids:
            if 0 <= idx < bounds.patch_start and math.isfinite(float(base_logp[idx])) and \
                    not any(option.token_id == idx for option in choices):
                choices.append(Option(idx, float(base_logp[idx]), "language"))
        if valid_patch.any():
            spatial_logits = torch.where(valid_patch, patch, -float("inf"))
            spatial_logp = _log_softmax_valid(spatial_logits)
            forced = [i for i in dict.fromkeys(forced_patch_indices)
                      if 0 <= i < patch.numel() and bool(valid_patch[i])][:max_spatial]
            ids = forced[:]
            for idx in torch.topk(spatial_logp,
                                  min(int(valid_patch.sum()), max_spatial+len(forced))).indices.tolist():
                if len(ids) >= max_spatial:
                    break
                if idx not in ids:
                    ids.append(idx)
            stop_id = getattr(bounds, "no_more_points_token_id", -1)
            stop_index = stop_id - bounds.patch_start
            if 0 <= stop_index < patch.numel() and bool(valid_patch[stop_index]) and stop_index not in ids:
                ids.append(stop_index)
            for idx in ids:
                kind = "no_more_points" if idx == stop_index else "patch"
                choices.append(Option(bounds.patch_start+idx,
                                      float(base_logp[-1]+spatial_logp[idx]), kind))
        return sorted(choices, key=lambda x: x.log_probability, reverse=True)

    masked = processor(prefix, raw.clone()[None, :])[0].float()
    if bounds.patch_start <= last < bounds.patch_end:
        start, end, kind = bounds.subpatch_start, bounds.subpatch_end, "subpatch"
        if output.subpatch_logits is None:
            raise ValueError("subpatch head missing at subpatch step")
    elif bounds.subpatch_start <= last < bounds.subpatch_end:
        start, end, kind = bounds.location_start, bounds.location_end, "location"
        if output.location_logits is None:
            raise ValueError("location head missing at location step")
    else:
        raise ValueError("unexpected native grammar state")
    active = masked[start:end]
    logp = _log_softmax_valid(active)
    ids = torch.topk(logp, min(max_spatial, int(torch.isfinite(active).sum()))).indices.tolist()
    return [Option(start+i, float(logp[i]), kind) for i in ids]


def native_greedy_next(output, prefix: torch.Tensor, processor) -> int:
    """The official width-one choice from its already collapsed logits."""
    masked = processor(prefix, output.logits[:, -1].detach().float().clone())
    return int(masked[0].argmax())
