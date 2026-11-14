"""Slow, branch-isolated forward replay using the official HF generation hooks.

Each call creates a fresh language KV cache. This is the correctness reference,
not a production beam implementation. Never share `past_key_values` or
`last_predicted_patch_id` between alternative prefixes.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch


@dataclass
class BranchState:
    """Output and private generation state for one complete token prefix."""
    prefix: torch.Tensor
    kwargs: dict
    output: object


def start_branch(model, inputs: dict) -> BranchState:
    """Encode the image/prompt once and retain its official generation cache."""
    prefix = inputs["input_ids"]
    if prefix.ndim != 2 or prefix.shape[0] != 1:
        raise ValueError("cached branch requires one processor input")
    kwargs = {key: value for key, value in inputs.items() if key != "input_ids"}
    kwargs["use_cache"] = True
    kwargs["logits_to_keep"] = 1
    kwargs["cache_position"] = torch.arange(prefix.shape[1], device=prefix.device)
    with torch.inference_mode():
        prepared = model.prepare_inputs_for_generation(prefix, **kwargs)
        output = model(**prepared, return_dict=True)
    if output.past_key_values is None:
        raise ValueError("model did not return a generation cache")
    return BranchState(prefix, kwargs, output)


def advance_branch(model, parent: BranchState, token_id: int) -> BranchState:
    """Fork a prefix without sharing mutable KV state with its siblings."""
    with torch.inference_mode():
        kwargs = model._update_model_kwargs_for_generation(
            parent.output, dict(parent.kwargs), is_encoder_decoder=False)
        if kwargs.get("past_key_values") is None:
            raise ValueError("generation hook dropped the KV cache")
        kwargs["past_key_values"] = copy.deepcopy(kwargs["past_key_values"])
        prefix = torch.cat((parent.prefix,
                            torch.tensor([[token_id]], dtype=parent.prefix.dtype,
                                         device=parent.prefix.device)), dim=1)
        prepared = model.prepare_inputs_for_generation(prefix, **kwargs)
        output = model(**prepared, return_dict=True)
    return BranchState(prefix, kwargs, output)


def replay(model, inputs: dict, continuation: list[int]):
    """Return the official forward output at a complete generated prefix.

    The returned output predicts the *next* token. `inputs` is the processor
    batch after removing pointing metadata. Image data are encoded once per
    replay and then cached by the official `_update_model_kwargs...` hook.
    """
    if "input_ids" not in inputs or inputs["input_ids"].shape[0] != 1:
        raise ValueError("reference replay requires one processor input")
    prefix = inputs["input_ids"]
    if prefix.ndim != 2:
        raise ValueError("input_ids must be [1, sequence]")
    kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}
    kwargs["use_cache"] = True
    # GenerationMixin sets this before its decoding loop.  Calling
    # prepare_inputs_for_generation directly without it leaves an explicit
    # ``None`` that overrides MolmoPoint.forward's default.  Its custom LM head
    # then indexes hidden_states with None and produces four-dimensional
    # logits.  We only ever score the next token, so retain the native value 1.
    kwargs["logits_to_keep"] = 1
    kwargs["cache_position"] = torch.arange(prefix.shape[1], device=prefix.device)
    with torch.inference_mode():
        prepared = model.prepare_inputs_for_generation(prefix, **kwargs)
        output = model(**prepared, return_dict=True)
        for token_id in continuation:
            kwargs = model._update_model_kwargs_for_generation(
                output, kwargs, is_encoder_decoder=False)
            prefix = torch.cat((prefix, torch.tensor([[token_id]], device=prefix.device,
                                                     dtype=prefix.dtype)), dim=1)
            prepared = model.prepare_inputs_for_generation(prefix, **kwargs)
            output = model(**prepared, return_dict=True)
    return output
