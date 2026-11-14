"""Unoptimized branch-isolated MolmoPoint spatial search for parity work."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch

from .head_scoring import native_greedy_next, score_step
from .reference_replay import advance_branch, replay, start_branch


@dataclass(frozen=True)
class SearchPath:
    tokens: tuple[int, ...]
    log_probability: float
    complete: bool


def _eos_ids(model) -> tuple[int, ...]:
    ids = model.generation_config.eos_token_id
    return (ids,) if isinstance(ids, int) else tuple(ids or ())


def _context(model, inputs):
    if "image_token_pooling" not in inputs:
        raise ValueError("image_token_pooling required")
    bounds = model.model.build_token_bounds(inputs["image_token_pooling"])
    processor = model.build_logit_processor_from_inputs(inputs)[0]
    return bounds, processor


def native_replay(model, inputs: dict, *, max_new_tokens: int) -> SearchPath:
    """Width-one reference using the exact native collapsed-logit policy."""
    _, processor = _context(model, inputs)
    eos = _eos_ids(model)
    tokens: list[int] = []
    for _ in range(max_new_tokens):
        output = replay(model, inputs, tokens)
        prefix = torch.cat((inputs["input_ids"],
                            torch.tensor([tokens], device=inputs["input_ids"].device,
                                         dtype=inputs["input_ids"].dtype)), dim=1)
        token = native_greedy_next(output, prefix, processor)
        tokens.append(token)
        if token in eos:
            return SearchPath(tuple(tokens), float("nan"), True)
    return SearchPath(tuple(tokens), float("nan"), False)


def spatial_search(model, inputs: dict, *, beam_width: int, max_new_tokens: int,
                   max_language: int, max_spatial: int,
                   guided_patch_indices: tuple[int, ...] = ()) -> tuple[list[SearchPath], dict]:
    """MAP beam over complete token sequences, replaying each prefix independently.

    `guided_patch_indices` only changes which native spatial addresses are
    *explored*. It never changes model log probabilities. Empty/no-more and EOS
    branches remain available. This is deliberately slow and should be used
    for correctness/parity samples before cache-sharing optimizations.
    """
    if beam_width < 1 or max_new_tokens < 1:
        raise ValueError("positive beam and output limits required")
    bounds, processor = _context(model, inputs)
    started = time.monotonic()
    eos = _eos_ids(model)
    active = [SearchPath((), 0.0, False)]
    complete: dict[tuple[int, ...], SearchPath] = {}
    forward_calls = 0
    expanded = 0
    for _ in range(max_new_tokens):
        next_active = []
        for path in active:
            output = replay(model, inputs, list(path.tokens))
            forward_calls += len(path.tokens)+1
            prefix = torch.cat((inputs["input_ids"],
                                torch.tensor([path.tokens], device=inputs["input_ids"].device,
                                             dtype=inputs["input_ids"].dtype)), dim=1)
            options = score_step(
                output, prefix, processor, bounds,
                max_language=max_language, max_spatial=max_spatial,
                forced_patch_indices=guided_patch_indices, eos_token_ids=eos)
            expanded += len(options)
            for option in options:
                tokens = path.tokens + (option.token_id,)
                candidate = SearchPath(tokens,
                                       path.log_probability+option.log_probability,
                                       option.token_id in eos)
                if candidate.complete:
                    old = complete.get(tokens)
                    if old is None or candidate.log_probability > old.log_probability:
                        complete[tokens] = candidate
                else:
                    next_active.append(candidate)
        if not next_active:
            active = []
            break
        next_active.sort(key=lambda path: path.log_probability, reverse=True)
        active = next_active[:beam_width]
    done = sorted(complete.values(), key=lambda path: path.log_probability, reverse=True)
    return done, {"complete_sequences": len(done), "active_truncated": len(active),
                  "replay_forward_calls": forward_calls,
                  "expanded_options": expanded,
                  "elapsed_seconds": time.monotonic()-started,
                  "beam_width": beam_width, "max_new_tokens": max_new_tokens}


def spatial_search_cached(model, inputs: dict, *, beam_width: int, max_new_tokens: int,
                          max_language: int, max_spatial: int,
                          guided_patch_indices: tuple[int, ...] = (),
                          tail_mass_epsilon: float | None = None) -> tuple[list[SearchPath], dict]:
    """Same beam policy as the reference, with one private KV cache per branch.

    Only candidates surviving the beam are advanced. The parent cache is copied
    before every child forward, so sibling paths cannot mutate one another.
    The original replay search remains the correctness oracle.
    """
    if beam_width < 1 or max_new_tokens < 1:
        raise ValueError("positive beam and output limits required")
    if tail_mass_epsilon is not None and not 0 < tail_mass_epsilon < 1:
        raise ValueError("tail mass epsilon must be between zero and one")
    bounds, processor = _context(model, inputs)
    started = time.monotonic()
    eos = _eos_ids(model)
    root = start_branch(model, inputs)
    active = [(SearchPath((), 0.0, False), root)]
    complete: dict[tuple[int, ...], SearchPath] = {}
    forward_calls = 1
    expanded = 0
    cache_clones = 0
    stopped_by_mass_bound = False
    remaining_mass_ratio_bound = None
    for depth in range(max_new_tokens):
        candidates = []
        for path, branch in active:
            options = score_step(
                branch.output, branch.prefix, processor, bounds,
                max_language=max_language, max_spatial=max_spatial,
                forced_patch_indices=guided_patch_indices, eos_token_ids=eos)
            expanded += len(options)
            for option in options:
                tokens = path.tokens + (option.token_id,)
                candidate = SearchPath(tokens,
                                       path.log_probability + option.log_probability,
                                       option.token_id in eos)
                if candidate.complete:
                    old = complete.get(tokens)
                    if old is None or candidate.log_probability > old.log_probability:
                        complete[tokens] = candidate
                else:
                    candidates.append((candidate, branch, option.token_id))
        if not candidates:
            active = []
            break
        if complete and tail_mass_epsilon is not None:
            pivot = max(max(path.log_probability for path, _, _ in candidates),
                        max(path.log_probability for path in complete.values()))
            open_mass = sum(math.exp(path.log_probability-pivot)
                            for path, _, _ in candidates)
            closed_mass = sum(math.exp(path.log_probability-pivot)
                              for path in complete.values())
            remaining_mass_ratio_bound = open_mass / closed_mass
            if remaining_mass_ratio_bound <= tail_mass_epsilon:
                active = []
                stopped_by_mass_bound = True
                break
        candidates.sort(key=lambda row: row[0].log_probability, reverse=True)
        chosen = candidates[:beam_width]
        if depth == max_new_tokens - 1:
            active = [(path, branch) for path, branch, _ in chosen]
            break
        active = []
        for path, parent, token_id in chosen:
            child = advance_branch(model, parent, token_id)
            active.append((path, child))
            forward_calls += 1
            cache_clones += 1
    done = sorted(complete.values(), key=lambda path: path.log_probability, reverse=True)
    return done, {"complete_sequences": len(done), "active_truncated": len(active),
                  "cached_forward_calls": forward_calls, "cache_clones": cache_clones,
                  "stopped_by_mass_bound": stopped_by_mass_bound,
                  "tail_mass_epsilon": tail_mass_epsilon,
                  "remaining_mass_ratio_bound": remaining_mass_ratio_bound,
                  "expanded_options": expanded,
                  "elapsed_seconds": time.monotonic()-started,
                  "beam_width": beam_width, "max_new_tokens": max_new_tokens}
