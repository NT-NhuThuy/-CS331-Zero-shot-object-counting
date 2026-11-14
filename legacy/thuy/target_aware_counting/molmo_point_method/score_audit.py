"""Independent log-probability checks for the frozen MolmoPoint heads."""
from __future__ import annotations

import math

import torch

from .head_scoring import score_step
from .reference_replay import replay


def _log_prob(logits: torch.Tensor, index: int) -> float:
    values = logits.float()
    if index < 0 or index >= values.numel() or not torch.isfinite(values[index]):
        raise ValueError("selected native token is masked")
    return float(values[index] - torch.logsumexp(values, dim=0))


def direct_step_probability(output, prefix, processor, bounds, token_id: int) -> tuple[str, float]:
    """Normalize one native token directly from wrapper and raw head tensors."""
    raw = output.logits[0, -1].detach().float()
    if raw.ndim != 1 or raw.numel() != bounds.location_end:
        raise ValueError("unexpected expanded logit shape")
    last = int(prefix[0, -1])
    if bounds.patch_start <= last < bounds.patch_end:
        if output.subpatch_logits is None or not bounds.subpatch_start <= token_id < bounds.subpatch_end:
            raise ValueError("expected a subpatch head choice")
        kind, start, end = "subpatch", bounds.subpatch_start, bounds.subpatch_end
    elif bounds.subpatch_start <= last < bounds.subpatch_end:
        if output.location_logits is None or not bounds.location_start <= token_id < bounds.location_end:
            raise ValueError("expected a location head choice")
        kind, start, end = "location", bounds.location_start, bounds.location_end
    else:
        kind, start, end = "language", 0, bounds.patch_start

    if kind in ("subpatch", "location"):
        masked = processor(prefix, raw[None, :].clone())[0].float()
        head = output.subpatch_logits if kind == "subpatch" else output.location_logits
        head_values = head[0, -1].detach().float()
        if head_values.numel() != end - start:
            raise ValueError("head width does not match token bounds")
        # Native wrapper may contain finite -100000 sentinels for invalid
        # positions; every selectable position must equal its raw head value.
        valid = torch.isfinite(masked[start:end]) & (masked[start:end] > -10000)
        if not torch.allclose(masked[start:end][valid], head_values[valid], atol=0.1, rtol=0):
            raise ValueError("wrapper/head logits disagree")
        active = masked[start:end].masked_fill(masked[start:end] <= -10000, -float("inf"))
        return kind, _log_prob(active, token_id - start)

    if output.patch_logits is None:
        masked = processor(prefix, raw[None, :].clone())[0].float()
        return "language", _log_prob(masked.masked_fill(masked <= -10000, -float("inf")), token_id)

    # The wrapper writes the LM PATCH_TOKEN logit at its selected patch only.
    # Recover that shared logit, then let the native grammar mask patch choices.
    wrapper_patch = raw[bounds.patch_start:bounds.patch_end]
    present = wrapper_patch > -10000
    if int(present.sum()) != 1:
        raise ValueError("wrapper did not retain exactly one patch logit")
    patch_token_logit = wrapper_patch[present][0]
    patch_head = output.patch_logits[0, -1].detach().float()
    if patch_head.numel() != bounds.patch_end - bounds.patch_start:
        raise ValueError("patch head width does not match token bounds")
    reconstructed = raw.clone()
    reconstructed[bounds.patch_start:bounds.patch_end] = patch_token_logit
    masked = processor(prefix, reconstructed[None, :])[0].float()
    valid_patch = torch.isfinite(masked[bounds.patch_start:bounds.patch_end]) & (patch_head > -10000)
    language = masked[:bounds.patch_start].masked_fill(masked[:bounds.patch_start] <= -10000,
                                                       -float("inf"))
    base = torch.cat((language, patch_token_logit[None])) if valid_patch.any() else language
    if bounds.patch_start <= token_id < bounds.patch_end:
        patch_index = token_id - bounds.patch_start
        if not bool(valid_patch[patch_index]):
            raise ValueError("selected patch is masked")
        patch = patch_head.masked_fill(~valid_patch, -float("inf"))
        kind = "no_more_points" if token_id == bounds.no_more_points_token_id else "patch"
        return kind, _log_prob(base, base.numel()-1) + _log_prob(patch, patch_index)
    if token_id >= bounds.patch_start:
        raise ValueError("expected language or patch choice")
    return "language", _log_prob(base, token_id)


def audit_native_path(model, inputs: dict, token_ids: list[int], *,
                      selected_steps: list[int], tolerance: float = 0.02) -> dict:
    """Compare search scorer and independently normalized native-head scores."""
    if not token_ids or not selected_steps or len(set(selected_steps)) != len(selected_steps):
        raise ValueError("provide a nonempty path and distinct audit steps")
    if any(step < 0 or step >= len(token_ids) for step in selected_steps):
        raise ValueError("audit step outside native path")
    bounds = model.model.build_token_bounds(inputs["image_token_pooling"])
    processor = model.build_logit_processor_from_inputs(inputs)[0]
    records = []
    for step in selected_steps:
        output = replay(model, inputs, token_ids[:step])
        prefix = torch.cat((inputs["input_ids"],
                            torch.tensor([token_ids[:step]], device=inputs["input_ids"].device,
                                         dtype=inputs["input_ids"].dtype)), dim=1)
        chosen = token_ids[step]
        kind, direct = direct_step_probability(output, prefix, processor, bounds, chosen)
        forced = (chosen-bounds.patch_start,) if bounds.patch_start <= chosen < bounds.patch_end else ()
        options = score_step(output, prefix, processor, bounds,
                             max_language=8, max_spatial=16,
                             forced_patch_indices=forced,
                             eos_token_ids=(chosen,) if chosen < bounds.patch_start else ())
        selected = next((option for option in options if option.token_id == chosen), None)
        if selected is None or selected.kind != kind:
            raise ValueError(f"native token absent from scored choices at step {step}")
        delta = abs(selected.log_probability-direct)
        if not math.isfinite(delta) or delta > tolerance:
            raise ValueError(f"head log-probability mismatch at step {step}: {delta}")
        records.append({"step": step, "token_id": chosen, "kind": kind,
                        "direct_log_probability": direct,
                        "search_log_probability": selected.log_probability,
                        "absolute_delta": delta})
    return {"score_audit_pass": True, "steps": records,
            "max_absolute_delta": max(row["absolute_delta"] for row in records),
            "tolerance": tolerance}


def representative_steps(token_ids: list[int], bounds, eos_ids: tuple[int, ...]) -> list[int]:
    """Audit first and last point triples, native stop, and final EOS."""
    groups: dict[str, list[int]] = {key: [] for key in
                                    ("patch", "subpatch", "location", "stop", "eos")}
    for i, token in enumerate(token_ids):
        if token == bounds.no_more_points_token_id:
            groups["stop"].append(i)
        elif bounds.patch_start <= token < bounds.patch_end:
            groups["patch"].append(i)
        elif bounds.subpatch_start <= token < bounds.subpatch_end:
            groups["subpatch"].append(i)
        elif bounds.location_start <= token < bounds.location_end:
            groups["location"].append(i)
        elif token in eos_ids:
            groups["eos"].append(i)
    if not all(groups[key] for key in ("patch", "subpatch", "location", "stop", "eos")):
        raise ValueError("native path lacks a point triple, stop or EOS")
    steps = {i for key in ("patch", "subpatch", "location")
             for i in (groups[key][0], groups[key][-1])}
    steps.update((groups["stop"][-1], groups["eos"][-1]))
    return sorted(steps)
