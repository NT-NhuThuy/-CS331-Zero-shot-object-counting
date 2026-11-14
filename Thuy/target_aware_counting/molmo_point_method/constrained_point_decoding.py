"""Strict serialization grammar for MolmoPoint greedy generation."""
from __future__ import annotations


class StrictPointGrammarProcessor:
    """Keep point choices native while making their text serialization finite."""

    def __init__(self, bounds, prompt_length: int, tokenizer, eos_token_id: int):
        self.bounds = bounds
        self.prompt_length = prompt_length
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.prefix = tokenizer.encode('<points coords="', add_special_tokens=False)
        if not self.prefix:
            raise ValueError("strict point prefix tokenized to an empty sequence")

    @staticmethod
    def _force(scores, batch: int, allowed) -> None:
        import torch
        keep = torch.as_tensor(list(allowed), device=scores.device, dtype=torch.long)
        saved = scores[batch, keep].clone()
        scores[batch].fill_(-float("inf"))
        scores[batch, keep] = saved

    def _number_tokens(self, index: int) -> list[int]:
        tokens = self.tokenizer.encode(str(index), add_special_tokens=False)
        if not tokens:
            raise ValueError(f"point index {index} tokenized to an empty sequence")
        return tokens

    def _allowed_next(self, generated: list[int]):
        b = self.bounds
        prefix_seen = min(len(generated), len(self.prefix))
        if generated[:prefix_seen] != self.prefix[:prefix_seen]:
            raise ValueError("generated output escaped the strict point prefix")
        if len(generated) < len(self.prefix):
            return [self.prefix[len(generated)]]

        tail = generated[len(self.prefix):]
        point_index = 1
        cursor = 0
        while True:
            if cursor == len(tail):
                # Real patch addresses plus the learned no-more-points class.
                return range(b.patch_start, b.patch_end)
            token = tail[cursor]
            if token == b.no_more_points_token_id:
                if cursor == len(tail) - 1:
                    return [self.eos_token_id]
                if tail[cursor + 1:] == [self.eos_token_id]:
                    return [self.eos_token_id]
                raise ValueError("tokens found after strict point terminator")
            if not (b.patch_start <= token < b.patch_end_without_no_more_points):
                raise ValueError("strict point grammar expected patch or terminator")
            cursor += 1
            if cursor == len(tail):
                return range(b.subpatch_start, b.subpatch_end)
            if not (b.subpatch_start <= tail[cursor] < b.subpatch_end):
                raise ValueError("strict point grammar expected subpatch")
            cursor += 1
            if cursor == len(tail):
                return range(b.location_start, b.location_end)
            if not (b.location_start <= tail[cursor] < b.location_end):
                raise ValueError("strict point grammar expected location")
            cursor += 1
            number = self._number_tokens(point_index)
            present = tail[cursor:cursor + len(number)]
            if present != number[:len(present)]:
                raise ValueError("strict point grammar found a non-canonical point index")
            if len(present) < len(number):
                return [number[len(present)]]
            cursor += len(number)
            point_index += 1

    def __call__(self, input_ids, scores):
        for batch in range(len(input_ids)):
            generated = input_ids[batch, self.prompt_length:].tolist()
            self._force(scores, batch, self._allowed_next(generated))
        return scores


def strict_point_processor(model, processor, inputs):
    eos = model.generation_config.eos_token_id
    if not isinstance(eos, int):
        if not eos or len(eos) != 1:
            raise ValueError("strict point grammar requires exactly one EOS token")
        eos = eos[0]
    bounds = model.model.build_token_bounds(inputs["image_token_pooling"])
    return StrictPointGrammarProcessor(
        bounds, int(inputs["input_ids"].shape[1]), processor.tokenizer, eos)
