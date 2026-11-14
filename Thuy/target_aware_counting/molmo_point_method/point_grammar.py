"""Validate native generated point-token triples before treating empty as zero."""
from __future__ import annotations


def validate_point_tokens(token_ids: list[int], bounds, extracted_count: int) -> dict:
    real_patch_end = bounds.patch_end_without_no_more_points
    no_more = bounds.no_more_points_token_id
    triples = 0
    stop_count = 0
    i = 0
    while i < len(token_ids):
        token = token_ids[i]
        if bounds.patch_start <= token < real_patch_end:
            if stop_count:
                raise ValueError("point token after no-more-points")
            if i + 2 >= len(token_ids):
                raise ValueError("incomplete point-token triple")
            subpatch, location = token_ids[i+1:i+3]
            if not (bounds.subpatch_start <= subpatch < bounds.subpatch_end) or \
                    not (bounds.location_start <= location < bounds.location_end):
                raise ValueError("invalid point-token triple order")
            triples += 1
            i += 3
            continue
        if token == no_more:
            stop_count += 1
            if stop_count > 1:
                raise ValueError("repeated no-more-points token")
        elif bounds.subpatch_start <= token < bounds.location_end:
            raise ValueError("orphan subpatch/location token")
        i += 1
    if triples != extracted_count:
        raise ValueError(f"decoded point count {extracted_count} differs from {triples} triples")
    return {"point_triples": triples, "no_more_points_tokens": stop_count}
