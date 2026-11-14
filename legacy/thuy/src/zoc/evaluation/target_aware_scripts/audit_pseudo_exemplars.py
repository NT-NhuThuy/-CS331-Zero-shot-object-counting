#!/usr/bin/env python3
"""Audit CountGD++ Forward-1 pseudo-exemplar purity.

This script is intentionally OFFLINE: GT is only used to evaluate the boxes
selected by CountGD++ after Forward 1. No GT is used by the selector itself.

Input:
  --f1-dir  directory produced by CountGD++ with --save_sp1_f1
  --gt      FSC-147 GT JSON (e.g. instances_val.json)

For every image it verifies:
  1. the saved top-K indices exactly reproduce native CountGD++ selection;
  2. selected boxes can be matched to GT with the same IoU-based semantics
     used by the existing SP1 evaluators;
  3. top-1/top-3 purity, distractor contamination, duplicate/merged/ambiguous
     rates, and target coverage are reported.

Important limitation:
  FSC-147 annotations represent the queried counting category. Therefore an
  unmatched selected box is treated as a non-target for this *target-purity*
  audit, but the script cannot name the true semantic category of that box.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

MATCH_IOU = 0.50
AMBIGUOUS_IOU = 0.10
MERGE_GT_COVERAGE = 0.50
EXPECTED_QUERIES = 900


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Audit CountGD++ Forward-1 pseudo-exemplars")
    p.add_argument("--f1-dir", type=Path, required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--top-k", type=str, default="1,2,3")
    p.add_argument("--iou", type=float, default=MATCH_IOU)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_meta(gt_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(x["id"]): x for x in gt_data.get("images", [])}


def build_gt_by_image(gt_data: dict[str, Any]) -> dict[int, np.ndarray]:
    out: dict[int, list[list[float]]] = defaultdict(list)
    for ann in gt_data.get("annotations", []):
        x, y, w, h = map(float, ann["bbox"])
        if w <= 0 or h <= 0:
            continue
        out[int(ann["image_id"])].append([x, y, x + w, y + h])
    return {
        image_id: np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        for image_id, boxes in out.items()
    }


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return np.zeros((pred.shape[0], gt.shape[0]), dtype=np.float64)
    x0 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y0 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x1 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y1 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    pa = np.maximum(0.0, pred[:, 2] - pred[:, 0]) * np.maximum(0.0, pred[:, 3] - pred[:, 1])
    ga = np.maximum(0.0, gt[:, 2] - gt[:, 0]) * np.maximum(0.0, gt[:, 3] - gt[:, 1])
    union = pa[:, None] + ga[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def coverage_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return np.zeros((pred.shape[0], gt.shape[0]), dtype=np.float64)
    x0 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y0 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x1 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y1 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    ga = np.maximum(0.0, gt[:, 2] - gt[:, 0]) * np.maximum(0.0, gt[:, 3] - gt[:, 1])
    return np.where(ga[None, :] > 0, inter / ga[None, :], 0.0)


def classify_candidates(pred: np.ndarray, gt: np.ndarray, iou_threshold: float) -> list[str]:
    if pred.shape[0] == 0:
        return []
    if gt.shape[0] == 0:
        return ["distractor"] * pred.shape[0]

    ious = iou_matrix(pred, gt)
    cov = coverage_matrix(pred, gt)
    rows, cols = linear_sum_assignment(-ious)
    matched = {r for r, c in zip(rows.tolist(), cols.tolist()) if ious[r, c] >= iou_threshold}
    n_covered = (cov >= MERGE_GT_COVERAGE).sum(axis=1)
    best_iou = ious.max(axis=1)
    best_cov = cov.max(axis=1)

    labels = []
    for i in range(pred.shape[0]):
        if n_covered[i] >= 2:
            labels.append("merged")
        elif i in matched:
            labels.append("target")
        elif best_iou[i] >= iou_threshold:
            labels.append("duplicate")
        elif best_iou[i] >= AMBIGUOUS_IOU or best_cov[i] >= AMBIGUOUS_IOU:
            labels.append("ambiguous")
        else:
            labels.append("distractor")
    return labels


def unique_target_matches(pred: np.ndarray, gt: np.ndarray, iou_threshold: float) -> int:
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return 0
    ious = iou_matrix(pred, gt)
    rows, cols = linear_sum_assignment(-ious)
    return sum(float(ious[r, c]) >= iou_threshold for r, c in zip(rows, cols))


def norm_to_xyxy(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    boxes = boxes.astype(np.float64, copy=False)
    cx = boxes[:, 0] * width
    cy = boxes[:, 1] * height
    bw = boxes[:, 2] * width
    bh = boxes[:, 3] * height
    return np.stack(
        [
            np.clip(cx - bw / 2.0, 0, width),
            np.clip(cy - bh / 2.0, 0, height),
            np.clip(cx + bw / 2.0, 0, width),
            np.clip(cy + bh / 2.0, 0, height),
        ],
        axis=1,
    )


def verify_selection_trace(payload: dict[str, Any]) -> None:
    scores = torch.as_tensor(payload["native_scores"], dtype=torch.float32).flatten()
    boxes = torch.as_tensor(payload["pred_boxes_norm"], dtype=torch.float32)
    if scores.numel() != EXPECTED_QUERIES:
        raise RuntimeError(f"{payload['file_name']}: expected 900 scores, got {scores.numel()}")
    if boxes.shape != (EXPECTED_QUERIES, 4):
        raise RuntimeError(f"{payload['file_name']}: expected boxes (900,4), got {tuple(boxes.shape)}")

    threshold = float(payload["box_threshold"])
    expected = torch.nonzero(scores > threshold, as_tuple=False).flatten()
    if expected.numel() > 0:
        order = torch.argsort(scores[expected], descending=True)
        expected = expected[order]
    k = int(payload["num_exemplars_requested"])
    expected_selected = expected[:k]
    actual_selected = torch.as_tensor(payload["selected_indices"], dtype=torch.long).flatten()

    batch_k = int(payload.get("batch_truncation_count", len(actual_selected)))
    expected_selected = expected_selected[:batch_k]
    if not torch.equal(expected_selected, actual_selected):
        raise RuntimeError(
            f"{payload['file_name']}: saved selected_indices do not reproduce "
            f"native threshold/sort/top-K. expected={expected_selected.tolist()} "
            f"actual={actual_selected.tolist()}"
        )

    # Verify score and box copies too.
    actual_scores = torch.as_tensor(payload["selected_scores"], dtype=torch.float32).flatten()
    if not torch.allclose(actual_scores, scores[actual_selected], atol=1e-6, rtol=1e-6):
        raise RuntimeError(f"{payload['file_name']}: selected_scores mismatch native_scores")

    actual_norm = torch.as_tensor(payload["selected_boxes_norm"], dtype=torch.float32)
    if not torch.allclose(actual_norm, boxes[actual_selected], atol=1e-6, rtol=1e-6):
        raise RuntimeError(f"{payload['file_name']}: selected_boxes_norm mismatch pred_boxes_norm")


def aggregate(rows: list[dict[str, Any]], top_ks: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_images": len(rows),
        "top_k": {},
    }
    for k in top_ks:
        usable = [r for r in rows if r["selected_count"] >= k]
        if not usable:
            summary["top_k"][str(k)] = {"num_images": 0}
            continue

        total_slots = k * len(usable)
        strict_target_slots = sum(sum(1 for x in r["labels"][:k] if x == "target") for r in usable)
        targetish_slots = sum(sum(1 for x in r["labels"][:k] if x in {"target", "duplicate"}) for r in usable)
        counts = Counter(x for r in usable for x in r["labels"][:k])

        summary["top_k"][str(k)] = {
            "num_images": len(usable),
            "slot_count": total_slots,
            "strict_target_purity": strict_target_slots / total_slots,
            "target_or_duplicate_fraction": targetish_slots / total_slots,
            "distractor_fraction": counts["distractor"] / total_slots,
            "ambiguous_fraction": counts["ambiguous"] / total_slots,
            "merged_fraction": counts["merged"] / total_slots,
            "duplicate_fraction": counts["duplicate"] / total_slots,
            "at_least_one_target_rate": sum(any(x == "target" for x in r["labels"][:k]) for r in usable) / len(usable),
            "all_strict_target_rate": sum(all(x == "target" for x in r["labels"][:k]) for r in usable) / len(usable),
            "all_target_or_duplicate_rate": sum(all(x in {"target", "duplicate"} for x in r["labels"][:k]) for r in usable) / len(usable),
            "mean_unique_target_matches": float(np.mean([r["unique_target_matches"] for r in usable])),
        }
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_ks = sorted({int(x.strip()) for x in args.top_k.split(",") if x.strip()})
    if not top_ks or any(k <= 0 for k in top_ks):
        raise SystemExit("--top-k must contain positive integers")

    gt_data = load_json(args.gt)
    gt_by_image = build_gt_by_image(gt_data)
    img_by_id = image_meta(gt_data)

    files = sorted(args.f1_dir.glob("*.pt"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No .pt files found under {args.f1_dir}")

    rows: list[dict[str, Any]] = []
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        verify_selection_trace(payload)

        image_id = int(payload["image_id"])
        gt = gt_by_image.get(image_id, np.zeros((0, 4), dtype=np.float64))
        meta = img_by_id.get(image_id, {})

        # The saved XYXY boxes are in resized model-input coordinates.
        # Convert normalized boxes to the original GT image frame for matching.
        if not meta.get("width") or not meta.get("height"):
            raise RuntimeError(f"{payload['file_name']}: original image dimensions unavailable")
        norm = torch.as_tensor(payload["selected_boxes_norm"], dtype=torch.float32).numpy()
        selected = np.asarray(
            norm_to_xyxy(norm, float(meta["width"]), float(meta["height"])),
            dtype=np.float64,
        )
        labels = classify_candidates(selected, gt, args.iou)

        row = {
            "image_id": image_id,
            "file_name": payload["file_name"],
            "query": payload["query"],
            "selected_count": int(selected.shape[0]),
            "requested_k": int(payload["num_exemplars_requested"]),
            "threshold": float(payload["box_threshold"]),
            "batch_truncation_count": int(payload.get("batch_truncation_count", selected.shape[0])),
            "selected_indices": torch.as_tensor(payload["selected_indices"]).tolist(),
            "selected_scores": torch.as_tensor(payload["selected_scores"]).tolist(),
            "labels": labels,
            "strict_target_count": labels.count("target"),
            "duplicate_count": labels.count("duplicate"),
            "distractor_count": labels.count("distractor"),
            "ambiguous_count": labels.count("ambiguous"),
            "merged_count": labels.count("merged"),
            "unique_target_matches": unique_target_matches(selected, gt, args.iou),
            "gt_count": int(gt.shape[0]),
        }
        rows.append(row)

    summary = aggregate(rows, top_ks)
    summary.update(
        {
            "format_version": 1,
            "gt": str(args.gt),
            "f1_dir": str(args.f1_dir),
            "iou": args.iou,
            "selection_trace_verified": True,
        }
    )

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (args.output_dir / "per_image.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {args.output_dir / 'summary.json'}")
    print(f"Wrote: {args.output_dir / 'per_image.json'}")


if __name__ == "__main__":
    main()
