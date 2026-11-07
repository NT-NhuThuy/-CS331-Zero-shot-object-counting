"""All public box inputs are pixel-space xyxy in the actual image."""
import torch
from torchvision.ops import roi_align


def cxcywh_to_xyxy(boxes):
    c, wh = boxes[..., :2], boxes[..., 2:]
    return torch.cat([c - wh / 2, c + wh / 2], dim=-1)


def expand_boxes(boxes, scale, width, height):
    if scale < 1:
        raise ValueError("Context scale must be >= 1")
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    half = (boxes[:, 2:] - boxes[:, :2]) * scale / 2
    expanded = torch.cat([centers - half, centers + half], dim=-1)
    expanded[:, [0, 2]] = expanded[:, [0, 2]].clamp(0, width)
    expanded[:, [1, 3]] = expanded[:, [1, 3]].clamp(0, height)
    return expanded


def region_tokens(feature_maps, strides, boxes_xyxy, size, grid=7, context_scale=1.5):
    """Concat [level0 tight, level0 context, level1 tight, level1 context].

    Features are B=1 before multimodal fusion; use actual architectural strides,
    not feature width / image width (Swin pads internally).
    """
    width, height = size
    if not len(feature_maps) or len(feature_maps) != len(strides):
        raise ValueError("Feature/stride mismatch")
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
        raise ValueError("Expected N x 4 xyxy boxes")
    if not torch.isfinite(boxes_xyxy).all() or (boxes_xyxy[:, 2:] <= boxes_xyxy[:, :2]).any():
        raise ValueError("Nonfinite or zero-area ROI")
    if (boxes_xyxy < 0).any() or (boxes_xyxy[:, [0, 2]] > width).any() or (boxes_xyxy[:, [1, 3]] > height).any():
        raise ValueError("ROI outside image bounds")
    results = []
    context = expand_boxes(boxes_xyxy, context_scale, width, height)
    for feature, stride in zip(feature_maps, strides):
        if feature.ndim != 4 or feature.shape[0] != 1 or stride <= 0:
            raise ValueError("Expected single-image feature map and positive stride")
        for boxes in (boxes_xyxy, context):
            roi = roi_align(feature.float(), [boxes.to(feature.device, torch.float32)],
                            output_size=(grid, grid), spatial_scale=1.0 / stride,
                            sampling_ratio=2, aligned=True)
            results.append(roi.flatten(2).transpose(1, 2))
    return torch.cat(results, dim=1)
