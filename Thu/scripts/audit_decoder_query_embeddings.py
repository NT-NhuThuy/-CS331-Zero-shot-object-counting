#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = os.path.abspath(".")

COUNTGD_ROOT = os.path.join(
    ROOT,
    "third_party",
    "CountGDPlusPlus",
)

sys.path.insert(0, ROOT)
sys.path.insert(0, COUNTGD_ROOT)

import app as countgd_app


META = (
    "generated/presence/"
    "pecv_count_ground_truth.csv"
)

ANNOTATION = (
    "data/fscd147/FSC147_384_V2/"
    "annotations/annotation_FSC147_384.json"
)

IMAGE_ROOT = (
    "data/fscd147/FSC147_384_V2/"
    "images_384_VarV2"
)


# ============================================================
# EXEMPLARS
# ============================================================

def make_positive_prompts(image, annotation):

    points = []

    boxes = annotation[
        "box_examples_coordinates"
    ][:3]

    for box in boxes:

        x1 = float(box[0][0])
        y1 = float(box[0][1])

        x2 = float(box[2][0])
        y2 = float(box[2][1])

        points.append(
            [
                x1,
                y1,
                2.0,
                x2,
                y2,
                3.0,
            ]
        )

    return {
        "image": image,
        "points": points,
    }


def make_empty_prompts(image):

    return {
        "image": image,
        "points": [],
    }


# ============================================================
# GLOBAL CAPTURE
# ============================================================

RAW_CAPTURE = {}
HS_CAPTURE = {}


def capture_raw_prediction(model_output):

    global RAW_CAPTURE

    RAW_CAPTURE = {
        "input_ids":
            model_output["input_ids"][0]
            .detach()
            .cpu(),

        "logits":
            model_output["pred_logits"]
            .sigmoid()[0]
            .detach()
            .cpu(),

        "boxes":
            model_output["pred_boxes"][0]
            .detach()
            .cpu(),
    }

    # predict() needs something returned,
    # but we do not need filtered boxes here.
    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )


def decoder_hook(module, inputs):

    global HS_CAPTURE

    # class_embed is called as:
    # layer_cls_embed(layer_hs, text_dict)
    #
    # Therefore inputs[0] = layer_hs
    hs = inputs[0]

    HS_CAPTURE["hs"] = (
        hs.detach()
        .cpu()
    )


# ============================================================
# BOX UTILITIES
# ============================================================

def cxcywh_to_xyxy(boxes, width, height):

    boxes = np.asarray(
        boxes,
        dtype=np.float32,
    )

    cx = boxes[:, 0] * width
    cy = boxes[:, 1] * height

    bw = boxes[:, 2] * width
    bh = boxes[:, 3] * height

    x1 = cx - bw / 2
    y1 = cy - bh / 2

    x2 = cx + bw / 2
    y2 = cy + bh / 2

    return np.stack(
        [x1, y1, x2, y2],
        axis=1,
    )


def inside_gt_count(boxes, points):

    if len(points) == 0:

        return np.zeros(
            len(boxes),
            dtype=np.int32,
        )

    pts = np.asarray(
        points,
        dtype=np.float32,
    )

    px = pts[:, 0][None, :]
    py = pts[:, 1][None, :]

    x1 = boxes[:, 0][:, None]
    y1 = boxes[:, 1][:, None]

    x2 = boxes[:, 2][:, None]
    y2 = boxes[:, 3][:, None]

    inside = (
        (px >= x1)
        &
        (px <= x2)
        &
        (py >= y1)
        &
        (py <= y2)
    )

    return inside.sum(
        axis=1
    ).astype(np.int32)


# ============================================================
# OOF LINEAR PROBE
# ============================================================

def oof_probe(
    X,
    y,
    groups,
):

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=int,
    )

    groups = np.asarray(
        groups,
    )

    gkf = GroupKFold(
        n_splits=5
    )

    oof_prob = np.zeros(
        len(y),
        dtype=float,
    )

    for train_idx, test_idx in gkf.split(
        X,
        y,
        groups,
    ):

        model = Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        )

        model.fit(
            X[train_idx],
            y[train_idx],
        )

        oof_prob[test_idx] = (
            model.predict_proba(
                X[test_idx]
            )[:, 1]
        )

    auc = roc_auc_score(
        y,
        oof_prob,
    )

    ap = average_precision_score(
        y,
        oof_prob,
    )

    return (
        oof_prob,
        auc,
        ap,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protocol",
        choices=[
            "text_only",
            "pecv",
        ],
        default="text_only",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.23,
    )

    parser.add_argument(
        "--out",
        default=(
            "outputs/"
            "pecv_countgd_baseline/"
            "decoder_embedding_audit"
        ),
    )

    args = parser.parse_args()

    out_dir = (
        Path(args.out)
        /
        args.protocol
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "COUNTGD++ DECODER QUERY EMBEDDING AUDIT"
    )
    print("=" * 100)

    print(
        "Protocol:",
        args.protocol,
    )

    print(
        "Retention threshold:",
        args.threshold,
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    meta = pd.read_csv(
        META
    )

    if args.limit > 0:

        meta = (
            meta
            .head(args.limit)
            .copy()
        )

    with open(
        ANNOTATION,
        "r",
    ) as f:

        annotations = json.load(f)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    app_args = (
        countgd_app
        .get_args_parser()
        .parse_args([])
    )

    device = (
        countgd_app
        .get_device()
    )

    old_cwd = os.getcwd()

    try:

        os.chdir(
            COUNTGD_ROOT
        )

        app_args.pretrain_model_path = (
            "checkpoints/"
            "countgd_plusplus.pth"
        )

        model, transform = (
            countgd_app
            .build_model_and_transforms(
                app_args
            )
        )

    finally:

        os.chdir(
            old_cwd
        )

    model = model.to(device)
    model.eval()

    print(
        "Device:",
        device,
    )

    # --------------------------------------------------------
    # Hook final classification head.
    #
    # The last class_embed receives hs[-1].
    # --------------------------------------------------------

    hook_handle = (
        model.class_embed[-1]
        .register_forward_pre_hook(
            decoder_hook
        )
    )

    original_get_boxes = (
        countgd_app
        .get_boxes_from_prediction
    )

    countgd_app.get_boxes_from_prediction = (
        capture_raw_prediction
    )

    all_embeddings = []
    rows = []

    try:

        for idx, row in meta.iterrows():

            file_name = str(
                row["file_name"]
            )

            query = str(
                row["query"]
            )

            gt_presence = int(
                row["gt_presence"]
            )

            gt_count = int(
                row["gt_count"]
            )

            image = (
                Image.open(
                    os.path.join(
                        IMAGE_ROOT,
                        file_name,
                    )
                )
                .convert("RGB")
            )

            annotation = (
                annotations.get(
                    file_name
                )
            )

            if args.protocol == "text_only":

                positive_prompts = (
                    make_empty_prompts(
                        image
                    )
                )

            else:

                if (
                    gt_presence == 1
                    and
                    annotation is not None
                ):

                    positive_prompts = (
                        make_positive_prompts(
                            image,
                            annotation,
                        )
                    )

                else:

                    positive_prompts = (
                        make_empty_prompts(
                            image
                        )
                    )

            RAW_CAPTURE.clear()
            HS_CAPTURE.clear()

            with torch.no_grad():

                countgd_app.predict(
                    model,
                    transform,
                    image,
                    query,
                    positive_prompts,
                    [],
                    [],
                    device,
                )

            if not RAW_CAPTURE:

                raise RuntimeError(
                    "Raw output capture failed"
                )

            if not HS_CAPTURE:

                raise RuntimeError(
                    "Decoder hook failed"
                )

            input_ids = (
                RAW_CAPTURE[
                    "input_ids"
                ]
                .numpy()
            )

            logits = (
                RAW_CAPTURE[
                    "logits"
                ]
                .numpy()
            )

            boxes_norm = (
                RAW_CAPTURE[
                    "boxes"
                ]
                .numpy()
            )

            hs = (
                HS_CAPTURE["hs"]
                .numpy()
            )

            # Expected:
            # [batch, 900, hidden_dim]
            if hs.ndim == 3:

                hs = hs[0]

            if hs.shape[0] != logits.shape[0]:

                raise RuntimeError(
                    f"HS/logit mismatch: "
                    f"{hs.shape} vs "
                    f"{logits.shape}"
                )

            # --------------------------------------------
            # Positive text region
            # --------------------------------------------

            period_positions = np.where(
                input_ids == 1012
            )[0]

            if len(period_positions) == 0:

                raise RuntimeError(
                    "No '.' token found"
                )

            split_idx = int(
                period_positions[0]
            )

            pos_official = (
                logits[
                    :,
                    : split_idx + 1,
                ]
                .max(axis=1)
            )

            if split_idx > 1:

                pos_content = (
                    logits[
                        :,
                        1:split_idx,
                    ]
                    .max(axis=1)
                )

            else:

                pos_content = (
                    pos_official.copy()
                )

            keep = (
                pos_official
                >
                args.threshold
            )

            # --------------------------------------------
            # Geometry
            # --------------------------------------------

            width, height = image.size

            boxes_xyxy = (
                cxcywh_to_xyxy(
                    boxes_norm,
                    width,
                    height,
                )
            )

            if (
                gt_presence == 1
                and
                annotation is not None
            ):

                inside = (
                    inside_gt_count(
                        boxes_xyxy,
                        annotation["points"],
                    )
                )

            else:

                inside = np.zeros(
                    len(boxes_xyxy),
                    dtype=np.int32,
                )

            kept_indices = np.where(
                keep
            )[0]

            for qi in kept_indices:

                emb_idx = len(
                    all_embeddings
                )

                all_embeddings.append(
                    hs[qi].astype(
                        np.float32
                    )
                )

                if gt_presence == 0:

                    geometry_status = (
                        "ABSENT_QUERY"
                    )

                elif inside[qi] == 0:

                    geometry_status = (
                        "NO_GT_INSIDE"
                    )

                elif inside[qi] == 1:

                    geometry_status = (
                        "ONE_GT_INSIDE"
                    )

                else:

                    geometry_status = (
                        "MULTI_GT_INSIDE"
                    )

                rows.append(
                    {
                        "embedding_index":
                            emb_idx,

                        "file_name":
                            file_name,

                        "query":
                            query,

                        "gt_presence":
                            gt_presence,

                        "gt_count":
                            gt_count,

                        "query_index":
                            int(qi),

                        "pos_official_score":
                            float(
                                pos_official[qi]
                            ),

                        "pos_content_score":
                            float(
                                pos_content[qi]
                            ),

                        "inside_gt_count":
                            int(
                                inside[qi]
                            ),

                        "geometry_status":
                            geometry_status,
                    }
                )

            print(
                f"[{idx + 1:03d}/{len(meta):03d}] "
                f"{file_name:>10} | "
                f"{query:<25} | "
                f"P={gt_presence} | "
                f"kept={len(kept_indices):3d}"
            )

    finally:

        hook_handle.remove()

        countgd_app.get_boxes_from_prediction = (
            original_get_boxes
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    embeddings = np.stack(
        all_embeddings,
        axis=0,
    )

    info = pd.DataFrame(
        rows
    )

    emb_path = (
        out_dir
        /
        "decoder_embeddings.npy"
    )

    info_path = (
        out_dir
        /
        "decoder_embedding_metadata.csv"
    )

    np.save(
        emb_path,
        embeddings,
    )

    info.to_csv(
        info_path,
        index=False,
    )

    print()
    print("=" * 100)
    print("CAPTURED")
    print("=" * 100)

    print(
        "Embeddings:",
        embeddings.shape,
    )

    print(
        "Metadata rows:",
        len(info),
    )

    # ========================================================
    # EXPERIMENT A:
    # TARGET-BEARING POSITIVE
    # vs
    # SAFE ABSENT NEGATIVE
    #
    # Positive-image NO_GT boxes are EXCLUDED because
    # they may be partial/ambiguous targets.
    # ========================================================

    valid_mask = (
        (
            (info["gt_presence"] == 1)
            &
            (info["inside_gt_count"] > 0)
        )
        |
        (
            info["gt_presence"] == 0
        )
    )

    valid = (
        info[
            valid_mask
        ]
        .copy()
    )

    valid_indices = (
        valid[
            "embedding_index"
        ]
        .to_numpy(int)
    )

    X = embeddings[
        valid_indices
    ]

    y = (
        (
            valid[
                "gt_presence"
            ] == 1
        )
        .astype(int)
        .to_numpy()
    )

    groups = (
        valid[
            "file_name"
        ]
        .to_numpy()
    )

    scalar = (
        valid[
            "pos_content_score"
        ]
        .to_numpy(float)
    )

    baseline_auc = (
        roc_auc_score(
            y,
            scalar,
        )
    )

    baseline_ap = (
        average_precision_score(
            y,
            scalar,
        )
    )

    _, emb_auc, emb_ap = (
        oof_probe(
            X,
            y,
            groups,
        )
    )

    # Embedding + scalar score
    X_plus = np.concatenate(
        [
            X,
            scalar[:, None],
        ],
        axis=1,
    )

    _, plus_auc, plus_ap = (
        oof_probe(
            X_plus,
            y,
            groups,
        )
    )

    print()
    print("=" * 100)
    print(
        "EXPERIMENT A — TARGET VALIDITY"
    )
    print("=" * 100)

    print(
        "Positive target-bearing:",
        int(y.sum()),
    )

    print(
        "Safe absent negatives:",
        int((y == 0).sum()),
    )

    print()

    print(
        "Scalar score ROC-AUC:",
        round(
            baseline_auc,
            6,
        ),
    )

    print(
        "Scalar score AP:",
        round(
            baseline_ap,
            6,
        ),
    )

    print()

    print(
        "Decoder embedding OOF ROC-AUC:",
        round(
            emb_auc,
            6,
        ),
    )

    print(
        "Decoder embedding OOF AP:",
        round(
            emb_ap,
            6,
        ),
    )

    print()

    print(
        "Embedding + scalar OOF ROC-AUC:",
        round(
            plus_auc,
            6,
        ),
    )

    print(
        "Embedding + scalar OOF AP:",
        round(
            plus_ap,
            6,
        ),
    )

    # ========================================================
    # EXPERIMENT B:
    # ONE INSTANCE vs MULTI INSTANCE
    # ========================================================

    atomic = info[
        (
            info[
                "gt_presence"
            ] == 1
        )
        &
        (
            info[
                "geometry_status"
            ].isin(
                [
                    "ONE_GT_INSIDE",
                    "MULTI_GT_INSIDE",
                ]
            )
        )
    ].copy()

    atomic_indices = (
        atomic[
            "embedding_index"
        ]
        .to_numpy(int)
    )

    Xa = embeddings[
        atomic_indices
    ]

    # 1 = ONE object
    # 0 = MULTI object
    ya = (
        (
            atomic[
                "geometry_status"
            ]
            ==
            "ONE_GT_INSIDE"
        )
        .astype(int)
        .to_numpy()
    )

    ga = (
        atomic[
            "file_name"
        ]
        .to_numpy()
    )

    scalar_a = (
        atomic[
            "pos_content_score"
        ]
        .to_numpy(float)
    )

    scalar_a_auc = (
        roc_auc_score(
            ya,
            scalar_a,
        )
    )

    _, atomic_emb_auc, _ = (
        oof_probe(
            Xa,
            ya,
            ga,
        )
    )

    Xa_plus = np.concatenate(
        [
            Xa,
            scalar_a[:, None],
        ],
        axis=1,
    )

    _, atomic_plus_auc, _ = (
        oof_probe(
            Xa_plus,
            ya,
            ga,
        )
    )

    print()
    print("=" * 100)
    print(
        "EXPERIMENT B — INSTANCE ATOMICITY"
    )
    print("=" * 100)

    print(
        "ONE-GT:",
        int(ya.sum()),
    )

    print(
        "MULTI-GT:",
        int((ya == 0).sum()),
    )

    print()

    print(
        "Scalar ONE-vs-MULTI ROC-AUC:",
        round(
            scalar_a_auc,
            6,
        ),
    )

    print(
        "Decoder embedding OOF ROC-AUC:",
        round(
            atomic_emb_auc,
            6,
        ),
    )

    print(
        "Embedding + scalar OOF ROC-AUC:",
        round(
            atomic_plus_auc,
            6,
        ),
    )

    print()
    print("=" * 100)
    print("OUTPUT")
    print("=" * 100)

    print(
        emb_path
    )

    print(
        info_path
    )

    print()
    print("DONE")


if __name__ == "__main__":
    main()
