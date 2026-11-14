#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = (
    ROOT
    / "generated/joint_semantic_counting_v1"
    / "s1_proposal_train_v2"
)

DEV_DIR = (
    ROOT
    / "generated/joint_semantic_counting_v1"
    / "s1_proposal_dev_v2"
)

TRAIN_PAIRS = (
    ROOT
    / "generated/joint_semantic_counting_v1"
    / "lvis_train_semantic_pairs_v1.csv"
)

DEV_PAIRS = (
    ROOT
    / "generated/joint_semantic_counting_v1"
    / "lvis_semantic_dev_pairs_v1.csv"
)

CLIP_TRAIN = (
    ROOT
    / "generated/joint_semantic_counting_v1"
    / "clip_train_scores_v1.csv"
)

CLIP_DEV = (
    ROOT
    / "outputs/joint_semantic_counting_v1"
    / "s0_clip_dev_v1"
    / "clip_dev_scores_v1.csv"
)

OUT_ROOT = (
    ROOT
    / "outputs/joint_semantic_counting_v1"
    / "s1_verifier_a3_clip_v1"
)

EXPECTED_TRAIN_PAIR_SHA = (
    "ca0044258c74ad2706c7e066528010b9"
    "35c375f542b51426291f7e6693c35808"
)

EXPECTED_DEV_PAIR_SHA = (
    "d6a81892c3fdf5efc913e97f3e66e93"
    "d2e0943178d12afcebd9da90cc4de610e"
)

HS_DIM = 256


def sha256_file(path):
    import hashlib

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            x = f.read(1024 * 1024)

            if not x:
                break

            h.update(x)

    return h.hexdigest()


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metadata_files(root):

    return sorted(
        root.glob(
            "shard_*_metadata.csv"
        )
    )


def hs_path_from_meta(meta_path):

    return Path(
        str(meta_path).replace(
            "_metadata.csv",
            "_hs.npy",
        )
    )


def validate_pair_identity(
    source,
    scored,
    name,
):

    keys = [
        "pair_group_id",
        "image_id",
        "file_name",
        "label_present",
        "category_id",
        "query",
    ]

    if len(source) != len(scored):
        raise RuntimeError(
            f"{name}: row mismatch "
            f"{len(source)} vs {len(scored)}"
        )

    for c in keys:

        if c not in scored.columns:
            raise RuntimeError(
                f"{name}: missing column {c}"
            )

        a = (
            source[c]
            .astype(str)
            .to_numpy()
        )

        b = (
            scored[c]
            .astype(str)
            .to_numpy()
        )

        if not np.array_equal(
            a,
            b,
        ):
            bad = np.where(
                a != b
            )[0][0]

            raise RuntimeError(
                f"{name}: pair identity mismatch "
                f"at row {bad}, column {c}: "
                f"{a[bad]} != {b[bad]}"
            )


def load_clip_contract():

    if (
        sha256_file(TRAIN_PAIRS)
        !=
        EXPECTED_TRAIN_PAIR_SHA
    ):
        raise RuntimeError(
            "Frozen TRAIN pair SHA mismatch."
        )

    if (
        sha256_file(DEV_PAIRS)
        !=
        EXPECTED_DEV_PAIR_SHA
    ):
        raise RuntimeError(
            "Frozen DEV pair SHA mismatch."
        )

    train_source = pd.read_csv(
        TRAIN_PAIRS
    )

    dev_source = pd.read_csv(
        DEV_PAIRS
    )

    clip_train = pd.read_csv(
        CLIP_TRAIN
    )

    clip_dev = pd.read_csv(
        CLIP_DEV
    )

    if len(train_source) != 20000:
        raise RuntimeError(
            "Expected 20,000 TRAIN pairs."
        )

    if len(dev_source) != 2000:
        raise RuntimeError(
            "Expected 2,000 DEV pairs."
        )

    validate_pair_identity(
        train_source,
        clip_train,
        "TRAIN CLIP",
    )

    validate_pair_identity(
        dev_source,
        clip_dev,
        "DEV CLIP",
    )

    if not np.isfinite(
        clip_train[
            "clip_cosine"
        ].to_numpy()
    ).all():

        raise RuntimeError(
            "Non-finite TRAIN CLIP scores."
        )

    if not np.isfinite(
        clip_dev[
            "clip_cosine"
        ].to_numpy()
    ).all():

        raise RuntimeError(
            "Non-finite DEV CLIP scores."
        )

    # Pair-level normalization only.
    # Do NOT proposal-weight the mean/std.
    train_raw = (
        clip_train[
            "clip_cosine"
        ]
        .astype(np.float32)
        .to_numpy()
    )

    dev_raw = (
        clip_dev[
            "clip_cosine"
        ]
        .astype(np.float32)
        .to_numpy()
    )

    clip_mean = float(
        train_raw.mean()
    )

    clip_std = float(
        train_raw.std()
    )

    if clip_std <= 0:
        raise RuntimeError(
            "Invalid TRAIN CLIP std."
        )

    train_z = (
        (train_raw - clip_mean)
        /
        clip_std
    ).astype(
        np.float32
    )

    dev_z = (
        (dev_raw - clip_mean)
        /
        clip_std
    ).astype(
        np.float32
    )

    return (
        train_raw,
        dev_raw,
        train_z,
        dev_z,
        clip_mean,
        clip_std,
    )


def compute_pair_weights(meta):

    labels = (
        meta[
            "semantic_label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    pair_ids = (
        meta[
            "pair_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    weights = np.zeros(
        len(meta),
        dtype=np.float32,
    )

    for pid in np.unique(
        pair_ids
    ):

        idx = np.where(
            pair_ids == pid
        )[0]

        y = labels[
            idx
        ]

        pos = idx[
            y == 1
        ]

        neg = idx[
            y == 0
        ]

        if len(pos) and len(neg):

            weights[pos] = (
                0.5
                /
                len(pos)
            )

            weights[neg] = (
                0.5
                /
                len(neg)
            )

        elif len(pos):

            weights[pos] = (
                1.0
                /
                len(pos)
            )

        elif len(neg):

            weights[neg] = (
                1.0
                /
                len(neg)
            )

        else:

            raise RuntimeError(
                f"Pair {pid} has no "
                "clean training rows."
            )

    return weights


class A3SemanticVerifier(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.hs_norm = nn.LayerNorm(
            HS_DIM
        )

        self.net = nn.Sequential(
            nn.Linear(
                HS_DIM + 1,
                128,
            ),
            nn.GELU(),
            nn.Dropout(
                0.10
            ),

            nn.Linear(
                128,
                64,
            ),
            nn.GELU(),
            nn.Dropout(
                0.10
            ),

            nn.Linear(
                64,
                1,
            ),
        )

    def forward(
        self,
        hs,
        clip_z,
    ):

        h = self.hs_norm(
            hs
        )

        x = torch.cat(
            [
                h,
                clip_z.unsqueeze(1),
            ],
            dim=1,
        )

        return (
            self.net(x)
            .squeeze(1)
        )


def train_one_epoch(
    model,
    train_files,
    train_clip_z,
    optimizer,
    device,
    batch_size,
    epoch_seed,
):

    model.train()

    rng = np.random.default_rng(
        epoch_seed
    )

    order = np.arange(
        len(train_files)
    )

    rng.shuffle(order)

    total_weighted_loss = 0.0
    total_weight = 0.0

    for shard_step, file_idx in enumerate(
        order
    ):

        meta_path = train_files[
            int(file_idx)
        ]

        meta = pd.read_csv(
            meta_path
        )

        labels = (
            meta[
                "semantic_label"
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        if not np.all(
            np.isin(
                labels,
                [0, 1],
            )
        ):
            raise RuntimeError(
                "TRAIN contains "
                "non-clean labels."
            )

        pair_ids = (
            meta[
                "pair_index"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        if (
            pair_ids.min() < 0
            or
            pair_ids.max() >= 20000
        ):
            raise RuntimeError(
                "Invalid TRAIN pair_index."
            )

        clip_rows = (
            train_clip_z[
                pair_ids
            ]
        )

        weights = compute_pair_weights(
            meta
        )

        hs = np.load(
            hs_path_from_meta(
                meta_path
            ),
            mmap_mode="r",
        )

        if len(hs) != len(meta):
            raise RuntimeError(
                f"TRAIN HS/meta mismatch: "
                f"{meta_path.name}"
            )

        row_order = rng.permutation(
            len(meta)
        )

        for start in range(
            0,
            len(meta),
            batch_size,
        ):

            idx = row_order[
                start:
                start + batch_size
            ]

            hs_t = torch.from_numpy(
                np.asarray(
                    hs[idx],
                    dtype=np.float32,
                )
            ).to(
                device
            )

            clip_t = torch.from_numpy(
                clip_rows[idx]
            ).to(
                device
            )

            y = torch.from_numpy(
                labels[idx]
            ).to(
                device
            )

            w = torch.from_numpy(
                weights[idx]
            ).to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                hs_t,
                clip_t,
            )

            loss_raw = (
                nn.functional
                .binary_cross_entropy_with_logits(
                    logits,
                    y,
                    reduction="none",
                )
            )

            loss = (
                loss_raw * w
            ).sum() / (
                w.sum()
                +
                1e-12
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            total_weighted_loss += float(
                (
                    loss_raw.detach()
                    *
                    w
                ).sum()
                .cpu()
            )

            total_weight += float(
                w.sum()
                .cpu()
            )

        if (
            (shard_step + 1) % 20 == 0
            or
            shard_step + 1
            ==
            len(order)
        ):

            print(
                f"[TRAIN "
                f"{shard_step+1}/"
                f"{len(order)}]"
            )

    return (
        total_weighted_loss
        /
        max(
            total_weight,
            1e-12,
        )
    )


@torch.no_grad()
def predict_dev(
    model,
    dev_files,
    dev_clip_z,
    device,
    batch_size,
    max_dev_shards=0,
):

    model.eval()

    score_parts = []
    label_parts = []
    pair_parts = []
    presence_parts = []
    native_parts = []

    files = dev_files

    if max_dev_shards > 0:
        files = files[
            :max_dev_shards
        ]

    for shard_idx, meta_path in enumerate(
        files
    ):

        meta = pd.read_csv(
            meta_path
        )

        hs = np.load(
            hs_path_from_meta(
                meta_path
            ),
            mmap_mode="r",
        )

        if len(meta) != len(hs):
            raise RuntimeError(
                f"DEV HS/meta mismatch: "
                f"{meta_path.name}"
            )

        pair_ids = (
            meta[
                "pair_index"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        if (
            pair_ids.min() < 0
            or
            pair_ids.max() >= 2000
        ):
            raise RuntimeError(
                "Invalid DEV pair_index."
            )

        clip_rows = (
            dev_clip_z[
                pair_ids
            ]
        )

        shard_scores = []

        for start in range(
            0,
            len(meta),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(meta),
            )

            hs_t = torch.from_numpy(
                np.asarray(
                    hs[
                        start:end
                    ],
                    dtype=np.float32,
                )
            ).to(
                device
            )

            clip_t = torch.from_numpy(
                clip_rows[
                    start:end
                ]
            ).to(
                device
            )

            logits = model(
                hs_t,
                clip_t,
            )

            probs = torch.sigmoid(
                logits
            )

            shard_scores.append(
                probs
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

        score_parts.append(
            np.concatenate(
                shard_scores
            )
        )

        label_parts.append(
            meta[
                "semantic_label"
            ]
            .to_numpy(
                dtype=np.int8
            )
        )

        pair_parts.append(
            pair_ids.astype(
                np.int32
            )
        )

        presence_parts.append(
            meta[
                "pair_label_present"
            ]
            .to_numpy(
                dtype=np.int8
            )
        )

        native_parts.append(
            meta[
                "native_pass_final"
            ]
            .astype(str)
            .str.lower()
            .eq("true")
            .to_numpy(
                dtype=bool
            )
        )

        if (
            (shard_idx + 1) % 10 == 0
            or
            shard_idx + 1
            ==
            len(files)
        ):

            print(
                f"[DEV "
                f"{shard_idx+1}/"
                f"{len(files)}]"
            )

    scores = np.concatenate(
        score_parts
    )

    labels = np.concatenate(
        label_parts
    )

    pair_ids = np.concatenate(
        pair_parts
    )

    pair_presence_rows = np.concatenate(
        presence_parts
    )

    native = np.concatenate(
        native_parts
    )

    # --------------------------------------------------------
    # Proposal metrics — clean labels only
    # --------------------------------------------------------

    clean = labels >= 0

    y = labels[
        clean
    ].astype(
        np.int64
    )

    s = scores[
        clean
    ]

    proposal_auc = float(
        roc_auc_score(
            y,
            s,
        )
    )

    proposal_ap = float(
        average_precision_score(
            y,
            s,
        )
    )

    # --------------------------------------------------------
    # Pair-level max proposal score
    # --------------------------------------------------------

    num_pairs = int(
        pair_ids.max()
        +
        1
    )

    pair_score = np.full(
        num_pairs,
        -np.inf,
        dtype=np.float32,
    )

    np.maximum.at(
        pair_score,
        pair_ids,
        scores,
    )

    pair_y = np.zeros(
        num_pairs,
        dtype=np.int8,
    )

    _, first_idx = np.unique(
        pair_ids,
        return_index=True,
    )

    pair_y[
        pair_ids[
            first_idx
        ]
    ] = pair_presence_rows[
        first_idx
    ]

    valid_pair = np.isfinite(
        pair_score
    )

    pair_score_eval = pair_score[
        valid_pair
    ]

    pair_y_eval = pair_y[
        valid_pair
    ]

    pair_auc = float(
        roc_auc_score(
            pair_y_eval,
            pair_score_eval,
        )
    )

    pair_ap = float(
        average_precision_score(
            pair_y_eval,
            pair_score_eval,
        )
    )

    # DEV-selected operating point.
    fpr, tpr, thresholds = (
        roc_curve(
            pair_y_eval,
            pair_score_eval,
        )
    )

    ba = (
        tpr
        +
        (1.0 - fpr)
    ) / 2.0

    best_idx = int(
        np.argmax(
            ba
        )
    )

    threshold = float(
        thresholds[
            best_idx
        ]
    )

    pair_pred = (
        pair_score_eval
        >=
        threshold
    )

    present_mask = (
        pair_y_eval == 1
    )

    absent_mask = (
        pair_y_eval == 0
    )

    present_recall = float(
        pair_pred[
            present_mask
        ].mean()
    )

    absent_rejection = float(
        (
            ~pair_pred[
                absent_mask
            ]
        ).mean()
    )

    balanced_accuracy = float(
        (
            present_recall
            +
            absent_rejection
        )
        /
        2.0
    )

    # --------------------------------------------------------
    # Proposal selection at same threshold
    # --------------------------------------------------------

    selected = (
        scores
        >=
        threshold
    )

    selected_count = np.bincount(
        pair_ids,
        weights=selected.astype(
            np.int32
        ),
        minlength=num_pairs,
    )

    absent_pair_ids = np.where(
        pair_y == 0
    )[0]

    present_pair_ids = np.where(
        pair_y == 1
    )[0]

    absent_zero_accuracy = float(
        (
            selected_count[
                absent_pair_ids
            ]
            ==
            0
        ).mean()
    )

    absent_mean_selected = float(
        selected_count[
            absent_pair_ids
        ].mean()
    )

    # --------------------------------------------------------
    # Clean positive recovery
    # --------------------------------------------------------

    positive_rows = (
        labels == 1
    )

    pos_available = np.zeros(
        num_pairs,
        dtype=bool,
    )

    pos_recovered = np.zeros(
        num_pairs,
        dtype=bool,
    )

    np.logical_or.at(
        pos_available,
        pair_ids[
            positive_rows
        ],
        True,
    )

    np.logical_or.at(
        pos_recovered,
        pair_ids[
            positive_rows
        ],
        selected[
            positive_rows
        ],
    )

    available_present = (
        present_pair_ids[
            pos_available[
                present_pair_ids
            ]
        ]
    )

    clean_positive_pair_recovery = float(
        pos_recovered[
            available_present
        ].mean()
    )

    # --------------------------------------------------------
    # Native reference
    # --------------------------------------------------------

    native_selected_count = np.bincount(
        pair_ids,
        weights=native.astype(
            np.int32
        ),
        minlength=num_pairs,
    )

    native_pair_present = (
        native_selected_count
        >
        0
    )

    native_present_recall = float(
        native_pair_present[
            present_pair_ids
        ].mean()
    )

    native_absent_rejection = float(
        (
            ~native_pair_present[
                absent_pair_ids
            ]
        ).mean()
    )

    native_pos_recovered = np.zeros(
        num_pairs,
        dtype=bool,
    )

    np.logical_or.at(
        native_pos_recovered,
        pair_ids[
            positive_rows
        ],
        native[
            positive_rows
        ],
    )

    native_clean_positive_pair_recovery = float(
        native_pos_recovered[
            available_present
        ].mean()
    )

    return {
        "proposal_clean_count":
            int(
                clean.sum()
            ),

        "proposal_positive_count":
            int(
                (
                    labels[
                        clean
                    ]
                    ==
                    1
                ).sum()
            ),

        "proposal_negative_count":
            int(
                (
                    labels[
                        clean
                    ]
                    ==
                    0
                ).sum()
            ),

        "proposal_auroc":
            proposal_auc,

        "proposal_ap":
            proposal_ap,

        "pair_count":
            int(
                valid_pair.sum()
            ),

        "pair_auroc":
            pair_auc,

        "pair_ap":
            pair_ap,

        "dev_selected_threshold":
            threshold,

        "pair_balanced_accuracy":
            balanced_accuracy,

        "pair_present_recall":
            present_recall,

        "pair_absent_rejection":
            absent_rejection,

        "absent_zero_proposal_accuracy":
            absent_zero_accuracy,

        "absent_mean_selected_proposals":
            absent_mean_selected,

        "clean_positive_available_present_pairs":
            int(
                len(
                    available_present
                )
            ),

        "clean_positive_pair_recovery":
            clean_positive_pair_recovery,

        "native_present_recall":
            native_present_recall,

        "native_absent_rejection":
            native_absent_rejection,

        "native_clean_positive_pair_recovery":
            native_clean_positive_pair_recovery,
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--max-train-shards",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--max-dev-shards",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--out",
        type=Path,
        default=None,
    )

    args = ap.parse_args()

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_files = metadata_files(
        TRAIN_DIR
    )

    dev_files = metadata_files(
        DEV_DIR
    )

    if len(train_files) != 100:
        raise RuntimeError(
            f"Expected 100 TRAIN shards, "
            f"got {len(train_files)}"
        )

    if len(dev_files) != 40:
        raise RuntimeError(
            f"Expected 40 DEV shards, "
            f"got {len(dev_files)}"
        )

    if args.max_train_shards > 0:
        train_files = train_files[
            :args.max_train_shards
        ]

    (
        train_clip_raw,
        dev_clip_raw,
        train_clip_z,
        dev_clip_z,
        clip_mean,
        clip_std,
    ) = load_clip_contract()

    out_dir = (
        args.out
        if args.out is not None
        else
        OUT_ROOT
        /
        f"s1_a3_seed{args.seed}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "S1-A3 — DECODER HS + GLOBAL CLIP"
    )
    print("=" * 100)

    print(
        "Device:",
        device,
    )

    print(
        "TRAIN shards:",
        len(train_files),
    )

    print(
        "DEV shards:",
        (
            args.max_dev_shards
            if args.max_dev_shards > 0
            else len(dev_files)
        ),
    )

    print(
        "Input dim:",
        HS_DIM + 1,
    )

    print(
        "CLIP TRAIN mean:",
        clip_mean,
    )

    print(
        "CLIP TRAIN std:",
        clip_std,
    )

    print(
        "CLIP TRAIN range:",
        float(
            train_clip_raw.min()
        ),
        float(
            train_clip_raw.max()
        ),
    )

    print(
        "CLIP DEV range:",
        float(
            dev_clip_raw.min()
        ),
        float(
            dev_clip_raw.max()
        ),
    )

    print(
        "Seed:",
        args.seed,
    )

    print(
        "Semantic TEST accessed: NO"
    )

    (
        out_dir
        /
        "clip_normalization.json"
    ).write_text(
        json.dumps(
            {
                "clip_train_mean":
                    clip_mean,

                "clip_train_std":
                    clip_std,

                "normalization":
                    (
                        "(clip_cosine - "
                        "TRAIN_mean) / TRAIN_std"
                    ),

                "input":
                    (
                        "decoder_hs_256 + "
                        "global_clip_cosine_1"
                    ),

                "semantic_test_accessed":
                    False,
            },
            indent=2,
        )
        +
        "\n"
    )

    model = A3SemanticVerifier().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []

    best_ap = -1.0
    best_epoch = -1

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        print()
        print("=" * 100)
        print(
            f"EPOCH {epoch}/"
            f"{args.epochs}"
        )
        print("=" * 100)

        train_loss = train_one_epoch(
            model=model,
            train_files=train_files,
            train_clip_z=train_clip_z,
            optimizer=optimizer,
            device=device,
            batch_size=args.batch_size,
            epoch_seed=(
                args.seed
                +
                epoch
            ),
        )

        print()
        print(
            "Weighted train loss:",
            f"{train_loss:.6f}",
        )

        dev = predict_dev(
            model=model,
            dev_files=dev_files,
            dev_clip_z=dev_clip_z,
            device=device,
            batch_size=args.batch_size,
            max_dev_shards=(
                args.max_dev_shards
            ),
        )

        row = {
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            **dev,
        }

        history.append(
            row
        )

        print()
        print(
            "DEV proposal AUROC:",
            f"{dev['proposal_auroc']:.6f}",
        )

        print(
            "DEV proposal AP:",
            f"{dev['proposal_ap']:.6f}",
        )

        print(
            "DEV pair AUROC:",
            f"{dev['pair_auroc']:.6f}",
        )

        print(
            "DEV pair AP:",
            f"{dev['pair_ap']:.6f}",
        )

        print(
            "DEV threshold:",
            f"{dev['dev_selected_threshold']:.6f}",
        )

        print(
            "DEV present recall:",
            f"{dev['pair_present_recall']:.6f}",
        )

        print(
            "DEV absent rejection:",
            f"{dev['pair_absent_rejection']:.6f}",
        )

        print(
            "DEV balanced acc:",
            f"{dev['pair_balanced_accuracy']:.6f}",
        )

        print(
            "Clean-positive pair recovery:",
            f"{dev['clean_positive_pair_recovery']:.6f}",
        )

        print(
            "Absent mean selected:",
            f"{dev['absent_mean_selected_proposals']:.6f}",
        )

        print(
            "Native present recall:",
            f"{dev['native_present_recall']:.6f}",
        )

        print(
            "Native absent rejection:",
            f"{dev['native_absent_rejection']:.6f}",
        )

        if (
            dev[
                "proposal_ap"
            ]
            >
            best_ap
        ):

            best_ap = float(
                dev[
                    "proposal_ap"
                ]
            )

            best_epoch = epoch

            torch.save(
                {
                    "mode":
                        "A3",

                    "input":
                        (
                            "decoder_hs_256 + "
                            "global_clip_cosine_1"
                        ),

                    "seed":
                        args.seed,

                    "epoch":
                        epoch,

                    "model_state":
                        model.state_dict(),

                    "clip_train_mean":
                        clip_mean,

                    "clip_train_std":
                        clip_std,

                    "dev_metrics":
                        dev,
                },
                out_dir
                /
                "best.pt",
            )

            print(
                ">>> NEW BEST CHECKPOINT"
            )

        pd.DataFrame(
            history
        ).to_csv(
            out_dir
            /
            "history.csv",
            index=False,
        )

        (
            out_dir
            /
            "history.json"
        ).write_text(
            json.dumps(
                history,
                indent=2,
            )
            +
            "\n"
        )

    summary = {
        "mode":
            "A3",

        "input":
            (
                "decoder_hs_256 + "
                "global_clip_cosine_1"
            ),

        "seed":
            args.seed,

        "epochs":
            args.epochs,

        "best_epoch":
            best_epoch,

        "best_proposal_ap":
            best_ap,

        "checkpoint_selection_metric":
            "DEV proposal AP",

        "clip_normalization":
            "TRAIN pair-level mean/std",

        "pair_weighting":
            (
                "equal total pair weight; "
                "if both labels exist, "
                "0.5 positive / 0.5 negative"
            ),

        "semantic_test_accessed":
            False,
    }

    (
        out_dir
        /
        "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        +
        "\n"
    )

    print()
    print("=" * 100)
    print("A3 TRAINING COMPLETE")
    print("=" * 100)

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best proposal AP:",
        best_ap,
    )

    print(
        "Output:",
        out_dir,
    )

    print(
        "SEMANTIC TEST ACCESSED: NO"
    )


if __name__ == "__main__":
    main()
