#!/usr/bin/env python3

import sys
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_instance_relation_encoder_v3 import (
    GraphBundle,
    attach_targets,
    GraphDatasetV3,
    collate_v3,
    RelationLayer,
    make_mlp,
    move_batch,
    graph_sum,
    seed_everything,
)


OUT_ROOT = Path(
    "outputs/instance_relation_encoder_v4"
)


# ============================================================
# MODEL
# ============================================================

class InstanceRelationEncoderV4(nn.Module):

    def __init__(
        self,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        edge_input_dim=7,
    ):

        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(
                256,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(
                edge_input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        self.layers = nn.ModuleList(
            [
                RelationLayer(
                    hidden_dim,
                    hidden_dim,
                    dropout,
                )
                for _ in range(
                    num_layers
                )
            ]
        )

        # ----------------------------------------------------
        # INDEPENDENT BINARY GATES
        # ----------------------------------------------------

        self.reduce_gate = make_mlp(
            hidden_dim,
            hidden_dim,
            1,
            dropout,
        )

        self.expand_gate = make_mlp(
            hidden_dim,
            hidden_dim,
            1,
            dropout,
        )

        # Neutral logits -> p=0.5.
        #
        # Since initial reduce and expand magnitudes
        # are equal, their effects cancel:
        #
        # r = 1 - .5*.1 + .5*.1 = 1
        #
        # So epoch 0 remains native CountGD++.
        for head in [
            self.reduce_gate,
            self.expand_gate,
        ]:

            nn.init.zeros_(
                head[-1].weight
            )

            nn.init.zeros_(
                head[-1].bias
            )

        # ----------------------------------------------------
        # MAGNITUDES
        # ----------------------------------------------------

        self.reduce_head = make_mlp(
            hidden_dim,
            hidden_dim,
            1,
            dropout,
        )

        self.expand_head = make_mlp(
            hidden_dim,
            hidden_dim,
            1,
            dropout,
        )

        nn.init.zeros_(
            self.reduce_head[-1].weight
        )

        nn.init.zeros_(
            self.expand_head[-1].weight
        )

        # sigmoid(logit(.1)) = .1
        reduce_init = math.log(
            0.1 / 0.9
        )

        # softplus(x) = .1
        expand_init = math.log(
            math.exp(0.1) - 1.0
        )

        nn.init.constant_(
            self.reduce_head[-1].bias,
            reduce_init,
        )

        nn.init.constant_(
            self.expand_head[-1].bias,
            expand_init,
        )

        # ----------------------------------------------------
        # EDGE AUXILIARY
        # ----------------------------------------------------

        self.edge_head = make_mlp(
            hidden_dim * 3,
            hidden_dim,
            3,
            dropout,
        )

    def forward(
        self,
        hs,
        edge_features,
        edge_index,
    ):

        z = self.node_encoder(
            hs
        )

        edge_z = self.edge_encoder(
            edge_features
        )

        for layer in self.layers:

            z = layer(
                z,
                edge_z,
                edge_index,
            )

        reduce_logit = (
            self.reduce_gate(z)
            .squeeze(-1)
        )

        expand_logit = (
            self.expand_gate(z)
            .squeeze(-1)
        )

        p_reduce = torch.sigmoid(
            reduce_logit
        )

        p_expand = torch.sigmoid(
            expand_logit
        )

        # Reduction cannot remove >1 baseline unit.
        reduce_mag = torch.sigmoid(
            self.reduce_head(z)
            .squeeze(-1)
        )

        # Expansion can be arbitrarily large.
        expand_mag = F.softplus(
            self.expand_head(z)
            .squeeze(-1)
        )

        contribution = (
            1.0
            -
            p_reduce
            *
            reduce_mag
            +
            p_expand
            *
            expand_mag
        )

        contribution = (
            contribution.clamp_min(
                0.0
            )
        )

        if edge_index.shape[1] > 0:

            src = edge_index[0]
            dst = edge_index[1]

            pair = torch.cat(
                [
                    torch.abs(
                        z[src]
                        -
                        z[dst]
                    ),
                    z[src]
                    *
                    z[dst],
                    edge_z,
                ],
                dim=-1,
            )

            edge_logits = (
                self.edge_head(
                    pair
                )
            )

        else:

            edge_logits = torch.empty(
                (0, 3),
                dtype=z.dtype,
                device=z.device,
            )

        return {
            "z":
                z,

            "reduce_logit":
                reduce_logit,

            "expand_logit":
                expand_logit,

            "p_reduce":
                p_reduce,

            "p_expand":
                p_expand,

            "reduce_mag":
                reduce_mag,

            "expand_mag":
                expand_mag,

            "contribution":
                contribution,

            "edge_logits":
                edge_logits,
        }


# ============================================================
# GATE WEIGHTS
# ============================================================

def binary_pos_weight(
    action_target,
    positive_class,
):

    x = np.asarray(
        action_target,
        dtype=np.int64,
    )

    valid = (
        x >= 0
    )

    positive = int(
        (
            x[valid]
            ==
            positive_class
        ).sum()
    )

    negative = int(
        valid.sum()
        -
        positive
    )

    # sqrt imbalance:
    #
    # enough to counter severe imbalance,
    # but less aggressive than full N/P weighting.
    weight = math.sqrt(
        negative
        /
        max(
            positive,
            1,
        )
    )

    return (
        positive,
        negative,
        float(weight),
    )


# ============================================================
# LOSS
# ============================================================

def compute_loss(
    output,
    batch,
    reduce_pos_weight,
    expand_pos_weight,
    edge_weights,
    args,
):

    n_graphs = (
        batch[
            "safe_target"
        ].shape[0]
    )

    pred_count = graph_sum(
        output[
            "contribution"
        ],
        batch[
            "batch_index"
        ],
        n_graphs,
    )

    # --------------------------------------------------------
    # MAIN SET-UNION COUNT
    # --------------------------------------------------------

    loss_union = F.smooth_l1_loss(
        pred_count,
        batch[
            "safe_target"
        ],
        beta=1.0,
    )

    supervised = (
        batch[
            "contrib_supervised"
        ]
    )

    action = (
        batch[
            "action_target"
        ]
    )

    # --------------------------------------------------------
    # Independent REDUCE gate
    # --------------------------------------------------------

    reduce_target = (
        action[
            supervised
        ]
        ==
        0
    ).float()

    loss_reduce_gate = (
        F.binary_cross_entropy_with_logits(
            output[
                "reduce_logit"
            ][
                supervised
            ],
            reduce_target,
            pos_weight=
                reduce_pos_weight,
        )
    )

    # --------------------------------------------------------
    # Independent EXPAND gate
    # --------------------------------------------------------

    expand_target = (
        action[
            supervised
        ]
        ==
        2
    ).float()

    loss_expand_gate = (
        F.binary_cross_entropy_with_logits(
            output[
                "expand_logit"
            ][
                supervised
            ],
            expand_target,
            pos_weight=
                expand_pos_weight,
        )
    )

    # --------------------------------------------------------
    # REDUCE magnitude
    # --------------------------------------------------------

    reduce_mask = (
        action == 0
    )

    if reduce_mask.any():

        loss_reduce_mag = (
            F.smooth_l1_loss(
                output[
                    "reduce_mag"
                ][
                    reduce_mask
                ],
                batch[
                    "reduce_target"
                ][
                    reduce_mask
                ],
                beta=0.10,
            )
        )

    else:

        loss_reduce_mag = (
            pred_count.sum()
            *
            0.0
        )

    # --------------------------------------------------------
    # EXPAND magnitude
    #
    # Log-domain auxiliary for long tail.
    # --------------------------------------------------------

    expand_mask = (
        action == 2
    )

    if expand_mask.any():

        pred_expand_log = torch.log1p(
            output[
                "expand_mag"
            ][
                expand_mask
            ]
        )

        true_expand_log = torch.log1p(
            batch[
                "expand_target"
            ][
                expand_mask
            ]
        )

        loss_expand_mag = (
            F.smooth_l1_loss(
                pred_expand_log,
                true_expand_log,
                beta=0.20,
            )
        )

    else:

        loss_expand_mag = (
            pred_count.sum()
            *
            0.0
        )

    # --------------------------------------------------------
    # Local canonical contribution auxiliary
    # --------------------------------------------------------

    loss_contrib = (
        F.smooth_l1_loss(
            output[
                "contribution"
            ][
                supervised
            ],
            batch[
                "contrib_target"
            ][
                supervised
            ],
            beta=0.25,
        )
    )

    # --------------------------------------------------------
    # Zero-point queries remain baseline-preserved.
    # --------------------------------------------------------

    uncertain = ~supervised

    if uncertain.any():

        loss_uncertain = (
            F.smooth_l1_loss(
                output[
                    "contribution"
                ][
                    uncertain
                ],
                torch.ones_like(
                    output[
                        "contribution"
                    ][
                        uncertain
                    ]
                ),
                beta=0.25,
            )
        )

    else:

        loss_uncertain = (
            pred_count.sum()
            *
            0.0
        )

    # --------------------------------------------------------
    # Do not encourage strong REDUCE and EXPAND
    # evidence simultaneously.
    # --------------------------------------------------------

    loss_exclusive = (
        (
            output[
                "p_reduce"
            ][
                supervised
            ]
            *
            output[
                "p_expand"
            ][
                supervised
            ]
        )
        .mean()
    )

    # --------------------------------------------------------
    # Edge auxiliary
    # --------------------------------------------------------

    edge_mask = (
        batch[
            "edge_target"
        ]
        >=
        0
    )

    if (
        edge_mask.any()
        and
        output[
            "edge_logits"
        ].shape[0]
        >
        0
    ):

        loss_edge = (
            F.cross_entropy(
                output[
                    "edge_logits"
                ][
                    edge_mask
                ],
                batch[
                    "edge_target"
                ][
                    edge_mask
                ],
                weight=
                    edge_weights,
            )
        )

    else:

        loss_edge = (
            pred_count.sum()
            *
            0.0
        )

    total = (
        loss_union
        +
        args.lambda_reduce_gate
        *
        loss_reduce_gate
        +
        args.lambda_expand_gate
        *
        loss_expand_gate
        +
        args.lambda_reduce_mag
        *
        loss_reduce_mag
        +
        args.lambda_expand_mag
        *
        loss_expand_mag
        +
        args.lambda_contrib
        *
        loss_contrib
        +
        args.lambda_uncertain
        *
        loss_uncertain
        +
        args.lambda_exclusive
        *
        loss_exclusive
        +
        args.lambda_edge
        *
        loss_edge
    )

    return {
        "total":
            total,

        "union":
            loss_union,

        "reduce_gate":
            loss_reduce_gate,

        "expand_gate":
            loss_expand_gate,

        "reduce_mag":
            loss_reduce_mag,

        "expand_mag":
            loss_expand_mag,

        "contrib":
            loss_contrib,

        "uncertain":
            loss_uncertain,

        "exclusive":
            loss_exclusive,

        "edge":
            loss_edge,
    }


# ============================================================
# TRAIN
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    device,
    reduce_pos_weight,
    expand_pos_weight,
    edge_weights,
    args,
):

    model.train()

    totals = {
        k: 0.0
        for k in [
            "total",
            "union",
            "reduce_gate",
            "expand_gate",
            "reduce_mag",
            "expand_mag",
            "contrib",
            "uncertain",
            "exclusive",
            "edge",
        ]
    }

    steps = 0

    for batch in loader:

        batch = move_batch(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            batch["hs"],
            batch["edge_features"],
            batch["edge_index"],
        )

        losses = compute_loss(
            output,
            batch,
            reduce_pos_weight,
            expand_pos_weight,
            edge_weights,
            args,
        )

        losses[
            "total"
        ].backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        optimizer.step()

        for k in totals:

            totals[k] += float(
                losses[k]
                .detach()
                .cpu()
            )

        steps += 1

    for k in totals:

        totals[k] /= max(
            steps,
            1,
        )

    return totals


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    predictions = []
    safe_targets = []
    gt_counts = []
    baselines = []

    reduce_prob = []
    reduce_true = []

    expand_prob = []
    expand_true = []

    reduce_mag_pred = []
    reduce_mag_true = []

    expand_mag_pred = []
    expand_mag_true = []

    edge_pred = []
    edge_true = []

    names = []

    for batch in loader:

        batch = move_batch(
            batch,
            device,
        )

        output = model(
            batch["hs"],
            batch["edge_features"],
            batch["edge_index"],
        )

        n_graphs = (
            batch[
                "safe_target"
            ].shape[0]
        )

        pred_count = graph_sum(
            output[
                "contribution"
            ],
            batch[
                "batch_index"
            ],
            n_graphs,
        )

        predictions.append(
            pred_count
            .cpu()
            .numpy()
        )

        safe_targets.append(
            batch[
                "safe_target"
            ]
            .cpu()
            .numpy()
        )

        gt_counts.append(
            batch[
                "gt_count"
            ]
            .cpu()
            .numpy()
        )

        baselines.append(
            batch[
                "baseline_count"
            ]
            .cpu()
            .numpy()
        )

        names.extend(
            batch[
                "file_names"
            ]
        )

        supervised = (
            batch[
                "contrib_supervised"
            ]
        )

        action = (
            batch[
                "action_target"
            ]
        )

        if supervised.any():

            reduce_prob.append(
                output[
                    "p_reduce"
                ][
                    supervised
                ]
                .cpu()
                .numpy()
            )

            reduce_true.append(
                (
                    action[
                        supervised
                    ]
                    ==
                    0
                )
                .cpu()
                .numpy()
                .astype(
                    np.int64
                )
            )

            expand_prob.append(
                output[
                    "p_expand"
                ][
                    supervised
                ]
                .cpu()
                .numpy()
            )

            expand_true.append(
                (
                    action[
                        supervised
                    ]
                    ==
                    2
                )
                .cpu()
                .numpy()
                .astype(
                    np.int64
                )
            )

        rm = (
            action == 0
        )

        if rm.any():

            reduce_mag_pred.append(
                output[
                    "reduce_mag"
                ][rm]
                .cpu()
                .numpy()
            )

            reduce_mag_true.append(
                batch[
                    "reduce_target"
                ][rm]
                .cpu()
                .numpy()
            )

        em = (
            action == 2
        )

        if em.any():

            expand_mag_pred.append(
                output[
                    "expand_mag"
                ][em]
                .cpu()
                .numpy()
            )

            expand_mag_true.append(
                batch[
                    "expand_target"
                ][em]
                .cpu()
                .numpy()
            )

        edge_mask = (
            batch[
                "edge_target"
            ]
            >=
            0
        )

        if (
            edge_mask.any()
            and
            output[
                "edge_logits"
            ].shape[0]
            >
            0
        ):

            edge_pred.append(
                output[
                    "edge_logits"
                ][
                    edge_mask
                ]
                .argmax(-1)
                .cpu()
                .numpy()
            )

            edge_true.append(
                batch[
                    "edge_target"
                ][
                    edge_mask
                ]
                .cpu()
                .numpy()
            )

    pred = np.concatenate(
        predictions
    )

    safe = np.concatenate(
        safe_targets
    )

    gt = np.concatenate(
        gt_counts
    )

    baseline = np.concatenate(
        baselines
    )

    pred_delta = (
        pred - baseline
    )

    true_delta = (
        safe - baseline
    )

    baseline_mae = float(
        np.abs(
            baseline - gt
        ).mean()
    )

    oracle_mae = float(
        np.abs(
            safe - gt
        ).mean()
    )

    gt_mae = float(
        np.abs(
            pred - gt
        ).mean()
    )

    denominator = (
        baseline_mae
        -
        oracle_mae
    )

    nonzero = (
        true_delta != 0
    )

    rp = np.concatenate(
        reduce_prob
    )

    rt = np.concatenate(
        reduce_true
    )

    ep = np.concatenate(
        expand_prob
    )

    et = np.concatenate(
        expand_true
    )

    reduce_binary = (
        rp > 0.5
    ).astype(
        np.int64
    )

    expand_binary = (
        ep > 0.5
    ).astype(
        np.int64
    )

    metrics = {
        "safe_mae":
            float(
                np.abs(
                    pred - safe
                ).mean()
            ),

        "gt_mae":
            gt_mae,

        "gt_rmse":
            float(
                np.sqrt(
                    np.mean(
                        (
                            pred
                            -
                            gt
                        )
                        **
                        2
                    )
                )
            ),

        "baseline_gt_mae":
            baseline_mae,

        "safe_oracle_gt_mae":
            oracle_mae,

        "oracle_headroom_recovery":
            float(
                (
                    baseline_mae
                    -
                    gt_mae
                )
                /
                denominator
            )
            if
            denominator > 1e-8
            else
            float("nan"),

        "mean_true_delta":
            float(
                true_delta.mean()
            ),

        "mean_pred_delta":
            float(
                pred_delta.mean()
            ),

        "delta_mae":
            float(
                np.abs(
                    pred_delta
                    -
                    true_delta
                ).mean()
            ),

        "nonzero_delta_sign_accuracy":
            float(
                (
                    np.sign(
                        pred_delta[
                            nonzero
                        ]
                    )
                    ==
                    np.sign(
                        true_delta[
                            nonzero
                        ]
                    )
                ).mean()
            )
            if nonzero.any()
            else
            float("nan"),

        "reduce_gate_f1":
            float(
                f1_score(
                    rt,
                    reduce_binary,
                    zero_division=0,
                )
            ),

        "expand_gate_f1":
            float(
                f1_score(
                    et,
                    expand_binary,
                    zero_division=0,
                )
            ),

        "true_reduce_pct":
            float(
                100.0
                *
                rt.mean()
            ),

        "pred_reduce_pct":
            float(
                100.0
                *
                reduce_binary.mean()
            ),

        "true_expand_pct":
            float(
                100.0
                *
                et.mean()
            ),

        "pred_expand_pct":
            float(
                100.0
                *
                expand_binary.mean()
            ),

        "mean_p_reduce":
            float(
                rp.mean()
            ),

        "mean_p_expand":
            float(
                ep.mean()
            ),
    }

    if reduce_mag_pred:

        metrics[
            "reduce_magnitude_mae"
        ] = float(
            np.abs(
                np.concatenate(
                    reduce_mag_pred
                )
                -
                np.concatenate(
                    reduce_mag_true
                )
            ).mean()
        )

    if expand_mag_pred:

        metrics[
            "expand_magnitude_mae"
        ] = float(
            np.abs(
                np.concatenate(
                    expand_mag_pred
                )
                -
                np.concatenate(
                    expand_mag_true
                )
            ).mean()
        )

    if edge_pred:

        metrics[
            "edge_macro_f1"
        ] = float(
            f1_score(
                np.concatenate(
                    edge_true
                ),
                np.concatenate(
                    edge_pred
                ),
                average="macro",
                zero_division=0,
            )
        )

    pred_df = pd.DataFrame(
        {
            "file_name":
                names,

            "prediction":
                pred,

            "safe_structural_target":
                safe,

            "gt_count":
                gt,

            "baseline_count":
                baseline,

            "predicted_delta":
                pred_delta,

            "true_structural_delta":
                true_delta,
        }
    )

    return (
        metrics,
        pred_df,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    metrics,
    args,
):

    torch.save(
        {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "metrics":
                metrics,

            "args":
                vars(args),
        },
        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--lambda-reduce-gate",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--lambda-expand-gate",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--lambda-reduce-mag",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--lambda-expand-mag",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--lambda-contrib",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--lambda-uncertain",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--lambda-exclusive",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--lambda-edge",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--max-train-graphs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-val-graphs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/instance_relation_encoder_v4",
    )

    args = parser.parse_args()

    seed_everything(
        args.seed
    )

    global OUT_ROOT

    OUT_ROOT = Path(
        args.output_dir
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # DATA
    # ========================================================

    train = GraphBundle(
        "train"
    )

    val = GraphBundle(
        "val",
        edge_mean=
            train.edge_mean,
        edge_std=
            train.edge_std,
    )

    attach_targets(train)
    attach_targets(val)

    train_ids = np.arange(
        train.num_graphs,
        dtype=np.int64,
    )

    val_ids = np.arange(
        val.num_graphs,
        dtype=np.int64,
    )

    if args.max_train_graphs is not None:

        train_ids = train_ids[
            :
            args.max_train_graphs
        ]

    if args.max_val_graphs is not None:

        val_ids = val_ids[
            :
            args.max_val_graphs
        ]

    train_loader = DataLoader(
        GraphDatasetV3(
            train,
            train_ids,
        ),
        batch_size=
            args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=
            collate_v3,
    )

    val_loader = DataLoader(
        GraphDatasetV3(
            val,
            val_ids,
        ),
        batch_size=
            args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=
            collate_v3,
    )

    # ========================================================
    # TRAIN-DERIVED BINARY WEIGHTS
    # ========================================================

    (
        reduce_pos,
        reduce_neg,
        reduce_weight,
    ) = binary_pos_weight(
        train.action_target,
        0,
    )

    (
        expand_pos,
        expand_neg,
        expand_weight,
    ) = binary_pos_weight(
        train.action_target,
        2,
    )

    print("=" * 110)
    print(
        "INSTANCE RELATION ENCODER V4"
    )
    print(
        "INDEPENDENT REDUCE / EXPAND GATES"
    )
    print("=" * 110)

    print(
        "REDUCE:",
        f"pos={reduce_pos}",
        f"neg={reduce_neg}",
        f"sqrt-pos-weight={reduce_weight:.4f}",
    )

    print(
        "EXPAND:",
        f"pos={expand_pos}",
        f"neg={expand_neg}",
        f"sqrt-pos-weight={expand_weight:.4f}",
    )

    # Edge class weights unchanged from previous protocol.
    edge_valid = (
        train.edge_target
        >=
        0
    )

    edge_counts = np.bincount(
        train.edge_target[
            edge_valid
        ],
        minlength=3,
    ).astype(
        np.float64
    )

    edge_w = np.sqrt(
        edge_counts.sum()
        /
        (
            3.0
            *
            np.maximum(
                edge_counts,
                1.0,
            )
        )
    )

    edge_w = (
        edge_w
        /
        edge_w.mean()
    )

    device = torch.device(
        "cuda"
        if
        torch.cuda.is_available()
        else
        "cpu"
    )

    print(
        "Device:",
        device
    )

    model = InstanceRelationEncoderV4(
        hidden_dim=
            args.hidden_dim,

        num_layers=
            args.num_layers,

        edge_input_dim=
            train.edge_features.shape[1],
    ).to(
        device
    )

    reduce_pos_weight = torch.tensor(
        reduce_weight,
        dtype=torch.float32,
        device=device,
    )

    expand_pos_weight = torch.tensor(
        expand_weight,
        dtype=torch.float32,
        device=device,
    )

    edge_weights = torch.tensor(
        edge_w,
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # ========================================================
    # INITIAL
    # ========================================================

    initial_metrics, initial_pred = (
        evaluate(
            model,
            val_loader,
            device,
        )
    )

    print()
    print("=" * 110)
    print("INITIAL")
    print("=" * 110)

    for k, v in initial_metrics.items():

        print(
            f"{k:32s}: {v}"
        )

    best_count = (
        initial_metrics[
            "gt_mae"
        ]
    )

    best_structural = (
        initial_metrics[
            "safe_mae"
        ]
    )

    best_count_epoch = 0
    best_structural_epoch = 0

    best_count_path = (
        OUT_ROOT /
        "best_count.pt"
    )

    best_structural_path = (
        OUT_ROOT /
        "best_structural.pt"
    )

    save_checkpoint(
        best_count_path,
        0,
        model,
        optimizer,
        initial_metrics,
        args,
    )

    save_checkpoint(
        best_structural_path,
        0,
        model,
        optimizer,
        initial_metrics,
        args,
    )

    initial_pred.to_csv(
        OUT_ROOT /
        "epoch0_val_predictions.csv",
        index=False,
    )

    history = []
    patience_count = 0

    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            reduce_pos_weight,
            expand_pos_weight,
            edge_weights,
            args,
        )

        metrics, predictions = (
            evaluate(
                model,
                val_loader,
                device,
            )
        )

        history.append(
            {
                "epoch":
                    epoch,

                **{
                    f"train_{k}":
                        v
                    for k, v
                    in train_losses.items()
                },

                **{
                    f"val_{k}":
                        v
                    for k, v
                    in metrics.items()
                },
            }
        )

        print()
        print(
            f"[Epoch {epoch:03d}] "
            f"loss={train_losses['total']:.4f} "
            f"GT={metrics['gt_mae']:.4f} "
            f"safe={metrics['safe_mae']:.4f} "
            f"rec={metrics['oracle_headroom_recovery']:.3f} "
            f"predΔ={metrics['mean_pred_delta']:.3f} "
            f"sign={metrics['nonzero_delta_sign_accuracy']:.3f} "
            f"R-F1={metrics['reduce_gate_f1']:.3f} "
            f"E-F1={metrics['expand_gate_f1']:.3f} "
            f"E%={metrics['pred_expand_pct']:.2f}"
        )

        improved = False

        if (
            metrics[
                "gt_mae"
            ]
            <
            best_count
        ):

            best_count = (
                metrics[
                    "gt_mae"
                ]
            )

            best_count_epoch = epoch

            save_checkpoint(
                best_count_path,
                epoch,
                model,
                optimizer,
                metrics,
                args,
            )

            predictions.to_csv(
                OUT_ROOT /
                "best_count_val_predictions.csv",
                index=False,
            )

            print(
                "  -> NEW BEST COUNT"
            )

            improved = True

        if (
            metrics[
                "safe_mae"
            ]
            <
            best_structural
        ):

            best_structural = (
                metrics[
                    "safe_mae"
                ]
            )

            best_structural_epoch = epoch

            save_checkpoint(
                best_structural_path,
                epoch,
                model,
                optimizer,
                metrics,
                args,
            )

            predictions.to_csv(
                OUT_ROOT /
                "best_structural_val_predictions.csv",
                index=False,
            )

            print(
                "  -> NEW BEST STRUCTURAL"
            )

            improved = True

        if improved:

            patience_count = 0

        else:

            patience_count += 1

        pd.DataFrame(
            history
        ).to_csv(
            OUT_ROOT /
            "history.csv",
            index=False,
        )

        if (
            patience_count
            >=
            args.patience
        ):

            print(
                "Early stopping."
            )
            break

    # ========================================================
    # FINAL BOTH CHECKPOINTS
    # ========================================================

    final = {}

    for name, path in [
        (
            "best_count",
            best_count_path,
        ),
        (
            "best_structural",
            best_structural_path,
        ),
    ]:

        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        metrics, predictions = (
            evaluate(
                model,
                val_loader,
                device,
            )
        )

        final[name] = {
            "epoch":
                int(
                    checkpoint[
                        "epoch"
                    ]
                ),

            "metrics":
                metrics,
        }

        predictions.to_csv(
            OUT_ROOT /
            f"{name}_final_predictions.csv",
            index=False,
        )

    with open(
        OUT_ROOT /
        "final_metrics.json",
        "w",
    ) as f:

        json.dump(
            {
                "best_count_epoch":
                    best_count_epoch,

                "best_structural_epoch":
                    best_structural_epoch,

                "results":
                    final,

                "test_accessed":
                    False,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 110)
    print("FINAL V4")
    print("=" * 110)

    for name, result in final.items():

        print()
        print(
            name.upper(),
            "epoch=",
            result["epoch"],
        )

        for k, v in (
            result[
                "metrics"
            ].items()
        ):

            print(
                f"{k:32s}: {v}"
            )

    print()
    print(
        "TEST split was NOT accessed."
    )


if __name__ == "__main__":
    main()
