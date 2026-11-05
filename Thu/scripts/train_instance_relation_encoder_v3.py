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

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_instance_relation_encoder_v1 import (
    seed_everything,
    GraphBundle,
    RelationLayer,
    make_mlp,
    move_batch,
    graph_sum,
    sqrt_inverse_class_weights,
)

OUT_ROOT = Path(
    "outputs/instance_relation_encoder_v3"
)


# ============================================================
# TARGETS
# ============================================================

def attach_targets(bundle):

    r = bundle.contrib_target.copy()

    supervised = (
        bundle.contrib_supervised == 1
    )

    action = np.full(
        len(r),
        -1,
        dtype=np.int64,
    )

    # 0 REDUCE
    # 1 KEEP
    # 2 EXPAND

    action[
        supervised & (r < 1.0 - 1e-6)
    ] = 0

    action[
        supervised &
        (np.abs(r - 1.0) <= 1e-6)
    ] = 1

    action[
        supervised & (r > 1.0 + 1e-6)
    ] = 2

    reduce_target = np.zeros(
        len(r),
        dtype=np.float32,
    )

    expand_target = np.zeros(
        len(r),
        dtype=np.float32,
    )

    reduce_target[
        supervised
    ] = np.maximum(
        1.0 - r[supervised],
        0.0,
    )

    expand_target[
        supervised
    ] = np.maximum(
        r[supervised] - 1.0,
        0.0,
    )

    bundle.action_target = action
    bundle.reduce_target = reduce_target
    bundle.expand_target = expand_target


# ============================================================
# DATASET
# ============================================================

class GraphDatasetV3(Dataset):

    def __init__(
        self,
        bundle,
        graph_ids=None,
    ):

        self.b = bundle

        if graph_ids is None:
            graph_ids = np.arange(
                bundle.num_graphs,
                dtype=np.int64,
            )

        self.graph_ids = np.asarray(
            graph_ids,
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.graph_ids)

    def __getitem__(self, idx):

        b = self.b
        graph_id = int(
            self.graph_ids[idx]
        )

        na = int(
            b.node_offsets[graph_id]
        )
        nb = int(
            b.node_offsets[graph_id + 1]
        )

        node_ids = b.node_order[na:nb]

        ea = int(
            b.edge_offsets[graph_id]
        )
        eb = int(
            b.edge_offsets[graph_id + 1]
        )

        edge_ids = b.edge_order[ea:eb]

        if len(edge_ids) > 0:

            global_edge = (
                b.edge_index[:, edge_ids]
            )

            local_edge = (
                b.node_local_index[
                    global_edge
                ]
            )

        else:

            local_edge = np.empty(
                (2, 0),
                dtype=np.int64,
            )

        info = b.graph_info.iloc[
            graph_id
        ]

        return {
            "hs":
                np.asarray(
                    b.hs[node_ids],
                    dtype=np.float32,
                ),

            "contrib_target":
                b.contrib_target[
                    node_ids
                ],

            "contrib_supervised":
                b.contrib_supervised[
                    node_ids
                ],

            "action_target":
                b.action_target[
                    node_ids
                ],

            "reduce_target":
                b.reduce_target[
                    node_ids
                ],

            "expand_target":
                b.expand_target[
                    node_ids
                ],

            "edge_index":
                local_edge.astype(
                    np.int64
                ),

            "edge_features":
                b.edge_features[
                    edge_ids
                ],

            "edge_target":
                b.edge_target[
                    edge_ids
                ],

            "safe_target":
                float(
                    info[
                        "structural_target_keep_uncertain"
                    ]
                ),

            "gt_count":
                float(
                    info["gt_count"]
                ),

            "baseline_count":
                float(
                    info["baseline_count"]
                ),

            "file_name":
                str(
                    info["file_name"]
                ),
        }


def collate_v3(samples):

    keys_node = [
        "hs",
        "contrib_target",
        "contrib_supervised",
        "action_target",
        "reduce_target",
        "expand_target",
    ]

    store = {
        k: []
        for k in keys_node
    }

    edge_index = []
    edge_features = []
    edge_target = []

    batch_index = []

    safe = []
    gt = []
    baseline = []
    names = []

    offset = 0

    for batch_id, s in enumerate(
        samples
    ):

        n = s["hs"].shape[0]

        store["hs"].append(
            torch.from_numpy(
                s["hs"]
            )
        )

        for k in [
            "contrib_target",
            "contrib_supervised",
            "action_target",
            "reduce_target",
            "expand_target",
        ]:
            store[k].append(
                torch.from_numpy(
                    s[k]
                )
            )

        batch_index.append(
            torch.full(
                (n,),
                batch_id,
                dtype=torch.long,
            )
        )

        if s["edge_index"].shape[1] > 0:

            edge_index.append(
                torch.from_numpy(
                    s["edge_index"]
                ).long()
                +
                offset
            )

            edge_features.append(
                torch.from_numpy(
                    s["edge_features"]
                )
            )

            edge_target.append(
                torch.from_numpy(
                    s["edge_target"]
                )
            )

        safe.append(
            s["safe_target"]
        )

        gt.append(
            s["gt_count"]
        )

        baseline.append(
            s["baseline_count"]
        )

        names.append(
            s["file_name"]
        )

        offset += n

    if edge_index:

        edge_index = torch.cat(
            edge_index,
            dim=1,
        )

        edge_features = torch.cat(
            edge_features,
            dim=0,
        ).float()

        edge_target = torch.cat(
            edge_target,
            dim=0,
        ).long()

    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

        edge_features = torch.empty(
            (0, 7),
            dtype=torch.float32,
        )

        edge_target = torch.empty(
            (0,),
            dtype=torch.long,
        )

    return {
        "hs":
            torch.cat(
                store["hs"],
                dim=0,
            ).float(),

        "contrib_target":
            torch.cat(
                store["contrib_target"],
                dim=0,
            ).float(),

        "contrib_supervised":
            torch.cat(
                store["contrib_supervised"],
                dim=0,
            ).bool(),

        "action_target":
            torch.cat(
                store["action_target"],
                dim=0,
            ).long(),

        "reduce_target":
            torch.cat(
                store["reduce_target"],
                dim=0,
            ).float(),

        "expand_target":
            torch.cat(
                store["expand_target"],
                dim=0,
            ).float(),

        "edge_index":
            edge_index,

        "edge_features":
            edge_features,

        "edge_target":
            edge_target,

        "batch_index":
            torch.cat(
                batch_index,
                dim=0,
            ),

        "safe_target":
            torch.tensor(
                safe,
                dtype=torch.float32,
            ),

        "gt_count":
            torch.tensor(
                gt,
                dtype=torch.float32,
            ),

        "baseline_count":
            torch.tensor(
                baseline,
                dtype=torch.float32,
            ),

        "file_names":
            names,
    }


# ============================================================
# MODEL
# ============================================================

class InstanceRelationEncoderV3(nn.Module):

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

        self.action_head = make_mlp(
            hidden_dim,
            hidden_dim,
            3,
            dropout,
        )

        # Neutral action initialization.
        #
        # p(REDUCE) = p(KEEP) = p(EXPAND) = 1/3.
        #
        # Together with equal initial reduce/expand
        # magnitudes, contribution starts at exactly ~1,
        # while gradients for REDUCE/EXPAND remain alive.
        final_action = (
            self.action_head[-1]
        )

        nn.init.zeros_(
            final_action.weight
        )

        nn.init.zeros_(
            final_action.bias
        )

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

        # Small initial magnitudes.
        nn.init.zeros_(
            self.reduce_head[-1].weight
        )
        nn.init.zeros_(
            self.expand_head[-1].weight
        )

        # Reduction uses sigmoid, expansion uses softplus.
        # Initialize both magnitudes to exactly 0.1.
        reduce_init = math.log(
            0.1 / 0.9
        )

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

        action_logits = (
            self.action_head(
                z
            )
        )

        action_prob = F.softmax(
            action_logits,
            dim=-1,
        )

        # reduction is physically bounded [0,1]
        reduce_mag = torch.sigmoid(
            self.reduce_head(
                z
            ).squeeze(-1)
        )

        # expansion is non-negative and unbounded
        expand_mag = F.softplus(
            self.expand_head(
                z
            ).squeeze(-1)
        )

        p_reduce = action_prob[:, 0]
        p_expand = action_prob[:, 2]

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

        contribution = contribution.clamp_min(
            0.0
        )

        if edge_index.shape[1] > 0:

            src = edge_index[0]
            dst = edge_index[1]

            pair = torch.cat(
                [
                    torch.abs(
                        z[src] - z[dst]
                    ),
                    z[src] * z[dst],
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
                device=z.device,
                dtype=z.dtype,
            )

        return {
            "z":
                z,

            "action_logits":
                action_logits,

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
# LOSS
# ============================================================

def compute_loss(
    output,
    batch,
    action_weights,
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

    # Main set-union count loss.
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

    # --------------------------------------------------------
    # Action direction
    # --------------------------------------------------------

    loss_action = F.cross_entropy(
        output[
            "action_logits"
        ][
            supervised
        ],
        batch[
            "action_target"
        ][
            supervised
        ],
        weight=
            action_weights,
    )

    # --------------------------------------------------------
    # REDUCE magnitude
    # --------------------------------------------------------

    reduce_mask = (
        batch[
            "action_target"
        ]
        ==
        0
    )

    if reduce_mask.any():

        loss_reduce = (
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

        loss_reduce = (
            pred_count.sum() * 0
        )

    # --------------------------------------------------------
    # EXPAND magnitude
    #
    # log1p stabilizes the heavy tail.
    # Image union loss still supervises absolute count.
    # --------------------------------------------------------

    expand_mask = (
        batch[
            "action_target"
        ]
        ==
        2
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

        loss_expand = (
            F.smooth_l1_loss(
                pred_expand_log,
                true_expand_log,
                beta=0.20,
            )
        )

    else:

        loss_expand = (
            pred_count.sum() * 0
        )

    # --------------------------------------------------------
    # Canonical contribution auxiliary
    # --------------------------------------------------------

    loss_contrib = F.smooth_l1_loss(
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

    # --------------------------------------------------------
    # Zero-point nodes:
    # preserve baseline, not background.
    # --------------------------------------------------------

    uncertain = ~supervised

    if uncertain.any():

        loss_uncertain = F.smooth_l1_loss(
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

    else:

        loss_uncertain = (
            pred_count.sum() * 0
        )

    # --------------------------------------------------------
    # Edge auxiliary
    # --------------------------------------------------------

    edge_mask = (
        batch["edge_target"] >= 0
    )

    if (
        edge_mask.any()
        and
        output[
            "edge_logits"
        ].shape[0] > 0
    ):

        loss_edge = F.cross_entropy(
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

    else:

        loss_edge = (
            pred_count.sum() * 0
        )

    total = (
        loss_union
        +
        args.lambda_action
        *
        loss_action
        +
        args.lambda_reduce
        *
        loss_reduce
        +
        args.lambda_expand
        *
        loss_expand
        +
        args.lambda_contrib
        *
        loss_contrib
        +
        args.lambda_uncertain
        *
        loss_uncertain
        +
        args.lambda_edge
        *
        loss_edge
    )

    return {
        "total": total,
        "union": loss_union,
        "action": loss_action,
        "reduce": loss_reduce,
        "expand": loss_expand,
        "contrib": loss_contrib,
        "uncertain": loss_uncertain,
        "edge": loss_edge,
    }


# ============================================================
# TRAIN
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    device,
    action_weights,
    edge_weights,
    args,
):

    model.train()

    totals = {
        k: 0.0
        for k in [
            "total",
            "union",
            "action",
            "reduce",
            "expand",
            "contrib",
            "uncertain",
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

        out = model(
            batch["hs"],
            batch["edge_features"],
            batch["edge_index"],
        )

        losses = compute_loss(
            out,
            batch,
            action_weights,
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
# EVAL
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

    true_actions = []
    pred_actions = []

    reduce_true = []
    reduce_pred = []

    expand_true = []
    expand_pred = []

    edge_true = []
    edge_pred = []

    names = []

    for batch in loader:

        batch = move_batch(
            batch,
            device,
        )

        out = model(
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
            out[
                "contribution"
            ],
            batch[
                "batch_index"
            ],
            n_graphs,
        )

        predictions.append(
            pred_count.cpu().numpy()
        )

        safe_targets.append(
            batch[
                "safe_target"
            ].cpu().numpy()
        )

        gt_counts.append(
            batch[
                "gt_count"
            ].cpu().numpy()
        )

        baselines.append(
            batch[
                "baseline_count"
            ].cpu().numpy()
        )

        names.extend(
            batch["file_names"]
        )

        supervised = (
            batch[
                "contrib_supervised"
            ]
        )

        if supervised.any():

            pa = (
                out[
                    "action_logits"
                ][
                    supervised
                ]
                .argmax(-1)
            )

            pred_actions.append(
                pa.cpu().numpy()
            )

            true_actions.append(
                batch[
                    "action_target"
                ][
                    supervised
                ]
                .cpu()
                .numpy()
            )

        rm = (
            batch[
                "action_target"
            ]
            ==
            0
        )

        if rm.any():

            reduce_pred.append(
                out[
                    "reduce_mag"
                ][rm]
                .cpu()
                .numpy()
            )

            reduce_true.append(
                batch[
                    "reduce_target"
                ][rm]
                .cpu()
                .numpy()
            )

        em = (
            batch[
                "action_target"
            ]
            ==
            2
        )

        if em.any():

            expand_pred.append(
                out[
                    "expand_mag"
                ][em]
                .cpu()
                .numpy()
            )

            expand_true.append(
                batch[
                    "expand_target"
                ][em]
                .cpu()
                .numpy()
            )

        edge_mask = (
            batch[
                "edge_target"
            ] >= 0
        )

        if (
            edge_mask.any()
            and
            out[
                "edge_logits"
            ].shape[0] > 0
        ):

            edge_pred.append(
                out[
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

    pred_delta = pred - baseline
    true_delta = safe - baseline

    baseline_gt_mae = float(
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
        baseline_gt_mae
        -
        oracle_mae
    )

    nonzero = (
        true_delta != 0
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
                        (pred - gt) ** 2
                    )
                )
            ),

        "baseline_gt_mae":
            baseline_gt_mae,

        "safe_oracle_gt_mae":
            oracle_mae,

        "oracle_headroom_recovery":
            float(
                (
                    baseline_gt_mae
                    -
                    gt_mae
                )
                /
                denominator
            )
            if denominator > 1e-8
            else float("nan"),

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
                        pred_delta[nonzero]
                    )
                    ==
                    np.sign(
                        true_delta[nonzero]
                    )
                ).mean()
            )
            if nonzero.any()
            else float("nan"),
    }

    if pred_actions:

        true_action_arr = np.concatenate(
            true_actions
        )

        pred_action_arr = np.concatenate(
            pred_actions
        )

        metrics[
            "action_macro_f1"
        ] = float(
            f1_score(
                true_action_arr,
                pred_action_arr,
                average="macro",
                zero_division=0,
            )
        )

        for cls, name in [
            (0, "REDUCE"),
            (1, "KEEP"),
            (2, "EXPAND"),
        ]:

            metrics[
                f"pred_action_{name.lower()}_pct"
            ] = float(
                100.0
                *
                (
                    pred_action_arr
                    ==
                    cls
                ).mean()
            )

            metrics[
                f"true_action_{name.lower()}_pct"
            ] = float(
                100.0
                *
                (
                    true_action_arr
                    ==
                    cls
                ).mean()
            )

    if reduce_pred:

        metrics[
            "reduce_magnitude_mae"
        ] = float(
            np.abs(
                np.concatenate(
                    reduce_pred
                )
                -
                np.concatenate(
                    reduce_true
                )
            ).mean()
        )

    if expand_pred:

        metrics[
            "expand_magnitude_mae"
        ] = float(
            np.abs(
                np.concatenate(
                    expand_pred
                )
                -
                np.concatenate(
                    expand_true
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

    return metrics, pred_df


# ============================================================
# MAIN
# ============================================================

def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    p.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )

    p.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    p.add_argument(
        "--lambda-action",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--lambda-reduce",
        type=float,
        default=0.25,
    )

    p.add_argument(
        "--lambda-expand",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--lambda-contrib",
        type=float,
        default=0.2,
    )

    p.add_argument(
        "--lambda-uncertain",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--lambda-edge",
        type=float,
        default=0.2,
    )

    p.add_argument(
        "--max-train-graphs",
        type=int,
        default=None,
    )

    p.add_argument(
        "--max-val-graphs",
        type=int,
        default=None,
    )

    args = p.parse_args()

    seed_everything(
        args.seed
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        train.num_graphs
    )

    val_ids = np.arange(
        val.num_graphs
    )

    if args.max_train_graphs:
        train_ids = train_ids[
            :args.max_train_graphs
        ]

    if args.max_val_graphs:
        val_ids = val_ids[
            :args.max_val_graphs
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

    action_weights_np, action_counts = (
        sqrt_inverse_class_weights(
            train.action_target,
            3,
        )
    )

    edge_weights_np, _ = (
        sqrt_inverse_class_weights(
            train.edge_target,
            3,
        )
    )

    print("=" * 110)
    print(
        "INSTANCE RELATION ENCODER V3"
    )
    print(
        "SEPARATE REDUCE / EXPAND RESOLUTION"
    )
    print("=" * 110)

    print(
        "Action counts "
        "[REDUCE,KEEP,EXPAND]:",
        action_counts
    )

    print(
        "Action weights:",
        action_weights_np
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = InstanceRelationEncoderV3(
        hidden_dim=
            args.hidden_dim,
        num_layers=
            args.num_layers,
    ).to(device)

    action_weights = torch.tensor(
        action_weights_np,
        dtype=torch.float32,
        device=device,
    )

    edge_weights = torch.tensor(
        edge_weights_np,
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # epoch 0
    initial, initial_pred = evaluate(
        model,
        val_loader,
        device,
    )

    print()
    print("=" * 110)
    print("INITIAL")
    print("=" * 110)

    for k, v in initial.items():
        print(
            f"{k:32s}: {v}"
        )

    best_mae = initial[
        "gt_mae"
    ]

    best_epoch = 0

    best_path = (
        OUT_ROOT /
        "best_count.pt"
    )

    torch.save(
        {
            "epoch": 0,
            "model_state_dict":
                model.state_dict(),
            "metrics":
                initial,
            "args":
                vars(args),
        },
        best_path,
    )

    history = []
    patience = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            action_weights,
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
                    f"train_{k}": v
                    for k, v
                    in train_loss.items()
                },

                **{
                    f"val_{k}": v
                    for k, v
                    in metrics.items()
                },
            }
        )

        print()
        print(
            f"[Epoch {epoch:03d}] "
            f"loss={train_loss['total']:.4f} "
            f"GT_MAE={metrics['gt_mae']:.4f} "
            f"recovery={metrics['oracle_headroom_recovery']:.3f} "
            f"predΔ={metrics['mean_pred_delta']:.3f} "
            f"sign={metrics['nonzero_delta_sign_accuracy']:.3f} "
            f"actionF1={metrics['action_macro_f1']:.3f} "
            f"expandMAE={metrics['expand_magnitude_mae']:.3f}"
        )

        if metrics["gt_mae"] < best_mae:

            best_mae = metrics[
                "gt_mae"
            ]

            best_epoch = epoch
            patience = 0

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "metrics":
                        metrics,

                    "args":
                        vars(args),
                },
                best_path,
            )

            predictions.to_csv(
                OUT_ROOT /
                "best_val_predictions.csv",
                index=False,
            )

            print(
                "  -> NEW BEST COUNT"
            )

        else:
            patience += 1

        pd.DataFrame(
            history
        ).to_csv(
            OUT_ROOT /
            "history.csv",
            index=False,
        )

        if patience >= args.patience:

            print(
                "Early stopping."
            )
            break

    checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    final, pred = evaluate(
        model,
        val_loader,
        device,
    )

    pred.to_csv(
        OUT_ROOT /
        "final_val_predictions.csv",
        index=False,
    )

    with open(
        OUT_ROOT /
        "final_metrics.json",
        "w",
    ) as f:

        json.dump(
            {
                "best_epoch":
                    best_epoch,
                "metrics":
                    final,
                "test_accessed":
                    False,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 110)
    print("FINAL V3")
    print("=" * 110)

    print(
        "Best epoch:",
        best_epoch
    )

    for k, v in final.items():
        print(
            f"{k:32s}: {v}"
        )

    print()
    print(
        "TEST split was NOT accessed."
    )


if __name__ == "__main__":
    main()
