#!/usr/bin/env python3

import os
import json
import glob
import math
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score


# ============================================================
# PATHS
# ============================================================

SUP_ROOT = Path(
    "generated/fsc147_instance_set_supervision"
)

FEATURE_ROOT = Path(
    "outputs/fsc147_instance_reasoning_features"
)

OUT_ROOT = Path(
    "outputs/instance_relation_encoder_v1"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOCATE DECODER EMBEDDING SHARD
# ============================================================

def locate_embedding_file(
    feature_dir,
    shard_name,
    expected_rows,
):

    candidates = [
        feature_dir / f"{shard_name}_embeddings.npy",
        feature_dir / f"{shard_name}_embedding.npy",
        feature_dir / f"{shard_name}.embeddings.npy",
        feature_dir / f"{shard_name}.embedding.npy",
        feature_dir / f"{shard_name}.npy",
    ]

    for p in candidates:

        if not p.exists():
            continue

        arr = np.load(
            p,
            mmap_mode="r",
        )

        if (
            arr.ndim == 2
            and
            arr.shape[0] == expected_rows
            and
            arr.shape[1] == 256
        ):
            return p

    wildcard = sorted(
        feature_dir.glob(
            f"{shard_name}*.npy"
        )
    )

    valid = []

    for p in wildcard:

        try:

            arr = np.load(
                p,
                mmap_mode="r",
            )

        except Exception:

            continue

        if (
            arr.ndim == 2
            and
            arr.shape[0] == expected_rows
            and
            arr.shape[1] == 256
        ):

            valid.append(p)

    if len(valid) == 1:

        return valid[0]

    raise RuntimeError(
        f"Cannot uniquely locate embedding file "
        f"for {shard_name}; "
        f"expected shape=({expected_rows},256); "
        f"candidates={valid}"
    )


# ============================================================
# BUILD / LOAD CONSOLIDATED HS
# ============================================================

def load_hs_embeddings(
    split,
    nodes,
):

    base = (
        SUP_ROOT
        /
        split
    )

    cache_path = (
        base
        /
        "decoder_hs_float32.npy"
    )

    if cache_path.exists():

        arr = np.load(
            cache_path,
            mmap_mode="r",
        )

        if arr.shape != (
            len(nodes),
            256,
        ):

            raise RuntimeError(
                f"Bad cached HS shape "
                f"{arr.shape}"
            )

        print(
            f"{split}: using cached HS "
            f"{cache_path} "
            f"{arr.shape}"
        )

        return arr

    if "source_shard" not in nodes.columns:

        raise RuntimeError(
            "nodes.csv does not contain "
            "source_shard."
        )

    feature_dir = (
        FEATURE_ROOT
        /
        split
        /
        "threshold_023"
    )

    print(
        f"{split}: consolidating decoder HS..."
    )

    out = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(nodes),
            256,
        ),
    )

    groups = list(
        nodes.groupby(
            "source_shard",
            sort=False,
        )
    )

    for idx, (
        shard_name,
        g,
    ) in enumerate(
        groups,
        start=1,
    ):

        shard_name = str(
            shard_name
        )

        emb_path = locate_embedding_file(
            feature_dir,
            shard_name,
            len(g),
        )

        emb = np.load(
            emb_path,
            mmap_mode="r",
        )

        gids = (
            g[
                "global_node_id"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        out[gids] = (
            np.asarray(
                emb,
                dtype=np.float32,
            )
        )

        if (
            idx % 10 == 0
            or
            idx == len(groups)
        ):

            print(
                f"  shards "
                f"{idx}/"
                f"{len(groups)}"
            )

    out.flush()

    del out

    arr = np.load(
        cache_path,
        mmap_mode="r",
    )

    print(
        f"{split}: HS cached "
        f"{cache_path} "
        f"{arr.shape}"
    )

    return arr


# ============================================================
# CLASS WEIGHTS
# ============================================================

def sqrt_inverse_class_weights(
    targets,
    num_classes,
):

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    valid = (
        targets >= 0
    )

    counts = np.bincount(
        targets[valid],
        minlength=num_classes,
    ).astype(
        np.float64
    )

    counts = np.maximum(
        counts,
        1.0,
    )

    weights = np.sqrt(
        counts.sum()
        /
        (
            num_classes
            *
            counts
        )
    )

    weights = (
        weights
        /
        weights.mean()
    )

    return (
        weights.astype(
            np.float32
        ),
        counts.astype(
            np.int64
        ),
    )


# ============================================================
# DATA BUNDLE
# ============================================================

class GraphBundle:

    def __init__(
        self,
        split,
        edge_mean=None,
        edge_std=None,
    ):

        self.split = split

        base = (
            SUP_ROOT
            /
            split
        )

        self.nodes = pd.read_csv(
            base
            /
            "nodes.csv"
        )

        self.contrib = pd.read_csv(
            base
            /
            "node_instance_contribution_targets.csv"
        )

        self.graph_info = pd.read_csv(
            base
            /
            "relation_graph_images.csv"
        )

        graph_npz = np.load(
            base
            /
            "relation_graph_iou_gt0.npz"
        )

        self.edge_index = (
            graph_npz[
                "edge_index"
            ].astype(
                np.int64
            )
        )

        self.edge_features_raw = (
            graph_npz[
                "edge_features"
            ].astype(
                np.float32
            )
        )

        self.edge_shared_count = (
            graph_npz[
                "edge_shared_count"
            ].astype(
                np.int32
            )
        )

        self.edge_supervised = (
            graph_npz[
                "edge_supervised"
            ].astype(
                np.int8
            )
        )

        self.edge_graph_id = (
            graph_npz[
                "edge_graph_id"
            ].astype(
                np.int32
            )
        )

        self.node_graph_id = (
            graph_npz[
                "node_graph_id"
            ].astype(
                np.int32
            )
        )

        # ----------------------------------------------------
        # Node alignment checks
        # ----------------------------------------------------

        gids = (
            self.nodes[
                "global_node_id"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        contrib_gids = (
            self.contrib[
                "global_node_id"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        expected = np.arange(
            len(
                self.nodes
            ),
            dtype=np.int64,
        )

        if not np.array_equal(
            gids,
            expected,
        ):

            raise RuntimeError(
                f"{split}: nodes are not "
                f"global-id aligned."
            )

        if not np.array_equal(
            contrib_gids,
            expected,
        ):

            raise RuntimeError(
                f"{split}: contribution rows "
                f"misaligned."
            )

        # ----------------------------------------------------
        # Decoder HS
        # ----------------------------------------------------

        self.hs = load_hs_embeddings(
            split,
            self.nodes,
        )

        # ----------------------------------------------------
        # Contribution target
        # ----------------------------------------------------

        self.contrib_target = (
            self.contrib[
                "instance_contribution_target"
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        self.contrib_supervised = (
            self.contrib[
                "contribution_supervised"
            ]
            .to_numpy(
                dtype=np.int8
            )
        )

        # ----------------------------------------------------
        # Cardinality class:
        #
        # 1   -> 0
        # 2   -> 1
        # 3   -> 2
        # 4+  -> 3
        # 0   -> -1 (uncertain)
        # ----------------------------------------------------

        point_card = (
            self.nodes[
                "point_cardinality"
            ]
            .to_numpy(
                dtype=np.int32
            )
        )

        self.card_target = np.full(
            len(
                self.nodes
            ),
            -1,
            dtype=np.int64,
        )

        self.card_target[
            point_card == 1
        ] = 0

        self.card_target[
            point_card == 2
        ] = 1

        self.card_target[
            point_card == 3
        ] = 2

        self.card_target[
            point_card >= 4
        ] = 3

        # ----------------------------------------------------
        # Edge class:
        #
        # 0 shared  -> 0
        # 1 shared  -> 1
        # >=2       -> 2
        #
        # Edges touching zero-point uncertain nodes
        # receive -1 / ignored.
        # ----------------------------------------------------

        self.edge_target = np.full(
            len(
                self.edge_shared_count
            ),
            -1,
            dtype=np.int64,
        )

        supervised = (
            self.edge_supervised
            ==
            1
        )

        self.edge_target[
            supervised
            &
            (
                self.edge_shared_count
                ==
                0
            )
        ] = 0

        self.edge_target[
            supervised
            &
            (
                self.edge_shared_count
                ==
                1
            )
        ] = 1

        self.edge_target[
            supervised
            &
            (
                self.edge_shared_count
                >=
                2
            )
        ] = 2

        # ----------------------------------------------------
        # Normalize edge geometry using TRAIN stats only
        # ----------------------------------------------------

        if edge_mean is None:

            edge_mean = (
                self.edge_features_raw
                .mean(
                    axis=0
                )
            )

        if edge_std is None:

            edge_std = (
                self.edge_features_raw
                .std(
                    axis=0
                )
            )

        edge_std = np.maximum(
            edge_std,
            1e-6,
        )

        self.edge_mean = (
            np.asarray(
                edge_mean,
                dtype=np.float32,
            )
        )

        self.edge_std = (
            np.asarray(
                edge_std,
                dtype=np.float32,
            )
        )

        self.edge_features = (
            (
                self.edge_features_raw
                -
                self.edge_mean[
                    None,
                    :
                ]
            )
            /
            self.edge_std[
                None,
                :
            ]
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Graph indexing
        # ----------------------------------------------------

        self.num_graphs = int(
            len(
                self.graph_info
            )
        )

        graph_ids = (
            self.graph_info[
                "graph_id"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        if not np.array_equal(
            graph_ids,
            np.arange(
                self.num_graphs
            ),
        ):

            raise RuntimeError(
                f"{split}: graph ids "
                f"are not contiguous."
            )

        # Node lists by graph
        self.node_order = np.argsort(
            self.node_graph_id,
            kind="stable",
        )

        node_counts = np.bincount(
            self.node_graph_id,
            minlength=self.num_graphs,
        )

        self.node_offsets = np.concatenate(
            [
                np.array(
                    [0],
                    dtype=np.int64,
                ),
                np.cumsum(
                    node_counts,
                    dtype=np.int64,
                ),
            ]
        )

        # local node index within each graph
        self.node_local_index = np.empty(
            len(
                self.nodes
            ),
            dtype=np.int32,
        )

        for graph_id in range(
            self.num_graphs
        ):

            a = (
                self.node_offsets[
                    graph_id
                ]
            )

            b = (
                self.node_offsets[
                    graph_id + 1
                ]
            )

            ids = (
                self.node_order[
                    a:b
                ]
            )

            self.node_local_index[
                ids
            ] = np.arange(
                len(ids),
                dtype=np.int32,
            )

        # Edge lists by graph
        self.edge_order = np.argsort(
            self.edge_graph_id,
            kind="stable",
        )

        edge_counts = np.bincount(
            self.edge_graph_id,
            minlength=self.num_graphs,
        )

        self.edge_offsets = np.concatenate(
            [
                np.array(
                    [0],
                    dtype=np.int64,
                ),
                np.cumsum(
                    edge_counts,
                    dtype=np.int64,
                ),
            ]
        )

        print()
        print(
            f"{split}: graphs="
            f"{self.num_graphs} "
            f"nodes={len(self.nodes)} "
            f"edges={len(self.edge_target)}"
        )


# ============================================================
# GRAPH DATASET
# ============================================================

class InstanceGraphDataset(Dataset):

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

        return len(
            self.graph_ids
        )

    def __getitem__(
        self,
        index,
    ):

        graph_id = int(
            self.graph_ids[
                index
            ]
        )

        na = int(
            self.b.node_offsets[
                graph_id
            ]
        )

        nb = int(
            self.b.node_offsets[
                graph_id + 1
            ]
        )

        node_ids = (
            self.b.node_order[
                na:nb
            ]
        )

        ea = int(
            self.b.edge_offsets[
                graph_id
            ]
        )

        eb = int(
            self.b.edge_offsets[
                graph_id + 1
            ]
        )

        edge_ids = (
            self.b.edge_order[
                ea:eb
            ]
        )

        if len(
            edge_ids
        ) > 0:

            global_edge = (
                self.b.edge_index[
                    :,
                    edge_ids
                ]
            )

            local_edge = (
                self.b.node_local_index[
                    global_edge
                ]
            )

        else:

            local_edge = np.empty(
                (
                    2,
                    0,
                ),
                dtype=np.int64,
            )

        info = (
            self.b.graph_info.iloc[
                graph_id
            ]
        )

        return {
            "graph_id":
                graph_id,

            "hs":
                np.asarray(
                    self.b.hs[
                        node_ids
                    ],
                    dtype=np.float32,
                ),

            "contrib_target":
                self.b.contrib_target[
                    node_ids
                ],

            "contrib_supervised":
                self.b.contrib_supervised[
                    node_ids
                ],

            "card_target":
                self.b.card_target[
                    node_ids
                ],

            "edge_index":
                local_edge.astype(
                    np.int64
                ),

            "edge_features":
                self.b.edge_features[
                    edge_ids
                ],

            "edge_target":
                self.b.edge_target[
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
                    info[
                        "gt_count"
                    ]
                ),

            "baseline_count":
                float(
                    info[
                        "baseline_count"
                    ]
                ),

            "file_name":
                str(
                    info[
                        "file_name"
                    ]
                ),
        }


# ============================================================
# COLLATE
# ============================================================

def collate_graphs(
    samples,
):

    hs = []
    contrib_target = []
    contrib_supervised = []
    card_target = []

    edge_index = []
    edge_features = []
    edge_target = []

    batch_index = []

    safe_targets = []
    gt_counts = []
    baseline_counts = []
    file_names = []

    node_offset = 0

    for batch_id, s in enumerate(
        samples
    ):

        n = (
            s[
                "hs"
            ].shape[0]
        )

        hs.append(
            torch.from_numpy(
                s["hs"]
            )
        )

        contrib_target.append(
            torch.from_numpy(
                s[
                    "contrib_target"
                ]
            )
        )

        contrib_supervised.append(
            torch.from_numpy(
                s[
                    "contrib_supervised"
                ]
            )
        )

        card_target.append(
            torch.from_numpy(
                s[
                    "card_target"
                ]
            )
        )

        batch_index.append(
            torch.full(
                (n,),
                batch_id,
                dtype=torch.long,
            )
        )

        e = (
            s[
                "edge_index"
            ].shape[1]
        )

        if e > 0:

            edge_index.append(
                torch.from_numpy(
                    s[
                        "edge_index"
                    ]
                ).long()
                +
                node_offset
            )

            edge_features.append(
                torch.from_numpy(
                    s[
                        "edge_features"
                    ]
                )
            )

            edge_target.append(
                torch.from_numpy(
                    s[
                        "edge_target"
                    ]
                )
            )

        safe_targets.append(
            s[
                "safe_target"
            ]
        )

        gt_counts.append(
            s[
                "gt_count"
            ]
        )

        baseline_counts.append(
            s[
                "baseline_count"
            ]
        )

        file_names.append(
            s[
                "file_name"
            ]
        )

        node_offset += n

    if edge_index:

        edge_index = torch.cat(
            edge_index,
            dim=1,
        )

        edge_features = torch.cat(
            edge_features,
            dim=0,
        )

        edge_target = torch.cat(
            edge_target,
            dim=0,
        )

    else:

        edge_index = torch.empty(
            (
                2,
                0,
            ),
            dtype=torch.long,
        )

        edge_features = torch.empty(
            (
                0,
                7,
            ),
            dtype=torch.float32,
        )

        edge_target = torch.empty(
            (
                0,
            ),
            dtype=torch.long,
        )

    return {
        "hs":
            torch.cat(
                hs,
                dim=0,
            ).float(),

        "contrib_target":
            torch.cat(
                contrib_target,
                dim=0,
            ).float(),

        "contrib_supervised":
            torch.cat(
                contrib_supervised,
                dim=0,
            ).bool(),

        "card_target":
            torch.cat(
                card_target,
                dim=0,
            ).long(),

        "edge_index":
            edge_index,

        "edge_features":
            edge_features.float(),

        "edge_target":
            edge_target.long(),

        "batch_index":
            torch.cat(
                batch_index,
                dim=0,
            ),

        "safe_target":
            torch.tensor(
                safe_targets,
                dtype=torch.float32,
            ),

        "gt_count":
            torch.tensor(
                gt_counts,
                dtype=torch.float32,
            ),

        "baseline_count":
            torch.tensor(
                baseline_counts,
                dtype=torch.float32,
            ),

        "file_names":
            file_names,
    }


# ============================================================
# MLP
# ============================================================

def make_mlp(
    in_dim,
    hidden_dim,
    out_dim,
    dropout=0.0,
):

    return nn.Sequential(
        nn.Linear(
            in_dim,
            hidden_dim,
        ),
        nn.GELU(),
        nn.Dropout(
            dropout
        ),
        nn.Linear(
            hidden_dim,
            out_dim,
        ),
    )


# ============================================================
# MESSAGE PASSING LAYER
# ============================================================

class RelationLayer(nn.Module):

    def __init__(
        self,
        hidden_dim,
        edge_dim,
        dropout,
    ):

        super().__init__()

        self.message = make_mlp(
            hidden_dim
            * 2
            +
            edge_dim,
            hidden_dim,
            hidden_dim,
            dropout,
        )

        self.update = make_mlp(
            hidden_dim
            * 2,
            hidden_dim,
            hidden_dim,
            dropout,
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        z,
        edge_z,
        edge_index,
    ):

        if (
            edge_index.shape[1]
            ==
            0
        ):

            return z

        src = edge_index[0]
        dst = edge_index[1]

        # Undirected graph:
        # same edge generates messages both ways.
        msg_src_to_dst = self.message(
            torch.cat(
                [
                    z[src],
                    z[dst],
                    edge_z,
                ],
                dim=-1,
            )
        )

        msg_dst_to_src = self.message(
            torch.cat(
                [
                    z[dst],
                    z[src],
                    edge_z,
                ],
                dim=-1,
            )
        )

        agg = torch.zeros_like(
            z
        )

        agg.index_add_(
            0,
            dst,
            msg_src_to_dst,
        )

        agg.index_add_(
            0,
            src,
            msg_dst_to_src,
        )

        degree = torch.zeros(
            (
                z.shape[0],
                1,
            ),
            dtype=z.dtype,
            device=z.device,
        )

        ones = torch.ones(
            (
                edge_index.shape[1],
                1,
            ),
            dtype=z.dtype,
            device=z.device,
        )

        degree.index_add_(
            0,
            dst,
            ones,
        )

        degree.index_add_(
            0,
            src,
            ones,
        )

        agg = (
            agg
            /
            degree.clamp_min(
                1.0
            )
        )

        delta = self.update(
            torch.cat(
                [
                    z,
                    agg,
                ],
                dim=-1,
            )
        )

        return self.norm(
            z
            +
            self.dropout(
                delta
            )
        )


# ============================================================
# MODEL
# ============================================================

class InstanceRelationEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        edge_input_dim=7,
    ):

        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.LayerNorm(
                256
            ),
            nn.Linear(
                256,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
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
        # Contribution head
        #
        # Output is positive via softplus.
        #
        # Initial prediction is exactly ~1 so the network
        # begins at the original CountGD++ counting rule.
        # ----------------------------------------------------

        self.contrib_body = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
        )

        self.contrib_out = nn.Linear(
            hidden_dim,
            1,
        )

        nn.init.zeros_(
            self.contrib_out.weight
        )

        nn.init.zeros_(
            self.contrib_out.bias
        )

        base_logit = math.log(
            math.exp(
                1.0
            )
            -
            1.0
        )

        self.register_buffer(
            "base_contribution_logit",
            torch.tensor(
                base_logit,
                dtype=torch.float32,
            ),
        )

        # 1 / 2 / 3 / 4+
        self.cardinality_head = make_mlp(
            hidden_dim,
            hidden_dim,
            4,
            dropout,
        )

        # shared count:
        # 0 / 1 / 2+
        self.edge_head = make_mlp(
            hidden_dim
            * 2
            +
            hidden_dim,
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

        contrib_hidden = (
            self.contrib_body(
                z
            )
        )

        raw_delta = (
            self.contrib_out(
                contrib_hidden
            )
            .squeeze(-1)
        )

        contribution = F.softplus(
            self.base_contribution_logit
            +
            raw_delta
        )

        cardinality_logits = (
            self.cardinality_head(
                z
            )
        )

        if (
            edge_index.shape[1]
            >
            0
        ):

            src = edge_index[0]
            dst = edge_index[1]

            # Symmetric pair representation.
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
                (
                    0,
                    3,
                ),
                dtype=z.dtype,
                device=z.device,
            )

        return {
            "contribution":
                contribution,

            "cardinality_logits":
                cardinality_logits,

            "edge_logits":
                edge_logits,

            "z":
                z,
        }


# ============================================================
# GRAPH SUM
# ============================================================

def graph_sum(
    values,
    batch_index,
    num_graphs,
):

    out = torch.zeros(
        num_graphs,
        dtype=values.dtype,
        device=values.device,
    )

    out.index_add_(
        0,
        batch_index,
        values,
    )

    return out


# ============================================================
# DEVICE MOVE
# ============================================================

def move_batch(
    batch,
    device,
):

    out = {}

    for k, v in (
        batch.items()
    ):

        if torch.is_tensor(v):

            out[k] = v.to(
                device,
                non_blocking=True,
            )

        else:

            out[k] = v

    return out


# ============================================================
# LOSS
# ============================================================

def compute_loss(
    output,
    batch,
    card_weights,
    edge_weights,
    lambda_contrib,
    lambda_uncertain,
    lambda_card,
    lambda_edge,
):

    contribution = (
        output[
            "contribution"
        ]
    )

    num_graphs = (
        batch[
            "safe_target"
        ].shape[0]
    )

    pred_count = graph_sum(
        contribution,
        batch[
            "batch_index"
        ],
        num_graphs,
    )

    # --------------------------------------------------------
    # MAIN:
    # differentiable set-union count approximation.
    # --------------------------------------------------------

    loss_union = F.smooth_l1_loss(
        pred_count,
        batch[
            "safe_target"
        ],
        beta=1.0,
    )

    # --------------------------------------------------------
    # Canonical local decomposition auxiliary.
    # --------------------------------------------------------

    contrib_mask = (
        batch[
            "contrib_supervised"
        ]
    )

    if contrib_mask.any():

        loss_contrib = F.smooth_l1_loss(
            contribution[
                contrib_mask
            ],
            batch[
                "contrib_target"
            ][
                contrib_mask
            ],
            beta=0.25,
        )

    else:

        loss_contrib = (
            contribution.sum()
            *
            0.0
        )

    # --------------------------------------------------------
    # Safe prior:
    #
    # Zero-point queries are NOT background labels.
    # The scoped structural target preserves them at
    # baseline contribution 1.
    # --------------------------------------------------------

    uncertain_mask = (
        ~
        batch[
            "contrib_supervised"
        ]
    )

    if uncertain_mask.any():

        loss_uncertain = F.smooth_l1_loss(
            contribution[
                uncertain_mask
            ],
            torch.ones_like(
                contribution[
                    uncertain_mask
                ]
            ),
            beta=0.25,
        )

    else:

        loss_uncertain = (
            contribution.sum()
            *
            0.0
        )

    # --------------------------------------------------------
    # Cardinality auxiliary
    # --------------------------------------------------------

    card_mask = (
        batch[
            "card_target"
        ]
        >=
        0
    )

    if card_mask.any():

        loss_card = F.cross_entropy(
            output[
                "cardinality_logits"
            ][
                card_mask
            ],
            batch[
                "card_target"
            ][
                card_mask
            ],
            weight=card_weights,
        )

    else:

        loss_card = (
            contribution.sum()
            *
            0.0
        )

    # --------------------------------------------------------
    # Shared-instance auxiliary
    # 0 / 1 / 2+
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
            weight=edge_weights,
        )

    else:

        loss_edge = (
            contribution.sum()
            *
            0.0
        )

    total = (
        loss_union
        +
        lambda_contrib
        *
        loss_contrib
        +
        lambda_uncertain
        *
        loss_uncertain
        +
        lambda_card
        *
        loss_card
        +
        lambda_edge
        *
        loss_edge
    )

    return {
        "total":
            total,

        "union":
            loss_union,

        "contrib":
            loss_contrib,

        "uncertain":
            loss_uncertain,

        "card":
            loss_card,

        "edge":
            loss_edge,

        "pred_count":
            pred_count,
    }


# ============================================================
# TRAIN EPOCH
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    device,
    card_weights,
    edge_weights,
    args,
):

    model.train()

    totals = {
        "total": 0.0,
        "union": 0.0,
        "contrib": 0.0,
        "uncertain": 0.0,
        "card": 0.0,
        "edge": 0.0,
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
            batch[
                "hs"
            ],
            batch[
                "edge_features"
            ],
            batch[
                "edge_index"
            ],
        )

        losses = compute_loss(
            output,
            batch,
            card_weights,
            edge_weights,
            args.lambda_contrib,
            args.lambda_uncertain,
            args.lambda_card,
            args.lambda_edge,
        )

        losses[
            "total"
        ].backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        for k in totals:

            totals[k] += float(
                losses[k].detach().cpu()
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

    pred_counts = []
    safe_targets = []
    gt_counts = []
    baseline_counts = []

    contrib_pred = []
    contrib_true = []

    card_pred = []
    card_true = []

    edge_pred = []
    edge_true = []

    file_names = []

    for batch in loader:

        batch = move_batch(
            batch,
            device,
        )

        output = model(
            batch[
                "hs"
            ],
            batch[
                "edge_features"
            ],
            batch[
                "edge_index"
            ],
        )

        num_graphs = (
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
            num_graphs,
        )

        pred_counts.append(
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

        baseline_counts.append(
            batch[
                "baseline_count"
            ].cpu().numpy()
        )

        file_names.extend(
            batch[
                "file_names"
            ]
        )

        # ----------------------------------------------------
        # contribution
        # ----------------------------------------------------

        mask = (
            batch[
                "contrib_supervised"
            ]
        )

        if mask.any():

            contrib_pred.append(
                output[
                    "contribution"
                ][
                    mask
                ]
                .cpu()
                .numpy()
            )

            contrib_true.append(
                batch[
                    "contrib_target"
                ][
                    mask
                ]
                .cpu()
                .numpy()
            )

        # ----------------------------------------------------
        # cardinality
        # ----------------------------------------------------

        cmask = (
            batch[
                "card_target"
            ]
            >=
            0
        )

        if cmask.any():

            cp = (
                output[
                    "cardinality_logits"
                ][
                    cmask
                ]
                .argmax(
                    dim=-1
                )
            )

            card_pred.append(
                cp.cpu().numpy()
            )

            card_true.append(
                batch[
                    "card_target"
                ][
                    cmask
                ]
                .cpu()
                .numpy()
            )

        # ----------------------------------------------------
        # edge relation
        # ----------------------------------------------------

        emask = (
            batch[
                "edge_target"
            ]
            >=
            0
        )

        if (
            emask.any()
            and
            output[
                "edge_logits"
            ].shape[0]
            >
            0
        ):

            ep = (
                output[
                    "edge_logits"
                ][
                    emask
                ]
                .argmax(
                    dim=-1
                )
            )

            edge_pred.append(
                ep.cpu().numpy()
            )

            edge_true.append(
                batch[
                    "edge_target"
                ][
                    emask
                ]
                .cpu()
                .numpy()
            )

    pred = np.concatenate(
        pred_counts
    )

    safe = np.concatenate(
        safe_targets
    )

    gt = np.concatenate(
        gt_counts
    )

    baseline = np.concatenate(
        baseline_counts
    )

    # --------------------------------------------------------
    # Count metrics
    # --------------------------------------------------------

    safe_err = (
        pred
        -
        safe
    )

    gt_err = (
        pred
        -
        gt
    )

    baseline_safe_err = (
        baseline
        -
        safe
    )

    baseline_gt_err = (
        baseline
        -
        gt
    )

    oracle_gt_err = (
        safe
        -
        gt
    )

    pred_round = np.rint(
        pred
    )

    metrics = {
        "safe_mae":
            float(
                np.abs(
                    safe_err
                ).mean()
            ),

        "safe_rmse":
            float(
                np.sqrt(
                    np.mean(
                        safe_err ** 2
                    )
                )
            ),

        "gt_mae":
            float(
                np.abs(
                    gt_err
                ).mean()
            ),

        "gt_rmse":
            float(
                np.sqrt(
                    np.mean(
                        gt_err ** 2
                    )
                )
            ),

        "rounded_gt_mae":
            float(
                np.abs(
                    pred_round
                    -
                    gt
                ).mean()
            ),

        "rounded_exact":
            int(
                (
                    pred_round
                    ==
                    gt
                ).sum()
            ),

        "rounded_within1":
            int(
                (
                    np.abs(
                        pred_round
                        -
                        gt
                    )
                    <=
                    1
                ).sum()
            ),

        "baseline_safe_mae":
            float(
                np.abs(
                    baseline_safe_err
                ).mean()
            ),

        "baseline_gt_mae":
            float(
                np.abs(
                    baseline_gt_err
                ).mean()
            ),

        "safe_oracle_gt_mae":
            float(
                np.abs(
                    oracle_gt_err
                ).mean()
            ),

        "mean_signed_gt_error":
            float(
                gt_err.mean()
            ),
    }

    denominator = (
        metrics[
            "baseline_gt_mae"
        ]
        -
        metrics[
            "safe_oracle_gt_mae"
        ]
    )

    if abs(
        denominator
    ) > 1e-8:

        metrics[
            "oracle_headroom_recovery"
        ] = float(
            (
                metrics[
                    "baseline_gt_mae"
                ]
                -
                metrics[
                    "gt_mae"
                ]
            )
            /
            denominator
        )

    else:

        metrics[
            "oracle_headroom_recovery"
        ] = float(
            "nan"
        )

    # --------------------------------------------------------
    # Contribution metric
    # --------------------------------------------------------

    if contrib_pred:

        cp = np.concatenate(
            contrib_pred
        )

        ct = np.concatenate(
            contrib_true
        )

        metrics[
            "contribution_mae"
        ] = float(
            np.abs(
                cp
                -
                ct
            ).mean()
        )

    else:

        metrics[
            "contribution_mae"
        ] = float(
            "nan"
        )

    # --------------------------------------------------------
    # Auxiliary macro-F1
    # --------------------------------------------------------

    if card_pred:

        metrics[
            "cardinality_macro_f1"
        ] = float(
            f1_score(
                np.concatenate(
                    card_true
                ),
                np.concatenate(
                    card_pred
                ),
                average="macro",
                zero_division=0,
            )
        )

    else:

        metrics[
            "cardinality_macro_f1"
        ] = float(
            "nan"
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

    else:

        metrics[
            "edge_macro_f1"
        ] = float(
            "nan"
        )

    prediction_df = pd.DataFrame(
        {
            "file_name":
                file_names,

            "prediction":
                pred,

            "prediction_rounded":
                pred_round,

            "safe_structural_target":
                safe,

            "gt_count":
                gt,

            "baseline_count":
                baseline,
        }
    )

    return (
        metrics,
        prediction_df,
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
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--lambda-contrib",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--lambda-uncertain",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--lambda-card",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--lambda-edge",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
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

    args = parser.parse_args()

    seed_everything(
        args.seed
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 110)
    print(
        "INSTANCE RELATION ENCODER V1"
    )
    print("=" * 110)

    print(args)

    # ========================================================
    # DATA
    # ========================================================

    train_bundle = GraphBundle(
        "train"
    )

    val_bundle = GraphBundle(
        "val",
        edge_mean=
            train_bundle.edge_mean,
        edge_std=
            train_bundle.edge_std,
    )

    train_ids = np.arange(
        train_bundle.num_graphs,
        dtype=np.int64,
    )

    val_ids = np.arange(
        val_bundle.num_graphs,
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

    train_dataset = InstanceGraphDataset(
        train_bundle,
        train_ids,
    )

    val_dataset = InstanceGraphDataset(
        val_bundle,
        val_ids,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=
            args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=
            collate_graphs,
        pin_memory=
            torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=
            args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=
            collate_graphs,
        pin_memory=
            torch.cuda.is_available(),
    )

    # ========================================================
    # AUXILIARY CLASS WEIGHTS
    # ========================================================

    card_weights_np, card_counts = (
        sqrt_inverse_class_weights(
            train_bundle.card_target,
            4,
        )
    )

    edge_weights_np, edge_counts = (
        sqrt_inverse_class_weights(
            train_bundle.edge_target,
            3,
        )
    )

    print()
    print(
        "Cardinality counts:",
        card_counts
    )

    print(
        "Cardinality weights:",
        card_weights_np
    )

    print(
        "Edge counts [0,1,2+]:",
        edge_counts
    )

    print(
        "Edge weights:",
        edge_weights_np
    )

    # ========================================================
    # MODEL
    # ========================================================

    device = torch.device(
        "cuda"
        if
        torch.cuda.is_available()
        else
        "cpu"
    )

    print()
    print(
        "Device:",
        device
    )

    model = InstanceRelationEncoder(
        hidden_dim=
            args.hidden_dim,
        num_layers=
            args.num_layers,
        dropout=
            args.dropout,
        edge_input_dim=
            train_bundle.edge_features.shape[1],
    ).to(
        device
    )

    card_weights = torch.tensor(
        card_weights_np,
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
        weight_decay=
            args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )
    )

    history = []

    best_safe_mae = float(
        "inf"
    )

    best_epoch = -1

    patience_count = 0

    best_path = (
        OUT_ROOT
        /
        "best.pt"
    )

    # ========================================================
    # BASELINE BEFORE TRAINING
    # ========================================================

    print()
    print("=" * 110)
    print(
        "INITIAL / BASELINE-LIKE MODEL"
    )
    print("=" * 110)

    initial_metrics, _ = evaluate(
        model,
        val_loader,
        device,
    )

    for k, v in (
        initial_metrics.items()
    ):

        print(
            f"{k:30s}: {v}"
        )

    # --------------------------------------------------------
    # Epoch 0 / native CountGD++ counting rule is a valid
    # baseline candidate. A trained checkpoint is considered
    # better only if it improves SAFE structural VAL MAE.
    # --------------------------------------------------------

    best_safe_mae = float(
        initial_metrics[
            "safe_mae"
        ]
    )

    best_epoch = 0

    patience_count = 0

    torch.save(
        {
            "epoch":
                0,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "args":
                vars(args),

            "edge_mean":
                train_bundle.edge_mean,

            "edge_std":
                train_bundle.edge_std,

            "card_weights":
                card_weights_np,

            "edge_weights":
                edge_weights_np,

            "val_metrics":
                initial_metrics,
        },
        best_path,
    )

    print(
        f"Epoch-0 baseline registered as best: "
        f"safe MAE={best_safe_mae:.4f}"
    )

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
            card_weights,
            edge_weights,
            args,
        )

        val_metrics, prediction_df = (
            evaluate(
                model,
                val_loader,
                device,
            )
        )

        scheduler.step(
            val_metrics[
                "safe_mae"
            ]
        )

        current_lr = (
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        row = {
            "epoch":
                epoch,

            "lr":
                current_lr,

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
                in val_metrics.items()
            },
        }

        history.append(
            row
        )

        print()
        print(
            f"[Epoch {epoch:03d}] "
            f"loss="
            f"{train_losses['total']:.4f} "
            f"union="
            f"{train_losses['union']:.4f} "
            f"| "
            f"VAL safe MAE="
            f"{val_metrics['safe_mae']:.4f} "
            f"GT MAE="
            f"{val_metrics['gt_mae']:.4f} "
            f"recovery="
            f"{val_metrics['oracle_headroom_recovery']:.3f} "
            f"cardF1="
            f"{val_metrics['cardinality_macro_f1']:.3f} "
            f"edgeF1="
            f"{val_metrics['edge_macro_f1']:.3f}"
        )

        if (
            val_metrics[
                "safe_mae"
            ]
            <
            best_safe_mae
        ):

            best_safe_mae = (
                val_metrics[
                    "safe_mae"
                ]
            )

            best_epoch = epoch

            patience_count = 0

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "args":
                        vars(args),

                    "edge_mean":
                        train_bundle.edge_mean,

                    "edge_std":
                        train_bundle.edge_std,

                    "card_weights":
                        card_weights_np,

                    "edge_weights":
                        edge_weights_np,

                    "val_metrics":
                        val_metrics,
                },
                best_path,
            )

            prediction_df.to_csv(
                OUT_ROOT
                /
                "best_val_predictions.csv",
                index=False,
            )

            print(
                f"  -> NEW BEST "
                f"(safe MAE "
                f"{best_safe_mae:.4f})"
            )

        else:

            patience_count += 1

        pd.DataFrame(
            history
        ).to_csv(
            OUT_ROOT
            /
            "history.csv",
            index=False,
        )

        if (
            patience_count
            >=
            args.patience
        ):

            print()
            print(
                f"Early stopping: "
                f"no improvement for "
                f"{args.patience} epochs."
            )

            break

    # ========================================================
    # FINAL BEST EVALUATION
    # ========================================================

    checkpoint = torch.load(
        best_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    best_metrics, best_predictions = (
        evaluate(
            model,
            val_loader,
            device,
        )
    )

    best_predictions.to_csv(
        OUT_ROOT
        /
        "final_best_val_predictions.csv",
        index=False,
    )

    with open(
        OUT_ROOT
        /
        "final_metrics.json",
        "w",
    ) as f:

        json.dump(
            {
                "best_epoch":
                    best_epoch,

                "selection_metric":
                    "VAL safe structural MAE",

                "metrics":
                    best_metrics,

                "test_accessed":
                    False,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 110)
    print(
        "BEST CLASS-DISJOINT VAL RESULT"
    )
    print("=" * 110)

    print(
        "Best epoch:",
        best_epoch
    )

    for k, v in (
        best_metrics.items()
    ):

        print(
            f"{k:30s}: {v}"
        )

    print()
    print(
        "Saved:",
        best_path
    )

    print(
        OUT_ROOT
        /
        "history.csv"
    )

    print(
        OUT_ROOT
        /
        "final_metrics.json"
    )

    print(
        OUT_ROOT
        /
        "final_best_val_predictions.csv"
    )

    print()
    print(
        "Model selection used "
        "SAFE STRUCTURAL VAL MAE, "
        "not TEST."
    )

    print(
        "TEST split was NOT accessed."
    )


if __name__ == "__main__":
    main()
