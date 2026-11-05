from pathlib import Path
import sys, hashlib, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

R = Path.cwd()
sys.path.insert(0, str(R / "scripts"))

from train_instance_relation_encoder_v3 import (
    GraphBundle, attach_targets, GraphDatasetV3,
    collate_v3, move_batch, graph_sum,
)
from train_instance_relation_encoder_v4 import InstanceRelationEncoderV4

O = R / "outputs/joint_semantic_counting_v1"
U = pd.read_csv(
    O / "p14_06_unified_matched_v1/unified_predictions.csv"
)
A3 = O / "p14_07b_a3_fullraw_fsc_unified_v1"
OUT = O / "p14_08a_frozen_v4_node_contrib_unified_v1"

assert len(U) == 1301 and U.pair_id.is_unique
assert not OUT.exists(), f"Output exists: {OUT}"

print("=" * 72)
print("P14-08A — FROZEN V4 + A3 NODE ALIGNMENT")
print("=" * 72)

m = json.loads(
    (A3 / "a3_unified_1301_metrics.json").read_text()
)
a3pair = pd.read_csv(
    A3 / "a3_unified_1301_predictions.csv"
)

assert len(a3pair) == 1301
assert a3pair.pair_id.tolist() == U.pair_id.tolist()

print(
    "\n[1] FROZEN A3 WHAT RESULTS:",
    {k: m[k] for k in (
        "tp", "fn", "tn", "fp",
        "positive_recall", "absent_rejection",
        "balanced_accuracy"
    )},
    flush=True,
)

ckpath = R / "outputs/instance_relation_encoder_v4/best_count.pt"

h = hashlib.sha256()
with ckpath.open("rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)

assert h.hexdigest() == (
    "310343ac678e6cc421749b4165c4a3253"
    "f047dd109795605dbd27b8212d98507"
)
print("\n[2] Frozen V4 SHA256: PASS", flush=True)

def set_roots(sup, feat):
    g0 = GraphBundle.__init__.__globals__
    fun = g0.get("load_hs_embeddings")

    globals_list = [
        g0,
        attach_targets.__globals__,
    ]
    if callable(fun):
        globals_list.append(fun.__globals__)

    touched = 0
    for g in {id(x): x for x in globals_list}.values():
        if "SUP_ROOT" in g:
            g["SUP_ROOT"] = sup
            touched += 1
        if "FEATURE_ROOT" in g:
            g["FEATURE_ROOT"] = feat

    assert touched > 0

standard_sup = R / "generated/fsc147_instance_set_supervision"
standard_feat = R / "outputs/fsc147_instance_reasoning_features"
absent_sup = R / "generated/fsc147_instance_set_supervision_absent111"
absent_feat = R / "outputs/fsc147_instance_reasoning_features_absent111"

print("\n[3] Loading original TRAIN normalization...", flush=True)

set_roots(standard_sup, standard_feat)
train = GraphBundle("train")

print("Loading FSC TEST positive graphs...", flush=True)

pos = GraphBundle(
    "test",
    edge_mean=train.edge_mean,
    edge_std=train.edge_std,
)
attach_targets(pos)

print("Loading FSC absent111 graphs...", flush=True)

set_roots(absent_sup, absent_feat)
neg = GraphBundle(
    "val",
    edge_mean=train.edge_mean,
    edge_std=train.edge_std,
)
attach_targets(neg)

assert (pos.num_graphs, neg.num_graphs) == (1190, 111)
assert (len(pos.hs), len(neg.hs)) == (81492, 6940)
assert pos.edge_features.shape[1] == neg.edge_features.shape[1] == 7

def graph_meta(sup, split, label):
    root = sup / split

    graphs = pd.read_csv(
        root / "relation_graph_images.csv"
    ).sort_values("graph_id").reset_index(drop=True)

    nodes = pd.read_csv(
        root / "nodes.csv"
    ).sort_values("global_node_id").reset_index(drop=True)

    assert np.array_equal(
        graphs.graph_id.to_numpy(),
        np.arange(len(graphs)),
    )

    assert np.array_equal(
        nodes.global_node_id.to_numpy(),
        np.arange(len(nodes)),
    )

    assert np.array_equal(
        np.bincount(
            nodes.dataset_index.to_numpy(int),
            minlength=len(graphs),
        ),
        graphs.num_nodes.to_numpy(int),
    )

    matches = graphs[["file_name", "query"]].merge(
        U.loc[
            U.presence_label == label,
            ["pair_id", "file_name", "query"]
        ],
        on=["file_name", "query"],
        how="left",
        validate="one_to_one",
        sort=False,
    )

    assert len(matches) == len(graphs)
    assert matches.pair_id.notna().all()

    return nodes, matches.pair_id.to_numpy()

posnodes, posids = graph_meta(standard_sup, "test", 1)
negnodes, negids = graph_meta(absent_sup, "val", 0)

assert np.array_equal(
    np.concatenate([posids, negids]),
    U.pair_id.to_numpy(),
)

print("Positive + absent graph/pair order: PASS", flush=True)

print("\n[4] Loading frozen V4 model...", flush=True)

ck = torch.load(
    ckpath,
    map_location="cpu",
    weights_only=False,
)

assert int(ck["epoch"]) == 10

opts = ck.get("args", {})

model = InstanceRelationEncoderV4(
    hidden_dim=int(opts.get("hidden_dim", 128)),
    num_layers=int(opts.get("num_layers", 2)),
    edge_input_dim=7,
)

model.load_state_dict(
    ck["model_state_dict"],
    strict=True,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model = model.to(device).eval()

print("Device:", device, flush=True)

def extract(bundle, label):
    loader = DataLoader(
        GraphDatasetV3(
            bundle,
            np.arange(bundle.num_graphs, dtype=np.int64),
        ),
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_v3,
    )

    rr = []
    pred = []

    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            batch = move_batch(batch, device)

            out = model(
                batch["hs"],
                batch["edge_features"],
                batch["edge_index"],
            )

            contrib = out["contribution"].reshape(-1)

            assert len(contrib) == len(batch["batch_index"])

            ng = int(batch["gt_count"].numel())

            counts = graph_sum(
                contrib,
                batch["batch_index"],
                ng,
            )

            rr.append(
                contrib.detach().cpu().numpy().astype(np.float64)
            )
            pred.append(
                counts.detach().cpu().numpy().astype(np.float64)
            )

            if step % 30 == 0:
                print(
                    f"[{label}] batches={step}",
                    flush=True,
                )

    r = np.concatenate(rr)
    p = np.concatenate(pred)

    assert len(r) == len(bundle.hs)
    assert len(p) == bundle.num_graphs
    assert np.isfinite(r).all() and (r >= 0).all()

    return r, p

print("\n[5] Frozen V4 node inference...", flush=True)

rp, pp = extract(pos, "positive")
rn, pn = extract(neg, "absent")

r = np.concatenate([rp, rn])
pred = np.concatenate([pp, pn])

ref = U["V4 seed42"].to_numpy(float)

maxdiff = float(np.max(np.abs(pred - ref)))

print(
    "V4 count parity max abs difference:",
    maxdiff,
    flush=True,
)

assert maxdiff < 1e-3, "STOP: V4 mismatch with P14-06"

node_gid = np.concatenate([
    posnodes.dataset_index.to_numpy(int),
    negnodes.dataset_index.to_numpy(int) + 1190,
])

assert np.array_equal(
    np.bincount(node_gid, minlength=1301),
    U["CountGD"].to_numpy(int),
)

assert np.max(
    np.abs(
        np.bincount(
            node_gid,
            weights=r,
            minlength=1301,
        ) - ref
    )
) < 1e-3

print("V4 contribution sums: PASS", flush=True)

print("\n[6] Align frozen A3 scores...", flush=True)

parts = []

for s in range(0, 1301, 25):
    e = min(s + 25, 1301)
    path = (
        A3 / "parts" /
        f"part_{s:04d}_{e-1:04d}_nodes.npz"
    )

    assert path.is_file(), path

    with np.load(path, allow_pickle=False) as z:
        x = pd.DataFrame({
            k: z[k].copy()
            for k in (
                "pair_index",
                "raw_query_index",
                "a3_score",
            )
        })

    assert (
        (x.pair_index >= s) & (x.pair_index < e)
    ).all()

    parts.append(x)

a3 = pd.concat(parts, ignore_index=True)

assert len(a3) == len(r)
assert not a3.duplicated(
    ["pair_index", "raw_query_index"]
).any()

pair_idx = node_gid.copy()

raw_idx = np.concatenate([
    posnodes.raw_query_index.to_numpy(int),
    negnodes.raw_query_index.to_numpy(int),
])

keys = pd.MultiIndex.from_arrays([
    a3.pair_index.to_numpy(int),
    a3.raw_query_index.to_numpy(int),
])

target = pd.MultiIndex.from_arrays([
    pair_idx,
    raw_idx,
])

index = keys.get_indexer(target)
missing = int((index < 0).sum())

print(
    "A3-V4 node alignment:",
    len(index) - missing,
    "/",
    len(index),
    flush=True,
)

assert missing == 0
assert keys.is_unique and target.is_unique

sem = a3.a3_score.to_numpy(float)[index]
assert np.isfinite(sem).all()

print("\n[7] Save audited features...", flush=True)

OUT.mkdir(parents=True, exist_ok=False)

np.savez_compressed(
    OUT / "frozen_v4_a3_unified_nodes.npz",
    node_graph_id=node_gid.astype(np.int32),
    raw_query_index=raw_idx.astype(np.int32),
    v4_contribution=r.astype(np.float32),
    a3_score=sem.astype(np.float32),
)

(OUT / "audit.json").write_text(
    json.dumps({
        "pairs": 1301,
        "positive_nodes": len(rp),
        "absent_nodes": len(rn),
        "v4_sha256": h.hexdigest(),
        "max_count_parity_error": maxdiff,
        "a3_raw_index_alignment": "PASS",
        "no_training": True,
        "no_threshold_tuning": True,
        "blind_claim": False,
    }, indent=2),
    encoding="utf-8",
)

print("=" * 72)
print("P14-08A PASS")
print("Frozen V4 node contributions + A3 scores aligned")
print(
    "Output:",
    OUT / "frozen_v4_a3_unified_nodes.npz",
)
print("=" * 72)
