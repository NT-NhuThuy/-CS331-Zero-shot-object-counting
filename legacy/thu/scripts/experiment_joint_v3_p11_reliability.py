from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge

ROOT = Path("outputs/joint_semantic_counting_v1")
OOF = ROOT / "p3_oof_v4_v1/assembled_oof_v1"
SEM = ROOT / "joint_v2_fsc_train_features_v1"

SEED = 42
LOG_CAP = np.log(4.0)
ALPHA = 100.0


def load_train():
    g = pd.read_csv(
        OOF / "oof_train_graph_predictions.csv"
    ).sort_values("graph_id").reset_index(drop=True)

    with np.load(
        OOF / "oof_train_node_contributions.npz",
        allow_pickle=False
    ) as z:
        ids = z["global_node_id"].astype(np.int64)
        gid = z["node_graph_id"].astype(np.int64)
        raw = z["raw_query_index"].astype(np.int64)
        r = z["oof_structural_contribution"].astype(float)

    with np.load(
        SEM / "frozen_train_node_scores.npz",
        allow_pickle=False
    ) as z:
        assert np.array_equal(z["global_node_id"], ids)
        assert np.array_equal(z["node_graph_id"], gid)
        assert np.array_equal(z["raw_query_index"], raw)
        s = z["semantic_score"].astype(float)

    G, N = 3659, 190353

    assert len(g) == G
    assert len(r) == len(s) == N
    assert np.array_equal(g["graph_id"], np.arange(G))
    assert np.array_equal(ids, np.arange(N))
    assert (r >= 0).all()
    assert np.isfinite(r).all()
    assert np.isfinite(s).all()

    gt = g["gt_count"].to_numpy(float)
    v4 = g["oof_v4_count"].to_numpy(float)
    native = g["baseline_count"].to_numpy(float)

    assert np.max(np.abs(
        np.bincount(gid, weights=r, minlength=G) - v4
    )) < 1e-3

    n = np.bincount(gid, minlength=G)
    assert np.array_equal(n, native.astype(int))

    den = np.maximum(n, 1)

    def frac(x):
        return np.bincount(
            gid, weights=x.astype(float), minlength=G
        ) / den

    mean_r = v4 / den

    std_r = np.sqrt(np.maximum(
        np.bincount(
            gid, weights=r*r, minlength=G
        ) / den - mean_r**2, 0
    ))

    mean_s = np.bincount(
        gid, weights=s, minlength=G
    ) / den

    std_s = np.sqrt(np.maximum(
        np.bincount(
            gid, weights=s*s, minlength=G
        ) / den - mean_s**2, 0
    ))

    weighted_s = np.bincount(
        gid, weights=s*r, minlength=G
    ) / np.maximum(v4, 1e-8)

    Xs = np.column_stack([
        np.log1p(native),
        np.log1p(v4),
        native-v4,
        v4/(native+1),
        np.log1p(np.abs(native-v4)),
        mean_r,
        std_r,
        frac(r < 0.5),
        frac(r > 1.0),
    ])

    Xj = np.column_stack([
        Xs,
        mean_s,
        std_s,
        weighted_s,
        frac(s < 0.5),
        frac(s > 0.99),
    ])

    assert Xs.shape == (G, 9)
    assert Xj.shape == (G, 14)
    assert np.isfinite(Xj).all()

    groups = (
        g["query"].astype(str)
        .str.strip().str.casefold().to_numpy()
    )

    assert np.unique(groups).size == 89

    return dict(
        gt=gt, v4=v4, native=native,
        xs=Xs, xj=Xj, groups=groups
    )


def logistic():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=SEED,
        )
    )


def ridge():
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=ALPHA)
    )


def fit_candidate(data, idx):
    gt = data["gt"][idx]
    v4 = data["v4"][idx]
    xs = data["xs"][idx]
    xj = data["xj"][idx]

    over = v4 > gt + 1e-4
    under = gt > v4 + 1e-4
    valid = over | under

    assert under.sum() > 20
    assert over.sum() > 20

    structural_router = logistic()
    semantic_router = logistic()

    structural_router.fit(
        xs[valid], over[valid].astype(int)
    )

    semantic_router.fit(
        xj[valid], over[valid].astype(int)
    )

    increase = ridge()
    decrease = ridge()

    increase.fit(
        xs[under],
        np.clip(
            np.log(
                (gt[under]+1)/(v4[under]+1)
            ), 0, LOG_CAP
        ),
    )

    decrease.fit(
        xs[over],
        np.clip(
            np.log(
                (v4[over]+1)/(gt[over]+1)
            ), 0, LOG_CAP
        ),
    )

    return (
        structural_router,
        semantic_router,
        increase,
        decrease,
    )


def candidate(data, idx, models, joint):
    structural_router, semantic_router, up, down = models

    xs = data["xs"][idx]

    if joint:
        router = semantic_router
        xroute = data["xj"][idx]
    else:
        router = structural_router
        xroute = xs

    p = router.predict_proba(xroute)[:, 1]

    u = np.clip(up.predict(xs), 0, LOG_CAP)
    d = np.clip(down.predict(xs), 0, LOG_CAP)

    v4 = data["v4"][idx]

    log_delta = (1-p)*u - p*d

    pred = np.maximum(
        0,
        (1+v4)*np.exp(log_delta)-1
    )

    pred = np.where(
        data["native"][idx] > 0,
        pred,
        0,
    )

    return dict(
        pred=pred, p=p, up=u, down=d
    )


def risk_features(data, idx, cand, semantic):
    v4 = data["v4"][idx]
    delta = cand["pred"] - v4

    p = cand["p"]
    u = cand["up"]
    d = cand["down"]

    x = (
        data["xj"][idx]
        if semantic
        else data["xs"][idx]
    )

    extra = np.column_stack([
        p,
        np.abs(2*p-1),
        u,
        d,
        u-d,
        np.sign(delta),
        np.log1p(np.abs(delta)),
        delta/(1+v4),
        np.abs(delta)/(1+v4),
    ])

    features = np.column_stack([x, extra])
    assert np.isfinite(features).all()

    return features


def risk_targets(v4, gt, pred):
    baseline_error = v4-gt
    corrected_error = pred-gt

    step = np.abs(pred-v4)

    mae_gain = np.zeros_like(step)
    mse_gain = np.zeros_like(step)

    mask = step > 1e-7

    # Normalized absolute-error improvement.
    mae_gain[mask] = (
        np.abs(baseline_error[mask])
        - np.abs(corrected_error[mask])
    ) / step[mask]

    # Normalized squared-error improvement.
    denominator = (
        step[mask]
        * (2*np.abs(baseline_error[mask])+step[mask])
    )

    signed_step = pred[mask]-v4[mask]

    mse_gain[mask] = (
        -signed_step
        * (2*baseline_error[mask]+signed_step)
        / denominator
    )

    # Both normalized targets are bounded.
    assert np.max(np.abs(mae_gain)) <= 1.000001
    assert np.max(np.abs(mse_gain)) <= 1.000001

    return (
        np.clip(mae_gain, -1, 1),
        np.clip(mse_gain, -1, 1),
    )


def fit_reliability(features, mae_target, mse_target):
    mae_model = ridge()
    mse_model = ridge()

    mae_model.fit(features, mae_target)
    mse_model.fit(features, mse_target)

    return mae_model, mse_model


def apply_reliability(data, idx, cand, models, semantic):
    features = risk_features(
        data, idx, cand, semantic
    )

    mae_model, mse_model = models

    predicted_mae_gain = mae_model.predict(features)
    predicted_mse_gain = mse_model.predict(features)

    # Fixed conservative decision rule:
    # both expected improvements must be positive.
    accept = (
        (predicted_mae_gain > 0)
        & (predicted_mse_gain > 0)
    )

    pred = np.where(
        accept,
        cand["pred"],
        data["v4"][idx],
    )

    return pred, accept


def inner_risk_examples(data, outer_idx):
    cv = GroupKFold(n_splits=4)

    buckets = {
        k: ([], [], [])
        for k in (
            "struct",
            "joint_struct",
            "joint_sem",
        )
    }

    for inner_tr, inner_te in cv.split(
        data["xs"][outer_idx],
        groups=data["groups"][outer_idx],
    ):
        tr = outer_idx[inner_tr]
        te = outer_idx[inner_te]

        assert set(data["groups"][tr]).isdisjoint(
            set(data["groups"][te])
        )

        models = fit_candidate(data, tr)

        structural = candidate(
            data, te, models, joint=False
        )

        joint = candidate(
            data, te, models, joint=True
        )

        for branch, cand, semantic in [
            ("struct", structural, False),
            ("joint_struct", joint, False),
            ("joint_sem", joint, True),
        ]:
            features = risk_features(
                data, te, cand, semantic
            )

            mae_gain, mse_gain = risk_targets(
                data["v4"][te],
                data["gt"][te],
                cand["pred"],
            )

            X, A, B = buckets[branch]

            X.append(features)
            A.append(mae_gain)
            B.append(mse_gain)

    return {
        name: tuple(
            np.concatenate(values)
            for values in bucket
        )
        for name, bucket in buckets.items()
    }


def metrics(pred, gt):
    error = pred-gt

    return (
        np.abs(error).mean(),
        np.sqrt(np.mean(error**2)),
        error.mean(),
    )


def query_bootstrap(delta, groups):
    unique, inverse = np.unique(
        groups, return_inverse=True
    )

    K = len(unique)

    sizes = np.bincount(
        inverse, minlength=K
    )

    sums = np.bincount(
        inverse, weights=delta, minlength=K
    )

    rng = np.random.default_rng(SEED)

    samples = rng.integers(
        0, K, size=(2000, K)
    )

    boot = (
        sums[samples].sum(axis=1)
        / sizes[samples].sum(axis=1)
    )

    return np.quantile(boot, [0.025, 0.975])


def main():
    data = load_train()
    G = len(data["gt"])

    names = [
        "raw_struct",
        "raw_joint",
        "p11_struct",
        "p11_joint_structrisk",
        "p11_joint_semrisk",
    ]

    predictions = {
        name: np.full(G, np.nan)
        for name in names
    }

    gated_names = [
        "p11_struct",
        "p11_joint_structrisk",
        "p11_joint_semrisk",
    ]

    accepted = {
        name: np.zeros(G, dtype=bool)
        for name in gated_names
    }

    print("="*72)
    print("P11 — JOINT CORRECTION RELIABILITY")
    print("FSC TRAIN OOF ONLY")
    print("="*72)

    cv = GroupKFold(n_splits=5)

    print("\n[1] OUTER QUERY-GROUPED CV")

    for fold, (tr, te) in enumerate(
        cv.split(
            data["xs"],
            groups=data["groups"],
        ), 1
    ):
        assert set(
            data["groups"][tr]
        ).isdisjoint(
            set(data["groups"][te])
        )

        candidate_models = fit_candidate(data, tr)

        structural = candidate(
            data, te, candidate_models, joint=False
        )

        joint = candidate(
            data, te, candidate_models, joint=True
        )

        predictions["raw_struct"][te] = structural["pred"]
        predictions["raw_joint"][te] = joint["pred"]

        # All reliability targets are generated by
        # inner held-out candidate predictions.
        risk_examples = inner_risk_examples(data, tr)

        for branch, cand, semantic, name in [
            (
                "struct", structural, False,
                "p11_struct"
            ),
            (
                "joint_struct", joint, False,
                "p11_joint_structrisk"
            ),
            (
                "joint_sem", joint, True,
                "p11_joint_semrisk"
            ),
        ]:
            X, mae_target, mse_target = risk_examples[branch]

            risk_models = fit_reliability(
                X, mae_target, mse_target
            )

            pred, gate = apply_reliability(
                data, te, cand, risk_models, semantic
            )

            predictions[name][te] = pred
            accepted[name][te] = gate

        print(
            f"Fold {fold}/5 complete;"
            f" heldout_graphs={len(te)}",
            flush=True,
        )

    assert all(
        np.isfinite(p).all()
        for p in predictions.values()
    )

    print("\n[2] COUNTING METRICS — QUERY HELD OUT")

    for name, pred in [
        ("OOF V4", data["v4"]),
        *predictions.items(),
    ]:
        mae, rmse, bias = metrics(
            pred, data["gt"]
        )

        print(
            f"{name:24s}"
            f" MAE={mae:.6f}"
            f" RMSE={rmse:.6f}"
            f" Bias={bias:+.6f}"
        )

    under = data["v4"] < data["gt"]-1e-4
    over = data["v4"] > data["gt"]+1e-4

    print("\n[3] RELIABILITY ACCEPTANCE")

    for name, gate in accepted.items():
        print(
            f"{name:24s}"
            f" accept={gate.mean():.4f}"
            f" under={gate[under].mean():.4f}"
            f" over={gate[over].mean():.4f}"
        )

    print("\n[4] QUERY-BOOTSTRAP DELTA MAE")

    comparisons = [
        (
            "Joint semantic risk - Joint structural risk",
            "p11_joint_semrisk",
            "p11_joint_structrisk",
        ),
        (
            "P11 Joint - Raw P8 Joint",
            "p11_joint_semrisk",
            "raw_joint",
        ),
        (
            "P11 Joint - OOF V4",
            "p11_joint_semrisk",
            None,
        ),
        (
            "P11 Joint - P11 Structural",
            "p11_joint_semrisk",
            "p11_struct",
        ),
    ]

    for label, first, second in comparisons:
        ref = (
            data["v4"]
            if second is None
            else predictions[second]
        )

        delta = (
            np.abs(predictions[first]-data["gt"])
            - np.abs(ref-data["gt"])
        )

        lo, hi = query_bootstrap(
            delta, data["groups"]
        )

        print(
            f"{label}:"
            f" delta={delta.mean():+.6f}"
            f" CI=[{lo:+.6f}, {hi:+.6f}]"
        )

    print("\n[5] COUNT DENSITY AND ERROR DIRECTION")

    for label, mask in [
        ("UNDERCOUNT", under),
        ("OVERCOUNT", over),
        ("GT > 100", data["gt"] > 100),
    ]:
        print(
            f"{label:12s} n={mask.sum():4d}"
        )

        for name, pred in [
            ("OOF V4", data["v4"]),
            ("Raw P8 Joint", predictions["raw_joint"]),
            (
                "P11 Joint",
                predictions["p11_joint_semrisk"]
            ),
        ]:
            mae = np.abs(
                pred[mask]-data["gt"][mask]
            ).mean()

            print(f"  {name:18s} MAE={mae:.6f}")

    print("\nDONE — FSC TRAIN OOF ONLY")
    print("NO FSC VAL / NO TEST / NO GPU / NO DATA WRITES")
    print(
        "LIMITATION: base V4 OOF models are not "
        "nested within P11 folds."
    )


if __name__ == "__main__":
    main()
