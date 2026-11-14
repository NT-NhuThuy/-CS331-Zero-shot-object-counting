#!/usr/bin/env python3
"""
P17-01: Fair paired evaluation.

A: FSC147 official TEST (1190 positive pairs)
B: 111 hard-negative absent pairs
C: Unified (1301 pairs)

Cache-only. No GPU, training, or model inference.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BASE = ROOT / "outputs/joint_semantic_counting_v1"

MANIFEST = (
    ROOT /
    "generated/fsc147_presence_v4/test_pairs_1301.csv"
)

UNIFIED = (
    BASE /
    "p14_06_unified_matched_v1/unified_predictions.csv"
)

A3 = (
    BASE /
    "p14_07b_a3_fullraw_fsc_unified_v1/"
    "a3_unified_1301_predictions.csv"
)

OUT = BASE / "p17_01_three_view_paired_v1"


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)

    return h.hexdigest()


def align(df, manifest, path):
    assert "pair_id" in df.columns, path
    assert len(df) == 1301, path
    assert df["pair_id"].is_unique, path

    assert set(df["pair_id"]) == set(
        manifest["pair_id"]
    ), f"Pair ID mismatch: {path}"

    return (
        df.set_index("pair_id")
        .loc[manifest["pair_id"]]
        .reset_index()
    )


def metrics(pred, gt, baseline):
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    baseline = np.asarray(baseline, dtype=float)

    error = pred - gt
    ae = np.abs(error)
    baseline_ae = np.abs(baseline - gt)

    # Accuracy uses integer counts consistently.
    rounded = np.maximum(0, np.rint(pred))
    rounded_ae = np.abs(rounded - gt)

    gain = baseline_ae - ae
    eps = 1e-9

    result = {
        "N": len(gt),
        "MAE": float(ae.mean()),
        "RMSE": float(
            np.sqrt(np.mean(error ** 2))
        ),
        "Exact_Acc_pct": float(
            100 * np.mean(rounded_ae == 0)
        ),
        "Acc_pm1_pct": float(
            100 * np.mean(rounded_ae <= 1)
        ),
        "Acc_pm5_pct": float(
            100 * np.mean(rounded_ae <= 5)
        ),
        "Bias": float(error.mean()),
        "HELP": int((gain > eps).sum()),
        "HARM": int((gain < -eps).sum()),
        "SAME": int(
            (np.abs(gain) <= eps).sum()
        ),
        "Mean_AE_Gain": float(gain.mean()),
    }

    assert (
        result["HELP"]
        + result["HARM"]
        + result["SAME"]
        == result["N"]
    )

    if np.all(gt > 0):
        result["Relative_Acc10_pct"] = float(
            100 * np.mean(
                rounded_ae <= 0.10 * gt
            )
        )

    if np.all(gt == 0):
        result["Zero_Acc_pct"] = result[
            "Exact_Acc_pct"
        ]
        result["Strict_Zero_pct"] = float(
            100 * np.mean(np.abs(pred) < eps)
        )
        result["Mean_False_Count"] = float(
            np.maximum(pred, 0).mean()
        )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preflight", action="store_true"
    )

    parser.add_argument(
        "--extra-csv",
        type=Path,
        default=None,
        help="Optional extra prediction CSV keyed by pair_id"
    )

    args = parser.parse_args()

    assert MANIFEST.is_file(), MANIFEST
    assert UNIFIED.is_file(), UNIFIED

    manifest = pd.read_csv(MANIFEST)

    assert len(manifest) == 1301
    assert manifest["pair_id"].is_unique

    labels = manifest[
        "presence_label"
    ].to_numpy(dtype=int)

    gt = manifest[
        "count_target"
    ].to_numpy(dtype=float)

    assert int((labels == 1).sum()) == 1190
    assert int((labels == 0).sum()) == 111

    assert np.isfinite(gt).all()
    assert (gt >= 0).all()

    assert np.all(gt[labels == 0] == 0)
    assert np.all(gt[labels == 1] > 0)

    assert np.allclose(gt, np.rint(gt))

    unified = align(
        pd.read_csv(UNIFIED),
        manifest,
        UNIFIED
    )

    assert np.allclose(
        unified["gt_count"].to_numpy(float),
        gt
    )

    frames = [("P14-06", unified)]
    sources = [MANIFEST, UNIFIED]

    if A3.is_file():
        a3 = align(
            pd.read_csv(A3),
            manifest,
            A3
        )

        assert np.array_equal(
            a3["presence_label"].to_numpy(int),
            labels
        )

        # A3 CSV may omit GT; the GT source of truth is
        # the manifest, already checked against unified.
        gt_columns = [
            col for col in ("gt_count", "count_target")
            if col in a3.columns
        ]

        for col in gt_columns:
            assert np.allclose(
                a3[col].to_numpy(float),
                gt
            ), f"A3 GT mismatch in {col}"

        if not gt_columns:
            print(
                "[A3 SCHEMA] No GT column; "
                "using validated manifest GT"
            )

        frames.append(("P14-07B", a3))
        sources.append(A3)

    if args.extra_csv is not None:
        assert args.extra_csv.is_file()

        extra = align(
            pd.read_csv(args.extra_csv),
            manifest,
            args.extra_csv
        )

        frames.append(("EXTRA", extra))
        sources.append(args.extra_csv)

    print("=" * 78)
    print("P17-01 — THREE-VIEW FAIR PAIRED EVALUATION")
    print("FSC147 TEST 1190 / ABSENT 111 / UNIFIED 1301")
    print("NO GPU / NO RETUNING / CACHE ONLY")
    print("=" * 78)

    for name, df in frames:
        print(
            f"[COLUMNS {name}]",
            df.columns.tolist()
        )

    selected = {}

    known_names = {
        "CountGD",
        "V4 seed42",
        "CountGD + A3",
        "V4 seed42 + A3",
    }

    def add_model(name, column, df):
        assert name not in selected, (
            "Duplicate model", name
        )

        pred = pd.to_numeric(
            df[column],
            errors="raise"
        ).to_numpy(dtype=float)

        assert len(pred) == 1301
        assert np.isfinite(pred).all()
        assert (pred >= -1e-6).all()

        selected[name] = np.maximum(
            pred, 0.0
        )

    excluded_tokens = (
        "score", "threshold", "prob",
        "label", "pair_id", "image",
        "query", "delta", "error",
        "gain", "mae", "rmse", "gate"
    )

    for source, df in frames:
        for column in df.columns:
            lower = column.lower()

            known = column in known_names

            joint = (
                "joint" in lower
                or "v3.2" in lower
                or "v3_2" in lower
            )

            excluded = any(
                token in lower
                for token in excluded_tokens
            )

            numeric = pd.api.types.is_numeric_dtype(
                df[column]
            )

            if (
                (known or joint)
                and not excluded
                and numeric
                and column not in selected
            ):
                add_model(column, column, df)

    assert "CountGD" in selected, (
        "CountGD baseline missing"
    )

    print("\n[AVAILABLE MODELS]")

    for name in selected:
        print(" -", name)

    has_joint = any(
        "joint" in x.lower()
        or "v3.2" in x.lower()
        or "v3_2" in x.lower()
        for x in selected
    )

    if not has_joint:
        print(
            "[WARNING] Joint predictions not found."
        )
        print(
            "Do not infer Joint predictions "
            "from CountGD or V4."
        )

    if args.preflight:
        print("\nP17-01 PREFLIGHT PASS")
        print("NO OUTPUT WRITES")
        return

    masks = {
        "FSC147_TEST_1190": labels == 1,
        "HARD_NEGATIVE_111": labels == 0,
        "UNIFIED_1301": np.ones(
            1301, dtype=bool
        ),
    }

    baseline = selected["CountGD"]

    rows = []

    predictions = manifest[
        [
            "pair_id",
            "file_name",
            "query",
            "presence_label",
            "count_target",
        ]
    ].copy()

    for model_name, pred in selected.items():
        predictions[
            f"{model_name}__prediction"
        ] = pred

        predictions[
            f"{model_name}__AE"
        ] = np.abs(pred - gt)

        for group_name, mask in masks.items():
            row = {
                "Dataset": group_name,
                "Model": model_name,
            }

            row.update(
                metrics(
                    pred[mask],
                    gt[mask],
                    baseline[mask]
                )
            )

            rows.append(row)

    report = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("RESULT A — FSC147 TEST (1190)")
    print("=" * 78)

    a = report[
        report["Dataset"] == "FSC147_TEST_1190"
    ]

    print(
        a.round(4).to_string(index=False)
    )

    print("\n" + "=" * 78)
    print("RESULT B — 111 HARD NEGATIVES")
    print("=" * 78)

    b = report[
        report["Dataset"] == "HARD_NEGATIVE_111"
    ]

    print(
        b.round(4).to_string(index=False)
    )

    print("\n" + "=" * 78)
    print("RESULT C — UNIFIED 1301")
    print("=" * 78)

    c = report[
        report["Dataset"] == "UNIFIED_1301"
    ]

    print(
        c.round(4).to_string(index=False)
    )

    print("\n[1] BALANCED MAE")

    for name in selected:
        p = a[a["Model"] == name].iloc[0]
        n = b[b["Model"] == name].iloc[0]
        u = c[c["Model"] == name].iloc[0]

        balanced = (
            p["MAE"] + n["MAE"]
        ) / 2

        weighted = (
            1190 * p["MAE"]
            + 111 * n["MAE"]
        ) / 1301

        assert np.isclose(
            weighted, u["MAE"]
        )

        weighted_rmse = np.sqrt(
            (
                1190 * p["RMSE"] ** 2
                + 111 * n["RMSE"] ** 2
            ) / 1301
        )

        assert np.isclose(
            weighted_rmse,
            u["RMSE"]
        )

        print(
            f"{name:26s} "
            f"BalancedMAE={balanced:.4f} "
            f"UnifiedMAE={u['MAE']:.4f}"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    report.to_csv(
        OUT / "three_view_metrics.csv",
        index=False
    )

    predictions.to_csv(
        OUT / "paired_predictions_and_errors.csv",
        index=False
    )

    provenance = {
        "version": "P17-01-V1",
        "evaluation": "posthoc_three_view",
        "note": (
            "FSC147 TEST already exposed; "
            "111 negatives hard-mined on CountGD"
        ),
        "models": list(selected),
        "dataset_sizes": {
            key: int(mask.sum())
            for key, mask in masks.items()
        },
        "source_sha256": {
            str(p): sha256(p)
            for p in sources
        },
    }

    (
        OUT / "provenance.json"
    ).write_text(
        json.dumps(
            provenance,
            indent=2
        )
    )

    print("\nP17-01 COMPLETE")
    print("Saved:", OUT)
    print("NO MODEL INFERENCE")


if __name__ == "__main__":
    main()
