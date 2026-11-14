from pathlib import Path
import argparse, hashlib, json, os, sys
import joblib
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
B = R / "outputs/joint_semantic_counting_v1"
P12 = B / "joint_v3_p12_locked_transfer_v1"
OUT = B / "p14_08b_09_joint_unified_frozen_v1"

FILES = {
    "p11": R / "scripts/experiment_joint_v3_p11_reliability.py",
    "lock": P12 / "lock_manifest.json",
    "model": P12 / "locked_p11_models.joblib",
    "vg": B / "joint_v1_fsc_val_failure_audit_v1/development_graph_diagnostics.csv",
    "vn": B / "joint_v1_fsc_val_failure_audit_v1/frozen_node_scores.npz",
    "vp": P12 / "p12_fsc_val_predictions.csv",
    "u": B / "p14_06_unified_matched_v1/unified_predictions.csv",
    "a3": B / "p14_07b_a3_fullraw_fsc_unified_v1/a3_unified_1301_predictions.csv",
    "n": B / "p14_08a_frozen_v4_node_contrib_unified_v1/frozen_v4_a3_unified_nodes.npz",
    "g": R / "outputs/fsc147_presence_v4/official_test_v2/official_test_predictions.csv",
}

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(p, x):
    t = p.with_name(p.name + ".tmp")
    t.write_text(json.dumps(x, indent=2))
    os.replace(t, p)

def write_csv(p, x):
    t = p.with_name(p.name + ".tmp")
    x.to_csv(t, index=False)
    os.replace(t, p)

def features(gid, r, s, native, v4):
    gid = np.asarray(gid, np.int64)
    r, s, native, v4 = (
        np.asarray(z, float)
        for z in (r, s, native, v4)
    )

    G = len(native)
    n = np.bincount(gid, minlength=G)

    assert len(v4) == G
    assert len(r) == len(s) == len(gid)
    assert np.isfinite(r).all() and np.isfinite(s).all()
    assert (gid >= 0).all() and (gid < G).all()
    assert (r >= 0).all()
    assert (native >= 0).all() and (v4 >= 0).all()
    assert np.array_equal(n, native.astype(int))

    delta = float(np.max(np.abs(
        np.bincount(gid, weights=r, minlength=G) - v4
    )))
    assert delta < 1e-3, f"V4 contribution mismatch: {delta}"

    den = np.maximum(n, 1)

    def f(z):
        return np.bincount(
            gid,
            weights=z.astype(float),
            minlength=G,
        ) / den

    mr = v4 / den
    sr = np.sqrt(np.maximum(f(r*r) - mr*mr, 0))
    ms = f(s)
    ss = np.sqrt(np.maximum(f(s*s) - ms*ms, 0))

    ws = np.bincount(
        gid, weights=s*r, minlength=G
    ) / np.maximum(v4, 1e-8)

    xs = np.column_stack([
        np.log1p(native),
        np.log1p(v4),
        native-v4,
        v4/(native+1),
        np.log1p(abs(native-v4)),
        mr, sr,
        f(r < 0.5),
        f(r > 1),
    ])

    xj = np.column_stack([
        xs, ms, ss, ws,
        f(s < 0.5),
        f(s > 0.99),
    ])

    assert xs.shape == (G, 9)
    assert xj.shape == (G, 14)
    assert np.isfinite(xj).all()

    return dict(xs=xs, xj=xj, v4=v4, native=native)

def infer(data, models, candidate, apply):
    idx = np.arange(len(data["v4"]))
    cm = models["candidate_models"]
    rm = models["risk_models"]

    assert len(cm) == 4
    assert set(rm) == {"struct", "joint_struct", "joint_sem"}

    sc = candidate(data, idx, cm, joint=False)
    jc = candidate(data, idx, cm, joint=True)

    a, ga = apply(
        data, idx, sc,
        rm["struct"], semantic=False
    )

    b, gb = apply(
        data, idx, jc,
        rm["joint_struct"], semantic=False
    )

    c, gc = apply(
        data, idx, jc,
        rm["joint_sem"], semantic=True
    )

    out = {
        "P8_joint": jc["pred"],
        "P11_struct": a,
        "P11_joint_structrisk": b,
        "Joint_V32": c,
        "p_over": jc["p"],
        "accepted": gc.astype(int),
    }

    for k, x in out.items():
        assert len(x) == len(idx)
        assert np.isfinite(x).all(), k
        if k != "p_over":
            assert (x >= 0).all(), k

    return out

def replay(models, candidate, apply):
    v = pd.read_csv(FILES["vg"]).sort_values(
        "graph_id"
    ).reset_index(drop=True)

    assert len(v) == 1286
    assert np.array_equal(
        v.graph_id.to_numpy(), np.arange(1286)
    )

    with np.load(FILES["vn"], allow_pickle=False) as z:
        d = features(
            z["node_graph_id"],
            z["structural_contribution"],
            z["semantic_score"],
            v.native,
            v.v4,
        )

    pred = infer(d, models, candidate, apply)

    expected = pd.read_csv(FILES["vp"]).sort_values(
        "graph_id"
    ).reset_index(drop=True)

    assert len(expected) == 1286

    for key, col in [
        ("P8_joint", "p8_joint_count"),
        ("P11_struct", "p11_struct_count"),
        ("P11_joint_structrisk", "p11_joint_structrisk_count"),
        ("Joint_V32", "p11_joint_semrisk_count"),
    ]:
        err = float(np.max(np.abs(
            pred[key] - expected[col].to_numpy(float)
        )))

        print(
            f"VAL {key}: max_diff={err:.9g}",
            flush=True,
        )

        assert err < 1e-4, f"VAL replay failed: {key}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print("=" * 75)
    print("P14-08B/09 — FROZEN JOINT V3.2 UNIFIED 1301")
    print("CPU ONLY / NO TRAIN / NO HOSTNAME GUARD")
    print("=" * 75, flush=True)

    for k, p in FILES.items():
        assert p.is_file(), f"Missing {k}: {p}"

    lock = json.loads(FILES["lock"].read_text())

    assert sha(FILES["model"]) == lock["model_sha256"]

    if "p11" in lock.get("source_sha256", {}):
        assert (
            sha(FILES["p11"])
            == lock["source_sha256"]["p11"]
        ), "Frozen P11 source mismatch"

    fp = {k: sha(p) for k, p in FILES.items()}
    manifest = OUT / "inputs.sha256.json"

    if OUT.exists():
        assert args.resume, "Output exists; use --resume"
        assert (
            manifest.is_file()
            and json.loads(manifest.read_text()) == fp
        ), "Source changed / partial prior run"
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        write_json(manifest, fp)

    sys.path.insert(0, str(R / "scripts"))

    from experiment_joint_v3_p11_reliability import (
        candidate, apply_reliability
    )

    models = joblib.load(FILES["model"])

    # STAGE 1 — VAL REPLAY
    replay_done = OUT / "val_replay_pass.json"

    if replay_done.exists():
        assert json.loads(replay_done.read_text()) == {
            "status": "PASS"
        }
        print("[RESUME] VAL replay PASS", flush=True)
    else:
        print("\n[1/3] Frozen P12 VAL replay", flush=True)
        replay(models, candidate, apply_reliability)
        write_json(replay_done, {"status": "PASS"})
        print("VAL REPLAY PASS", flush=True)

    # STAGE 2 — JOINT INFERENCE
    stage = OUT / "joint_stage.csv"

    if stage.exists():
        assert args.resume
        raw = pd.read_csv(stage)
        print("[RESUME] Saved Joint predictions", flush=True)
    else:
        print("\n[2/3] Frozen Joint inference", flush=True)

        u = pd.read_csv(FILES["u"])
        a3 = pd.read_csv(FILES["a3"])

        assert len(u) == len(a3) == 1301
        assert u.pair_id.tolist() == a3.pair_id.tolist()

        with np.load(FILES["n"], allow_pickle=False) as z:
            d = features(
                z["node_graph_id"],
                z["v4_contribution"],
                z["a3_score"],
                u["CountGD"],
                u["V4 seed42"],
            )

        pr = infer(d, models, candidate, apply_reliability)

        raw = u[["pair_id"]].copy()
        for k, x in pr.items():
            raw[k] = x

        write_csv(stage, raw)
        print("JOINT INFERENCE PASS", flush=True)

    # STAGE 3 — UNIFIED EVALUATION
    u = pd.read_csv(FILES["u"])
    a3 = pd.read_csv(FILES["a3"])

    assert len(raw) == len(u) == len(a3) == 1301
    assert (
        raw.pair_id.tolist()
        == u.pair_id.tolist()
        == a3.pair_id.tolist()
    )

    pm = pd.read_csv(FILES["g"])
    assert pm.pair_id.is_unique

    pm = u[["pair_id"]].merge(
        pm[[
            "pair_id",
            "presence_label",
            "presence_prediction_fusion_v2",
            "presence_prediction_clip",
        ]],
        on="pair_id",
        validate="one_to_one",
        sort=False,
    )

    assert len(pm) == 1301
    assert np.array_equal(
        pm.presence_label.to_numpy(int),
        u.presence_label.to_numpy(int),
    )

    def gate(x):
        y = pd.to_numeric(
            x, errors="raise"
        ).to_numpy(int)

        assert np.isin(y, [0, 1]).all()
        return y == 1

    keep = {
        "A3": gate(a3.a3_present),
        "CLIP": gate(pm.presence_prediction_clip),
        "Fusion V2": gate(
            pm.presence_prediction_fusion_v2
        ),
    }

    pos = u.presence_label.to_numpy(int) == 1
    neg = ~pos

    assert pos.sum() == 1190
    assert neg.sum() == 111

    assert int((~keep["CLIP"] & neg).sum()) == 39
    assert int((~keep["Fusion V2"] & neg).sum()) == 22
    assert int((~keep["CLIP"] & pos).sum()) == 6
    assert int((~keep["Fusion V2"] & pos).sum()) == 6

    v4 = u["V4 seed42"].to_numpy(float)

    assert np.allclose(
        np.where(keep["A3"], v4, 0),
        a3["V4 seed42 + A3"],
        atol=1e-5,
        rtol=0,
    )

    if (OUT / "audit.json").exists():
        assert args.resume
        print("[RESUME] Final output present:", OUT, flush=True)
        return

    print("\n[3/3] WHAT + HOW MANY metrics", flush=True)

    joint = raw.Joint_V32.to_numpy(float)

    pred = {
        "CountGD": u["CountGD"].to_numpy(float),
        "V4": v4,
        "P8 Joint": raw.P8_joint.to_numpy(float),
        "P11 Structural": raw.P11_struct.to_numpy(float),
        "P11 Joint + Structural Risk":
            raw.P11_joint_structrisk.to_numpy(float),
        "Joint V3.2": joint,
    }

    for k, g in keep.items():
        pred["Joint V3.2 + " + k] = np.where(
            g, joint, 0
        )

    gt = u.gt_count.to_numpy(float)

    full = u[[
        "pair_id", "file_name", "query",
        "presence_label", "gt_count"
    ]].copy()

    for name, g in keep.items():
        full[name + "_present"] = g.astype(int)

    full["p_over"] = raw.p_over
    full["accepted_correction"] = (
        raw.accepted.astype(int)
    )

    rows = []

    for name, x in pred.items():
        assert np.isfinite(x).all()
        assert (x >= 0).all()

        full[name] = x

        for group, mask in [
            ("positive", pos),
            ("absent", neg),
            ("unified", np.ones(1301, bool)),
        ]:
            e = x[mask] - gt[mask]

            rows.append(dict(
                model=name,
                group=group,
                n=int(mask.sum()),
                mae=float(np.mean(abs(e))),
                mse=float(np.mean(e*e)),
                rmse=float(np.sqrt(np.mean(e*e))),
                zero=int((x[mask] == 0).sum()),
            ))

    met = pd.DataFrame(rows)

    for name in pred:
        x = met[met.model == name].set_index("group")

        for k in ("mae", "mse"):
            expected = (
                1190*x.loc["positive", k]
                + 111*x.loc["absent", k]
            ) / 1301

            assert abs(
                expected - x.loc["unified", k]
            ) < 1e-8

    print(
        met[
            ["model", "group", "mae", "rmse", "zero"]
        ].to_string(index=False),
        flush=True,
    )

    print(
        "A3 recall:",
        int((keep["A3"] & pos).sum()),
        "/1190 | rejection:",
        int((~keep["A3"] & neg).sum()),
        "/111",
        flush=True,
    )

    write_csv(OUT / "unified_metrics.csv", met)

    write_csv(
        OUT / "unified_per_pair_predictions.csv",
        full,
    )

    write_json(
        OUT / "audit.json",
        dict(
            status="PASS",
            count=1301,
            p12_sha256=lock["model_sha256"],
            no_training=True,
            no_threshold_tuning=True,
            blind_claim=False,
            accepted=int(raw.accepted.sum()),
            inputs=fp,
        ),
    )

    print("\nP14-08B + P14-09 COMPLETE", flush=True)
    print("Output:", OUT, flush=True)

if __name__ == "__main__":
    main()
