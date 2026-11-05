#!/usr/bin/env python3

from pathlib import Path
import argparse, contextlib, hashlib, io, json, os, sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

R = Path(__file__).resolve().parents[1]

P = R / "generated/fsc147_presence_v4/test_pairs_1301.csv"
U = R / "outputs/joint_semantic_counting_v1/p14_06_unified_matched_v1/unified_predictions.csv"
C = R / "outputs/fsc147_presence_v4/official_test_v2/test_clip_scores.csv"
PT = R / "outputs/joint_semantic_counting_v1/s1_verifier_a3_clip_v1/s1_a3_seed42/best.pt"

OUT = R / "outputs/joint_semantic_counting_v1/p14_07b_a3_fullraw_fsc_unified_v1"

POS = R / "data/fscd147/FSC147_384_V2/annotations/annotation_FSC147_384.json"
NEG = R / "generated/fsc147_presence_v4/annotation_absent111_empty.json"

CK_SHA = "683e6009a0dad1f1f5bf9d1d1bb1b61365d82b1e0d844bf47de2e9ca17b62dc5"
THR = 0.9910586476325989


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def score_metric(y, p):
    e = np.asarray(p, float) - np.asarray(y, float)
    return {
        "mae": float(np.abs(e).mean()),
        "mse": float((e * e).mean()),
        "rmse": float(np.sqrt((e * e).mean())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--end", type=int, default=1301)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    assert 1 <= args.end <= 1301
    assert args.end == 1301 or args.end % 25 == 0

    pairs, uni, clip = (
        pd.read_csv(p).reset_index(drop=True)
        for p in (P, U, C)
    )

    assert len(pairs) == len(uni) == len(clip) == 1301
    assert pairs.pair_id.is_unique
    assert uni.pair_id.is_unique
    assert clip.pair_id.is_unique

    assert pairs.pair_id.tolist() == uni.pair_id.tolist()
    assert pairs.presence_label.value_counts().to_dict() == {
        1: 1190, 0: 111
    }

    assert np.array_equal(
        pairs.presence_label.to_numpy(int),
        uni.presence_label.to_numpy(int),
    )

    assert np.allclose(
        pairs.count_target.to_numpy(float),
        uni.gt_count.to_numpy(float),
        rtol=0,
        atol=0,
    )

    positive_exemplars = pairs.loc[
        pairs.presence_label == 1, "num_exemplars"
    ]

    print(
        "Positive manifest exemplar counts:",
        positive_exemplars.value_counts().sort_index().to_dict(),
        flush=True,
    )

    assert positive_exemplars.between(3, 5).all(), (
        "Unexpected positive exemplar metadata"
    )

    assert (
        pairs.loc[pairs.presence_label == 0, "num_exemplars"] == 0
    ).all()

    clip = pairs[["pair_id", "file_name", "query"]].merge(
        clip[["pair_id", "file_name", "query", "clip_cosine"]],
        on=["pair_id", "file_name", "query"],
        validate="one_to_one",
        sort=False,
    )

    assert np.isfinite(clip.clip_cosine).all()
    assert len(clip) == 1301
    assert digest(PT) == CK_SHA, "A3 checkpoint hash mismatch"

    anns = [
        json.loads(p.read_text())
        for p in (POS, NEG)
    ]

    for row in pairs.itertuples(index=False):
        assert (R / row.image_path).is_file(), row.image_path

        ann = anns[
            0 if row.presence_label == 1 else 1
        ][row.file_name]

        if row.presence_label == 0:
            assert not ann.get("points", [])
            assert not ann.get("box_examples_coordinates", [])
            assert not ann.get("box_examples_path", [])

    print(
        "P14-07B PREFLIGHT PASS | "
        "1190 positive + 111 absent | 3/0 exemplars",
        flush=True,
    )
    print("A3 frozen threshold:", THR, flush=True)

    if args.preflight:
        return

    sys.path.insert(0, str(R / "scripts"))

    import audit_decoder_query_embeddings as audit
    from train_s1_semantic_verifier_a3_clip_v1 import A3SemanticVerifier

    ck = torch.load(
        PT, map_location="cpu", weights_only=True
    )

    assert ck["mode"] == "A3"
    assert float(ck["clip_train_std"]) > 0

    a3 = A3SemanticVerifier().cpu().eval()
    a3.load_state_dict(ck["model_state"], strict=True)

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    cz = (
        (
            clip.clip_cosine.to_numpy(np.float32)
            - float(ck["clip_train_mean"])
        )
        / float(ck["clip_train_std"])
    ).astype(np.float32)

    provenance = {
        "pairs": digest(P),
        "unified": digest(U),
        "clip": digest(C),
        "checkpoint": CK_SHA,
        "threshold": THR,
    }

    partdir = OUT / "parts"
    partdir.mkdir(parents=True, exist_ok=True)

    chunks = [
        (s, min(s + 25, args.end))
        for s in range(0, args.end, 25)
    ]

    def paths(s, e):
        stem = f"part_{s:04d}_{e-1:04d}"
        return (
            partdir / (stem + ".csv"),
            partdir / (stem + "_nodes.npz"),
            partdir / (stem + ".json"),
        )

    todo = []

    for s, e in chunks:
        csv, npz, js = paths(s, e)
        exists = [x.exists() for x in (csv, npz, js)]

        if all(exists):
            assert args.resume, "Part exists; use --resume"
            assert json.loads(js.read_text()) == provenance, \
                "Resume provenance mismatch"

            tab = pd.read_csv(csv)

            assert tab.pair_id.tolist() == (
                pairs.pair_id.iloc[s:e].tolist()
            )

            with np.load(npz, allow_pickle=False) as z:
                assert (
                    len(z["pair_index"])
                    == len(z["raw_query_index"])
                    == len(z["a3_score"])
                )

                assert np.array_equal(
                    np.bincount(
                        z["pair_index"] - s,
                        minlength=e - s,
                    ),
                    tab.native_count.to_numpy(int),
                )

            print("[RESUME]", csv.name, flush=True)

        elif any(exists):
            raise RuntimeError(
                f"Incomplete part {s}:{e}; inspect manually"
            )
        else:
            todo.append((s, e))

    if todo:
        app = audit.countgd_app
        args_model = app.get_args_parser().parse_args([])
        device = app.get_device()

        oldcwd = os.getcwd()

        try:
            os.chdir(R / "third_party/CountGDPlusPlus")
            args_model.pretrain_model_path = (
                "checkpoints/countgd_plusplus.pth"
            )

            model, transform = (
                app.build_model_and_transforms(args_model)
            )
        finally:
            os.chdir(oldcwd)

        model = model.to(device).eval()

        for param in model.parameters():
            param.requires_grad_(False)

        hook = model.class_embed[-1].register_forward_pre_hook(
            audit.decoder_hook
        )

        original = app.get_boxes_from_prediction
        app.get_boxes_from_prediction = audit.capture_raw_prediction

        try:
            for s, e in todo:
                table = []
                node_pairs = []
                node_indices = []
                node_scores = []

                for i in range(s, e):
                    row = pairs.iloc[i]
                    ispos = int(row["presence_label"]) == 1

                    ann = anns[
                        0 if ispos else 1
                    ][str(row["file_name"])]

                    with Image.open(
                        R / str(row["image_path"])
                    ) as im:
                        image = im.convert("RGB")

                    prompts = audit.make_positive_prompts(
                        image, ann
                    )

                    audit.RAW_CAPTURE.clear()
                    audit.HS_CAPTURE.clear()

                    with (
                        torch.no_grad(),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        app.predict(
                            model,
                            transform,
                            image,
                            str(row["query"]),
                            prompts,
                            [],
                            [],
                            device,
                        )

                    assert audit.RAW_CAPTURE
                    assert audit.HS_CAPTURE

                    ids = audit.RAW_CAPTURE["input_ids"].numpy()
                    logits = audit.RAW_CAPTURE["logits"].numpy()
                    hs = audit.HS_CAPTURE["hs"].numpy()

                    if hs.ndim == 3:
                        hs = hs[0]

                    assert hs.shape == (900, 256)
                    assert logits.shape[0] == 900

                    periods = np.flatnonzero(ids == 1012)
                    assert len(periods) > 0

                    cut = int(periods[0])

                    pos = logits[:, :cut + 1].max(axis=1)
                    rest = logits[:, cut + 1:]

                    ref = (
                        rest.max(axis=1)
                        if rest.shape[1]
                        else np.zeros_like(pos)
                    )

                    retained = np.flatnonzero(
                        (pos > 0.23) & (pos > ref)
                    ).astype(np.int32)

                    native = int(len(retained))
                    expected = int(uni.iloc[i]["CountGD"])

                    assert native == expected, (
                        f"CountGD parity mismatch {row['pair_id']}: "
                        f"{native} vs {expected}"
                    )

                    # Follow the frozen LVIS A3 float16-HS protocol.
                    h = torch.from_numpy(
                        hs.astype(np.float16).astype(np.float32)
                    )

                    c = torch.full(
                        (900,),
                        float(cz[i]),
                        dtype=torch.float32,
                    )

                    with torch.inference_mode():
                        sc = torch.sigmoid(
                            a3(h, c)
                        ).reshape(-1).numpy().astype(np.float32)

                    assert np.isfinite(sc).all()
                    assert len(sc) == 900

                    pair_score = float(sc.max())

                    table.append({
                        "pair_index": i,
                        "pair_id": row["pair_id"],
                        "presence_label": int(row["presence_label"]),
                        "native_count": native,
                        "a3_pair_score": pair_score,
                        "a3_present": int(pair_score >= THR),
                    })

                    node_pairs.append(
                        np.full(native, i, dtype=np.int32)
                    )
                    node_indices.append(retained)
                    node_scores.append(sc[retained])

                    print(
                        f"[{i+1}/1301] {row['pair_id']} "
                        f"native={native} "
                        f"A3={pair_score:.6f} "
                        f"present={int(pair_score >= THR)}",
                        flush=True,
                    )

                csv, npz, js = paths(s, e)

                assert not any(
                    x.exists() for x in (csv, npz, js)
                )

                pd.DataFrame(table).to_csv(
                    csv, index=False
                )

                np.savez_compressed(
                    npz,
                    pair_index=np.concatenate(node_pairs),
                    raw_query_index=np.concatenate(node_indices),
                    a3_score=np.concatenate(node_scores),
                )

                js.write_text(
                    json.dumps(provenance, indent=2)
                )

                print("[SAVED]", csv.name, flush=True)

        finally:
            hook.remove()
            app.get_boxes_from_prediction = original

    full = pd.concat(
        [
            pd.read_csv(paths(s, e)[0])
            for s, e in chunks
        ],
        ignore_index=True,
    )

    assert full.pair_id.tolist() == (
        pairs.pair_id.iloc[:args.end].tolist()
    )

    if args.end < 1301:
        print(
            f"P14-07B SMOKE PASS ({len(full)}/1301). "
            "Run --resume for full.",
            flush=True,
        )
        return

    gt = uni.gt_count.to_numpy(float)
    label = full.presence_label.to_numpy(int)
    pred = full.a3_present.to_numpy(int)
    scores = full.a3_pair_score.to_numpy(float)

    positive = label == 1
    absent = label == 0

    tp = int(((pred == 1) & positive).sum())
    tn = int(((pred == 0) & absent).sum())

    summary = {
        "pairs": 1301,
        "positive": 1190,
        "absent": 111,
        "tp": tp,
        "fn": 1190 - tp,
        "tn": tn,
        "fp": 111 - tn,
        "positive_recall": tp / 1190,
        "absent_rejection": tn / 111,
        "balanced_accuracy": (
            tp / 1190 + tn / 111
        ) / 2,
        "auroc": float(
            roc_auc_score(label, scores)
        ),
        "ap": float(
            average_precision_score(label, scores)
        ),
        "frozen_threshold": THR,
        "protocol": (
            "mixed-exemplar posthoc diagnostic, not blind"
        ),
    }

    for modelcol in ("CountGD", "V4 seed42"):
        gated = np.where(
            pred == 1,
            uni[modelcol].to_numpy(float),
            0.0,
        )

        full[modelcol + " + A3"] = gated

        for group, mask in [
            ("positive", positive),
            ("absent", absent),
            ("unified", np.ones(1301, dtype=bool)),
        ]:
            summary[modelcol + " + A3 " + group] = (
                score_metric(gt[mask], gated[mask])
            )

    csv = OUT / "a3_unified_1301_predictions.csv"
    js = OUT / "a3_unified_1301_metrics.json"

    assert not csv.exists() and not js.exists(), \
        "Final outputs already exist"

    full.to_csv(csv, index=False)
    js.write_text(json.dumps(summary, indent=2))

    print(
        "P14-07B COMPLETE",
        json.dumps(summary, indent=2),
        flush=True,
    )

    print("Predictions:", csv, flush=True)
    print("Metrics:", js, flush=True)


if __name__ == "__main__":
    main()
