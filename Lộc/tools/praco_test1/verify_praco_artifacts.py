#!/usr/bin/env python3
"""Audit completed PrACo Test-1 CountGD++ official-adapter A and V7.6 B.

Reads the original ZIP archives without modifying them. Validates sealed file hashes,
protocol linkage, row identity, baseline parity and the deterministic NO/NO gate.
Does NOT claim PrACo Test-2 or scientifically verified absent negatives.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

EXPECTED_ROWS = 34_510
EXPECTED_IMAGES = 1_190
EXPECTED_CLASSES = 29
EXPECTED_PRACO_COMMIT = "438a968aef9d31ded7519733bd84e89baae68896"
EXPECTED_ROSTER_SHA256 = "b73d78d0e21d089cdd875913fe861929904bc7842e600e84ff5a562d439eae39"
EXPECTED_QWEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
EXPECTED_QWEN_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def load_csv(z: ZipFile, name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(z.read(name)), **kwargs)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_data(a_path: Path, b_path: Path, gt_path: Path | None = None) -> dict:
    for p in [a_path, b_path]:
        if not p.is_file():
            raise FileNotFoundError(p)

    with ZipFile(a_path) as za, ZipFile(b_path) as zb:
        assert za.testzip() is None and zb.testzip() is None, "ZIP CRC error"
        req_a = {"official_baseline_pairs.csv", "official_test1_roster.csv",
                 "Inference_Test1_CountGDPlusPlus_test.csv",
                 "official_baseline_protocol.json", "official_sources.json", "run_state.json"}
        req_b = {"praco_official_plus_v76_SEALED.csv", "prediction_sha256.txt",
                 "PrACo_OfficialAdapter_Paired_Test1_metrics.csv", "paired_protocol.json", "run_state.json"}
        assert req_a <= set(za.namelist()), f"A missing {req_a-set(za.namelist())}"
        assert req_b <= set(zb.namelist()), f"B missing {req_b-set(zb.namelist())}"

        a = load_csv(za, "official_baseline_pairs.csv", keep_default_na=False)
        roster = load_csv(za, "official_test1_roster.csv", keep_default_na=False)
        matrix = load_csv(za, "Inference_Test1_CountGDPlusPlus_test.csv", index_col=0)
        b_raw = zb.read("praco_official_plus_v76_SEALED.csv")
        b = pd.read_csv(io.BytesIO(b_raw), keep_default_na=False)
        metrics = load_csv(zb, "PrACo_OfficialAdapter_Paired_Test1_metrics.csv")
        a_proto = json.loads(za.read("official_baseline_protocol.json"))
        a_source = json.loads(za.read("official_sources.json"))
        b_proto = json.loads(zb.read("paired_protocol.json"))
        a_state = json.loads(za.read("run_state.json"))
        b_state = json.loads(zb.read("run_state.json"))
        hash_txt = zb.read("prediction_sha256.txt").decode().strip().split()

        assert len(a) == len(b) == len(roster) == EXPECTED_ROWS
        assert a.pair_id.is_unique and b.pair_id.is_unique and roster.pair_id.is_unique
        assert len(set(roster.image_name)) == EXPECTED_IMAGES
        assert len(set(roster.target)) == EXPECTED_CLASSES
        assert set(a.pair_id) == set(b.pair_id) == set(roster.pair_id)
        assert a.status.eq("ok").all() and b.status.eq("ok").all()
        assert a_state["completed"] == b_state["completed"] == EXPECTED_ROWS
        assert a_state["stage"] == b_state["stage"] == "FULL"

        roster_cols = ["pair_id", "image_name", "target", "positive_class", "exemplar_image"]
        # The original WSL2/Colab roster was sealed with LF line endings.
        # Use the same serialization on Windows without modifying artifact bytes.
        assert digest(roster[roster_cols].to_csv(index=False, lineterminator="\n").encode()) == EXPECTED_ROSTER_SHA256
        assert a_proto["roster_sha256"] == EXPECTED_ROSTER_SHA256
        assert a_source["official_schedule_sha256"] == EXPECTED_ROSTER_SHA256
        assert a_proto["praco_commit"] == a_source["adapter_git_sha"] == EXPECTED_PRACO_COMMIT
        assert a_proto["protocol_sha256"] == a_state["protocol_sha256"]
        assert set(a.protocol_sha256) == {a_proto["protocol_sha256"]}
        assert a_proto["ckpt_sha256"] == a_source["checkpoint_sha256"]
        assert a_proto["synth_json_sha256"] == a_source["synthetic_json_sha256"]
        assert a_source["checkpoint_source"].startswith("third-party HF mirror"), "Update provenance audit for new source"
        a_hashed = {k: v for k, v in a_proto.items() if k != "protocol_sha256"}
        assert digest(json.dumps(a_hashed, sort_keys=True).encode()) == a_proto["protocol_sha256"]

        assert b_proto["official_protocol_sha"] == a_proto["protocol_sha256"]
        assert b_proto["official_checkpoint_sha"] == a_source["checkpoint_sha256"]
        assert b_proto["official_synth_json_sha"] == a_source["synthetic_json_sha256"]
        assert b_proto["roster_sha"] == EXPECTED_ROSTER_SHA256
        assert b_proto["qwen_model"] == EXPECTED_QWEN_MODEL
        assert b_proto["qwen_revision"] == EXPECTED_QWEN_REVISION
        assert b_proto["suppress_policy"] == "baseline>0 and first==NO and second==NO"
        assert b_proto["protocol_sha256"] == b_state["protocol_sha256"]
        assert set(b.protocol_sha256) == {b_proto["protocol_sha256"]}
        assert hash_txt == [digest(b_raw), "praco_official_plus_v76_SEALED.csv"]

        a_ordered = a.sort_values("pair_id").reset_index(drop=True)
        b_ordered = b.sort_values("pair_id").reset_index(drop=True)
        roster_ordered = roster.sort_values("pair_id").reset_index(drop=True)
        for col in roster_cols:
            assert a_ordered[col].astype(str).equals(roster_ordered[col].astype(str)), f"Roster mismatch: {col}"
        for col in ("image_name", "target"):
            assert a_ordered[col].astype(str).equals(b_ordered[col].astype(str)), f"A/B mismatch: {col}"
        ac = pd.to_numeric(a_ordered.baseline_count, errors="raise").to_numpy(dtype=float)
        bc = pd.to_numeric(b_ordered.baseline_count, errors="raise").to_numpy(dtype=float)
        sc = pd.to_numeric(b_ordered.semantic_count, errors="raise").to_numpy(dtype=float)
        assert np.isfinite(ac).all() and np.isfinite(bc).all() and np.isfinite(sc).all()
        assert (ac >= 0).all() and (bc >= 0).all() and (sc >= 0).all()
        assert np.array_equal(ac, bc), "A/B baseline values differ"
        suppressed = (bc > 0) & (b_ordered.first_label.eq("NO").to_numpy()) & (b_ordered.second_label.eq("NO").to_numpy())
        expected = np.where(suppressed, 0.0, bc)
        assert np.array_equal(sc, expected), "Frozen V7.6 NO/NO rule violated"
        assert (sc <= bc).all(), "Semantic count should never increase"

        m = a.pivot(index="image_name", columns="target", values="baseline_count")
        m = m.reindex(index=matrix.index, columns=matrix.columns)
        assert matrix.shape == (EXPECTED_IMAGES, EXPECTED_CLASSES)
        assert np.allclose(m.to_numpy(dtype=float), matrix.to_numpy(dtype=float), atol=1e-8, rtol=0)

        joined = b.merge(a[["pair_id", "positive_class"]], on="pair_id", validate="one_to_one")
        positive = joined.target == joined.positive_class
        mismatch = ~positive
        positive_suppressed = joined.loc[positive & (joined.baseline_count > 0) & (joined.semantic_count == 0)]
        mismatch_suppressed = joined.loc[mismatch & (joined.baseline_count > 0) & (joined.semantic_count == 0)]
        mismatch_remaining = joined.loc[mismatch & (joined.semantic_count > 0)]
        negative_elapsed = int((pd.to_numeric(a.elapsed_sec, errors="coerce") < 0).sum()) if "elapsed_sec" in a else None

        summary = {
            "status": "PASS", "benchmark": "PrACo Test-1 Negative Label (NOT Mosaic Test-2)",
            "n_pairs": len(b), "n_images": EXPECTED_IMAGES, "n_classes": EXPECTED_CLASSES,
            "a_baseline_positive": int((bc > 0).sum()), "zero_no_qwen": int((bc == 0).sum()),
            "new_qwen_inference": int((b.decision_source == "NEW_QWEN_INFERENCE").sum()),
            "reused_qwen": int((b.decision_source == "REUSE_QWEN").sum()),
            "suppressed_total": int(suppressed.sum()),
            "positive_class_suppressed": len(positive_suppressed),
            "class_mismatch_suppressed": len(mismatch_suppressed),
            "class_mismatch_remaining_nonzero": len(mismatch_remaining),
            "negative_elapsed_sec_audit": negative_elapsed,
            "zip_a_sha256": digest(a_path.read_bytes()),
            "zip_b_sha256": digest(b_path.read_bytes()),
            "a_protocol_sha256": a_proto["protocol_sha256"],
            "b_protocol_sha256": b_proto["protocol_sha256"],
            "qwen_revision": b_proto["qwen_revision"],
            "sealed_predictions_sha256": digest(b_raw),
            "note": "Class mismatch is not the same as verified target absence. TEST has already been viewed; do not retune prompts to claim held-out performance.",
            "metrics_from_archive": metrics.to_dict(orient="records"),
        }
        if gt_path is not None:
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
            labels = a.drop_duplicates("image_name").set_index("image_name").positive_class.to_dict()
            p = joined.pivot(index="image_name", columns="target", values="baseline_count")
            s = joined.pivot(index="image_name", columns="target", values="semantic_count")
            measured=[]
            for key, mat in [("baseline_count", p), ("semantic_count", s)]:
                rows=[]
                for im in mat.index:
                    g=float(gt[im]); assert g>0
                    pred=float(mat.at[im,labels[im]])
                    neg=float(mat.loc[im,mat.columns!=labels[im]].mean())
                    rows.append((g,pred,neg))
                arr=np.array(rows,dtype=float)
                g,pred,neg=arr[:,0],arr[:,1],arr[:,2]
                measured.append(dict(Model=key,AvgNP=float(np.mean(pred/g)),AvgNMN=float(np.mean(neg/g)),
                    PCCN=float(100*np.mean(abs(pred-g)<abs(neg-g))),MAE=float(np.mean(abs(pred-g))),
                    RMSE=float(np.sqrt(np.mean((pred-g)**2)))))
            recorded = metrics.set_index("Model")
            for x in measured:
                for key in ("AvgNP", "AvgNMN", "PCCN", "MAE", "RMSE"):
                    assert np.isclose(x[key],float(recorded.at[x["Model"],key]),rtol=0,atol=1e-9), f"Metric mismatch {x['Model']} {key}"
            summary["gt_metrics_recomputed"] = "PASS"
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="A FULL ZIP")
    parser.add_argument("--b", type=Path, required=True, help="B FULL ZIP")
    parser.add_argument("--gt", type=Path, help="Optional official PrACo gt_cnts_FSC147.json for independent metric recalculation")
    parser.add_argument("--output", type=Path, help="Optional local JSON report path")
    args = parser.parse_args()
    s=check_data(args.a,args.b,args.gt)
    out=json.dumps(s,ensure_ascii=False,indent=2)
    print(out)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(out+'\n',encoding='utf-8')


if __name__ == "__main__":
    main()
