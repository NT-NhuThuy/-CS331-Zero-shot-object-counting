"""Read-only, protocol-aware matched counting evaluator (stdlib only).

Usage: countlab-audit --csv predictions.csv --id pair_id --cluster source_image_id
       --gt gt_count --baseline baseline_count --method corrected_count

CSV should be an evaluation export, never an inference input. Bootstrap clusters
represent source images; no claim that resampling proves split independence.
"""
import argparse
from collections import defaultdict
import csv
from itertools import repeat
import json
import math
import random
from pathlib import Path


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / (2 ** n)) if n else 1.0


def paired_audit(rows: list[dict], *, bootstrap: int = 2000, seed: int = 42) -> dict:
    if not rows:
        raise ValueError("no pairs")
    seen = set()
    data = []
    clusters = defaultdict(list)
    for x in rows:
        key = str(x["id"])
        if key in seen:
            raise ValueError(f"duplicate pair id: {key}")
        seen.add(key)
        cluster = str(x["source_image_id"])
        gt, base, pred = (float(x[k]) for k in ("gt", "baseline", "method"))
        if not all(math.isfinite(v) and v >= 0 for v in (gt, base, pred)):
            raise ValueError("GT and counts must be finite non-negative")
        row = {"id": key, "cluster": cluster, "baseline_ae": abs(base - gt), "method_ae": abs(pred - gt), "status": str(x.get("status", "unknown"))}
        data.append(row)
        clusters[cluster].append(row)
    n = len(data)
    bd = sum(d["baseline_ae"] for d in data) / n
    md = sum(d["method_ae"] for d in data) / n
    wins = sum(d["method_ae"] < d["baseline_ae"] for d in data)
    losses = sum(d["method_ae"] > d["baseline_ae"] for d in data)
    rng = random.Random(seed)
    keys = sorted(clusters)
    arr = []
    for _ in range(bootstrap):
        drawn = [x for k in rng.choices(keys, k=len(keys)) for x in clusters[k]]
        arr.append(sum(x["method_ae"] - x["baseline_ae"] for x in drawn) / len(drawn))
    arr.sort()
    ci = [arr[max(0, int(bootstrap * 0.025))], arr[min(bootstrap - 1, int(bootstrap * 0.975))]] if bootstrap else None
    by_status = {}
    for status in sorted(set(x["status"] for x in data)):
        ys = [x for x in data if x["status"] == status]
        by_status[status] = {"n": len(ys), "baseline_mae": sum(x["baseline_ae"] for x in ys) / len(ys),
                             "method_mae": sum(x["method_ae"] for x in ys) / len(ys)}
    return {"n_pairs": n, "n_source_clusters": len(keys), "baseline_mae": bd, "method_mae": md,
            "paired_delta_mae": md-bd, "relative_mae_reduction_pct": ((bd-md)/bd*100 if bd else None),
            "wins": wins, "losses": losses, "ties": n-wins-losses, "two_sided_sign_test_p": sign_test(wins, losses),
            "source_cluster_bootstrap_ci95": ci, "bootstrap_replicates": bootstrap,
            "seed": seed, "strata": by_status,
            "caveat": "Exploratory until lineage, locked split, GT access, and independence are externally verified."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    for field, default in [("id", "pair_id"), ("cluster", "source_image_id"), ("gt", "gt_count"), ("baseline", "baseline_count"), ("method", "method_count"), ("status", None)]:
        p.add_argument("--"+field, default=default)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    with args.csv.open(newline="", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))
    rows=[]
    for r in records:
        rows.append({"id": r[args.id], "source_image_id": r[args.cluster], "gt": r[args.gt],
                     "baseline": r[args.baseline], "method": r[args.method], "status": r[args.status] if args.status else "unknown"})
    result = paired_audit(rows, bootstrap=args.bootstrap, seed=args.seed)
    payload=json.dumps(result, indent=2, ensure_ascii=False)+"\n"
    if args.out:
        with args.out.open("x", encoding="utf-8") as f:
            f.write(payload)
    print(payload)


if __name__ == "__main__":
    main()
