"""Sanity-check representative experimental metrics. Python + pandas."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

v7 = pd.read_csv(ROOT / "results/01_official_test_1301/V7_1301_predictions.csv")
assert v7.group_id.is_unique and len(v7) == 1301
pos = v7[v7.pair_type == "POSITIVE"]
neg = v7[v7.pair_type == "HARD_NEGATIVE"]
assert len(pos) == 1190 and len(neg) == 111
print(f"V7 TEST positive MAE: {abs(pos.gt_count-pos.v7_count).mean():.4f}")
print(f"V7 TEST negative MAE: {abs(neg.gt_count-neg.v7_count).mean():.4f}")
print(f"V7 TEST negative FPR: {(neg.v7_count.gt(0)).mean()*100:.2f}%")

v75 = pd.read_csv(ROOT / "results/04_validation/V7_5_480_per_image.csv")
assert len(v75) == 480 and v75.group_id.is_unique
base_mae = abs(v75.baseline_count-v75.gt_count).mean()
v75_mae = abs(v75.v75_count-v75.gt_count).mean()
print(f"V7.5 480-image baseline MAE: {base_mae:.4f}")
print(f"V7.5 480-image multi-scale MAE: {v75_mae:.4f}")
print(f"Changed images: {(v75.baseline_count != v75.v75_count).sum()}")

fresh = pd.read_csv(ROOT / "data/fresh_negative_160_manual_review.csv")
assert len(fresh) == 160 and fresh.candidate_id.is_unique
print(f"Fresh negative labels: {fresh.presence_label.value_counts().to_dict()}")
