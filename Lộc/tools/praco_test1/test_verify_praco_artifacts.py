"""Unit tests for the deterministic suppression policy and bundle-based validation."""
from pathlib import Path
import os
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_praco_artifacts import check_data


def test_a_and_b_local_archives():
    root=Path(__file__).resolve().parents[2]/"results"/"05_praco_official_adapter"/"archives"
    a=root/"CS331_PrACo_OFFICIAL_ADAPTER_A_FULL_FULL.zip"
    b=root/"CS331_PrACo_OFFICIAL_ADAPTER_B_FULL.zip"
    assert a.is_file() and b.is_file()
    s=check_data(a,b)
    assert s["status"]=="PASS"
    assert s["n_pairs"]==34510
    assert s["suppressed_total"]==3060
    assert s["positive_class_suppressed"]==19
    assert s["class_mismatch_remaining_nonzero"]==72
    assert s["new_qwen_inference"]+s["reused_qwen"]==4303
    assert np.isfinite(s["metrics_from_archive"][0]["MAE"])


def test_roster_hash_with_windows_line_endings(monkeypatch):
    """The sealed WSL2 roster must validate with Windows CSV defaults too."""
    monkeypatch.setattr(os, "linesep", "\r\n")
    root = Path(__file__).resolve().parents[2] / "results" / "05_praco_official_adapter" / "archives"
    summary = check_data(
        root / "CS331_PrACo_OFFICIAL_ADAPTER_A_FULL_FULL.zip",
        root / "CS331_PrACo_OFFICIAL_ADAPTER_B_FULL.zip",
    )
    assert summary["status"] == "PASS"
    assert summary["n_pairs"] == 34510
