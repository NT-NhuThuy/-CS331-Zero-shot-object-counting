"""P06 synthetic smoke: no GT, no calibration, no GPU."""

from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy
import numpy as np

from countlab.integrations.p06_diagnostic import (
    extract_diagnostic_features,
)


def main():
    manifest = {
        "schema": "countlab_candidate_pool_v1",
        "image_id": "synthetic_image_001",
        "query_id": "synthetic_query_001",
        "query_text": "chair",
        "baseline_count": 1,
        "baseline": [
            {
                "candidate_id": "b1",
                "view_id": "full",
                "box_xyxy": [0, 0, 10, 20],
                "confidence": 0.9,
                "evidence": {
                    "semantic": "UNKNOWN",
                    "whole_instance": "UNKNOWN",
                },
            }
        ],
        "rescue": [
            {
                "candidate_id": "r1",
                "view_id": "g2_r0_c0",
                "box_xyxy": [30, 10, 38, 18],
                "confidence": 0.2,
                "evidence": {
                    "semantic": "UNKNOWN",
                    "whole_instance": "UNKNOWN",
                },
            }
        ],
        "identity_edges": [],
        "metadata": {
            "protocol_id": "P06_SYNTHETIC_SMOKE_V1"
        },
    }

    mapping = {
        "b1": {
            "node_graph_id": 1,
            "raw_query_index": 4,
        },
        "r1": {
            "node_graph_id": 2,
            "raw_query_index": 5,
        },
    }

    original = deepcopy(manifest)

    with TemporaryDirectory() as tmp:
        npz = Path(tmp) / "features.npz"

        np.savez(
            npz,
            node_graph_id=np.array([1, 2]),
            raw_query_index=np.array([4, 5]),
            v4_contribution=np.array([0.12, -0.25]),
            a3_score=np.array([0.8, 0.2]),
        )

        result = extract_diagnostic_features(
            manifest,
            npz,
            mapping,
            exposure_status="UNKNOWN",
        )

        replay = extract_diagnostic_features(
            manifest,
            npz,
            mapping,
            exposure_status="UNKNOWN",
        )

    assert result == replay
    assert manifest == original
    assert result["n_candidates"] == 2
    assert result["risk_hint_generated"] is False
    assert result["calibration_used"] is False
    assert result["g0_decision"] == "HOLD"

    print("=== P06 SYNTHETIC SMOKE ===")

    for row in result["candidates"]:
        print(
            row["candidate_id"],
            "| role:", row["role"],
            "| V4:", row["v4_contribution"],
            "| A3:", row["a3_score"],
        )

    print("Candidates:", result["n_candidates"])
    print("Pool SHA256:", result["candidate_pool_sha256"])
    print("Feature SHA256:", result["feature_npz_sha256"])
    print("Mapping SHA256:", result["candidate_map_sha256"])
    print("Risk generated:", result["risk_hint_generated"])
    print("Calibration:", result["calibration_used"])
    print("G0:", result["g0_decision"])
    print("P06 SYNTHETIC SMOKE: PASS")


if __name__ == "__main__":
    main()
