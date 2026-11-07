from __future__ import annotations

import unittest

from target_aware_counting.bounded_recovery import (
    RecoveryConfig,
    StrictVerifierConfig,
    apply_empty_recovery,
    apply_strict_verification,
    box_iou,
    select_bounded_candidates,
)


class BoundedRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            {"rank": 1, "box": [0, 0, 10, 10], "native_score": 0.2},
            {"rank": 2, "box": [1, 1, 11, 11], "native_score": 0.1},
            {"rank": 3, "box": [20, 20, 30, 30], "native_score": 0.05},
        ]

    def test_iou(self) -> None:
        self.assertEqual(box_iou([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertEqual(box_iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)

    def test_nonempty_baseline_bypasses_auxiliary_inputs(self) -> None:
        result = apply_empty_recovery([[2, 3, 4, 5]])
        self.assertEqual(result["mode"], "baseline")
        self.assertEqual(result["count"], len(result["regions"]))
        self.assertEqual(result["regions"][0]["box"], [2.0, 3.0, 4.0, 5.0])

    def test_owl_has_priority_and_nms_removes_duplicate(self) -> None:
        owl = [
            {"score": 0.7, "box": [0, 0, 10, 10]},
            {"score": 0.6, "box": [20, 20, 30, 30]},
        ]
        selected = select_bounded_candidates(self.candidates, owl, {1: 0.9, 2: 0.8, 3: 0.7})
        self.assertEqual([x["rank"] for x in selected], [1, 3])
        self.assertTrue(all(x["reason"] == "owl_supported" for x in selected))

    def test_siglip_fallback_is_bounded_to_one(self) -> None:
        selected = select_bounded_candidates(self.candidates, [], {1: 0.06, 2: 0.8, 3: 0.7})
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["rank"], 2)
        self.assertEqual(selected[0]["reason"], "siglip_single_fallback")

    def test_abstains_below_floors(self) -> None:
        result = apply_empty_recovery([], self.candidates, [], {1: 0.01, 2: 0.02, 3: 0.049})
        self.assertEqual(result["mode"], "abstain")
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["count"], len(result["regions"]))

    def test_threshold_boundaries_are_inclusive(self) -> None:
        selected = select_bounded_candidates(
            [{"rank": 1, "box": [0, 0, 10, 10]}],
            [{"score": 0.1, "box": [0, 0, 5, 10]}],
            {1: 0.0},
        )
        self.assertEqual(len(selected), 1)

    def test_invalid_schema_fails_loudly(self) -> None:
        with self.assertRaises(ValueError):
            select_bounded_candidates(
                [{"rank": 1, "box": [5, 0, 1, 2]}], [], {1: 0.8}
            )
        with self.assertRaises(ValueError):
            RecoveryConfig(max_siglip_fallback=2)

    def test_strict_verifier_preserves_baseline(self) -> None:
        decision = apply_empty_recovery([[2, 3, 4, 5]])
        strict = apply_strict_verification(decision)
        self.assertEqual(strict["mode"], "baseline")
        self.assertEqual(strict["regions"], decision["regions"])
        self.assertEqual(strict["count"], 1)

    def test_strict_verifier_filters_then_attaches_mask(self) -> None:
        decision = apply_empty_recovery(
            [], self.candidates, [], {1: 0.06, 2: 0.8, 3: 0.7}
        )
        self.assertEqual(decision["regions"][0]["rank"], 2)
        strict = apply_strict_verification(
            decision,
            {2: 0.004},
            {2: {"mask_bbox_xyxy": [2, 2, 9, 9], "mask_path": "mask.npz", "mask_area": 20}},
        )
        self.assertEqual(strict["mode"], "strict_recovery")
        self.assertEqual(strict["count"], len(strict["regions"]))
        self.assertEqual(strict["regions"][0]["box"], [2.0, 2.0, 9.0, 9.0])
        self.assertEqual(strict["regions"][0]["proposal_box"], [1.0, 1.0, 11.0, 11.0])
        self.assertEqual(strict["regions"][0]["region"]["type"], "sam2_mask")

    def test_strict_verifier_abstains_below_fixed_floor(self) -> None:
        decision = apply_empty_recovery(
            [], self.candidates, [], {1: 0.06, 2: 0.8, 3: 0.7}
        )
        strict = apply_strict_verification(decision, {2: 0.0029})
        self.assertEqual(strict["mode"], "abstain")
        self.assertEqual(strict["count"], 0)
        with self.assertRaises(ValueError):
            StrictVerifierConfig(siglip_tight_score_floor=float("nan"))


if __name__ == "__main__":
    unittest.main()
