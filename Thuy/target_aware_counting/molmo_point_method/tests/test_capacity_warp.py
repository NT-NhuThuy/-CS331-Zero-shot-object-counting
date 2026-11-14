import numpy as np
from PIL import Image
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

from target_aware_counting.molmo_point_method.capacity_warp import assignment_capacity, build_warp, collision_groups
from target_aware_counting.molmo_point_method.run_capacity_b_cohort import infer_item


class CapacityWarpTests(unittest.TestCase):
    def test_identity_inverse_and_expansion_direction(self):
        metadata = {"subpatch_mapping": [np.arange(64).reshape(8, 8)],
                    "token_pooling": np.arange(64).reshape(64, 1),
                    "image_sizes": [[80, 80]]}
        warp, diagnostic = build_warp(80, 80, [[15, 15], [16, 16]], metadata, bins=8)
        self.assertEqual(diagnostic["assignment_capacity"]["unmatched_count"], 1)
        points = np.array([[0, 0], [15, 15], [16, 16], [40, 42], [80, 80]])
        self.assertTrue(np.allclose(warp.inverse(warp.forward(points)), points))
        self.assertGreater(warp.forward(points)[2, 0]-warp.forward(points)[1, 0], 1.0)
        self.assertGreaterEqual(diagnostic["x_slope_minmax"][0], 0.5-1e-7)
        self.assertLessEqual(diagnostic["x_slope_minmax"][1], 2.0+1e-7)
        self.assertEqual(warp.render(Image.new("RGB", (80, 80))).size, (80, 80))
        identity, diagnostic = build_warp(80, 80, [[15, 15]], metadata, bins=8)
        self.assertEqual(diagnostic["assignment_capacity"]["unmatched_count"], 0)
        self.assertTrue(np.allclose(identity.forward(points), points))
        pixels = np.arange(80*80*3, dtype=np.uint8).reshape(80, 80, 3)
        self.assertTrue(np.array_equal(np.asarray(identity.render(Image.fromarray(pixels))), pixels))
        self.assertEqual(identity.forward([]).shape, (0, 2))
        self.assertEqual(identity.inverse([]).shape, (0, 2))

    def test_collision_pairs_are_unique(self):
        metadata = {"subpatch_mapping": [np.zeros((2, 2), dtype=int)],
                    "token_pooling": np.zeros((1, 1), dtype=int),
                    "image_sizes": [[20, 20]]}
        self.assertEqual(len(collision_groups([[1, 1], [2, 2], [3, 3]], metadata)), 3)
        self.assertEqual(assignment_capacity([[1, 1], [2, 2], [3, 3]], metadata)["matched_count"], 1)

    def test_shared_address_can_still_have_full_assignment(self):
        metadata = {"subpatch_mapping": [np.array([[0, 1]])],
                    "token_pooling": np.array([[0, 1], [0, -1]]),
                    "image_sizes": [[20, 10]]}
        points = [[2, 5], [4, 5]]
        self.assertEqual(len(collision_groups(points, metadata)), 1)
        capacity = assignment_capacity(points, metadata)
        self.assertEqual(capacity["matched_count"], 2)
        self.assertEqual(capacity["unmatched_count"], 0)
        warp, diagnostic = build_warp(20, 10, points, metadata, bins=5)
        self.assertEqual(diagnostic["demand_proposals"], 0)
        self.assertTrue(np.allclose(warp.forward(points), points))

    def test_cohort_policy_decodes_one_image_and_inverts_warp(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.png"
            Image.new("RGB", (20, 20)).save(path)
            item = {"image_path": str(path)}
            audit = {"status": "ok", "capacity_gain": 1, "source_size_wh": [20, 20],
                     "warp_x_edges": [0, 5, 20], "warp_y_edges": [0, 10, 20]}
            fake = {"status": "complete", "count": 1, "points_xy": [[10, 10]],
                    "raw_points": [[0, 0, 10, 10]]}
            with patch("target_aware_counting.molmo_point_method.run_capacity_b_cohort.infer_one",
                       return_value=fake.copy()) as decode:
                result = infer_item(object(), object(), item, audit, 100)
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(result["selection"], "capacity_warp")
            self.assertAlmostEqual(result["points_xy"][0][0], 5)
            self.assertEqual(result["points_xy_warped"], [[10, 10]])
            audit["capacity_gain"] = 0
            with patch("target_aware_counting.molmo_point_method.run_capacity_b_cohort.infer_one",
                       return_value=fake.copy()) as decode:
                result = infer_item(object(), object(), item, audit, 100)
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(result["selection"], "native_identity")
            audit["capacity_gain"] = 1
            empty = {"status": "complete", "count": 0, "points_xy": [], "raw_points": []}
            with patch("target_aware_counting.molmo_point_method.run_capacity_b_cohort.infer_one",
                       return_value=empty) as decode:
                result = infer_item(object(), object(), item, audit, 100)
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(result["count"], 0)
            self.assertEqual(result["points_xy"], [])


if __name__ == "__main__":
    unittest.main()
