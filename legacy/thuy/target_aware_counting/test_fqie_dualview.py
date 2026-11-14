import math
from pathlib import Path
import tempfile
import unittest

from target_aware_counting.fqie_dualview import atomic_once, geometry, keep, paired_bootstrap, strict_parse


class DualViewTest(unittest.TestCase):
    def test_translation_preserves_original_region(self):
        original = [10., 20., 30., 40.]
        clipped, crop, local = geometry(original, 100, 100)
        self.assertEqual(crop, [5, 15, 35, 45])
        self.assertEqual(local, [5, 5, 25, 25])
        self.assertEqual([local[0]+crop[0], local[1]+crop[1], local[2]+crop[0], local[3]+crop[1]], clipped)
        self.assertEqual(original, [10., 20., 30., 40.])

    def test_boundary_and_invalid(self):
        clipped, bounds, local = geometry([-1, 5, 101, 20], 100, 50)
        self.assertEqual(clipped, [0., 5, 100., 20])
        self.assertGreaterEqual(min(local), 0)
        for box in ([math.nan, 0, 1, 1], [101, 0, 102, 1], [5, 5, 2, 2]):
            with self.assertRaises(ValueError): geometry(box, 100, 50)

    def test_decision_abstains_and_preserves_full_keeps(self):
        for a in ("yes", "no", "uncertain", "UNKNOWN"):
            for b in ("yes", "no", "uncertain", "UNKNOWN"):
                self.assertEqual(keep(a, b), not (a == b == "no"))
                if a != "no": self.assertTrue(keep(a, b))

    def test_strict_parser_ceiling(self):
        self.assertEqual(strict_parse("<think></think>no", 3), "no")
        for raw, n in (("no because it is small", 5), ("no", 8), ("<think>no", 3), ("yes or no", 4)):
            self.assertEqual(strict_parse(raw, n), "UNKNOWN")

    def test_write_once_and_noop_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"record.json"
            atomic_once(path, {"a": 1})
            atomic_once(path, {"a": 1})
            with self.assertRaises(RuntimeError): atomic_once(path, {"a": 2})
        rows = [{"query_id": str(i), "source_image_id": str(i//2), "status": "present", "gt_count": 2, "count": 3, "tp50": 2}
                for i in range(6)]
        self.assertTrue(all(v == [0, 0] for v in paired_bootstrap(rows, rows, 30)["delta_95ci"].values()))


if __name__ == "__main__":
    unittest.main()
