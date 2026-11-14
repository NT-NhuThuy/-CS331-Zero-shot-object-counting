import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from .decide import decision, filter_candidates
from .parse_descriptor import compatibility, parse
from .region_compare import margin
from .run import run
from .evaluate import match_count
from .gpu import clip_box


def evidence(cid, full=0.2, context=0.3):
    return {"candidate_id": cid, "caption_status": "VALID", "compatibility": "CONTRADICTS",
            "views": {v: {"descriptor_cosine": score, "query_cosine": 0.1, "alpha_pixels": 10}
                      for v, score in (("full", full), ("context", context))}}


class CoreTest(unittest.TestCase):
    def test_strict_margin_and_unknown(self):
        self.assertEqual(decision(evidence("a", 0.2, 0.3), 0.09)[0], "REMOVE")
        self.assertEqual(decision(evidence("a", 0.2, 0.3), 0.1)[0], "KEEP_UNVERIFIED")
        bad = evidence("a")
        bad["views"]["full"]["descriptor_cosine"] = float("nan")
        self.assertEqual(decision(bad, 0)[0], "KEEP_UNVERIFIED")
        bad = evidence("a")
        bad["caption_status"] = "UNKNOWN"
        self.assertEqual(decision(bad, 0)[0], "KEEP_UNVERIFIED")

    def test_remove_only_binding(self):
        base = [{"candidate_id": "a", "box": [0, 0, 1, 1]}, {"candidate_id": "b", "box": [2, 2, 3, 3]}]
        ea, eb = evidence("a"), evidence("b")
        eb["compatibility"] = "UNKNOWN"
        kept, ledger = filter_candidates(base, {"a": ea, "b": eb}, 0)
        self.assertEqual(kept, base[1:])
        self.assertEqual(len(ledger), len(base))
        with self.assertRaises(ValueError):
            filter_candidates(base, {"a": ea}, 0)

    def test_parser_and_compatibility(self):
        self.assertEqual(parse("A dog."), ("dog", "VALID"))
        self.assertEqual(parse("Several dogs."), (None, "UNKNOWN"))
        self.assertEqual(parse("A dog.", truncated=True), (None, "UNKNOWN"))
        self.assertEqual(compatibility("dog", "pet"), "UNKNOWN")
        self.assertEqual(compatibility("dog", "dog"), "COMPATIBLE")
        self.assertEqual(margin(0.6, 0.2), -margin(0.2, 0.6))

    def test_frozen_run_and_noop_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / x for x in ("base.json", "evidence.json", "config.json", "output.json")]
            paths[0].write_text(json.dumps({"items": [{"candidate_id": "a", "box": [1, 2, 3, 4]}]}))
            ev = evidence("a")
            ev["caption_status"] = "UNKNOWN"
            paths[1].write_text(json.dumps({"items": [ev]}))
            hashes = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in zip(("candidates", "evidence"), paths)}
            paths[2].write_text(json.dumps({"tau": 0, "input_sha256": hashes}))
            result = run(*paths)
            self.assertEqual(result["items"], json.loads(paths[0].read_text())["items"])
            self.assertEqual(result["count"], 1)
            self.assertEqual(run(*paths), result)

    def test_maximum_point_matching(self):
        # A greedy first match would count one; augmenting finds two.
        boxes = [[0, 0, 2, 2], [0, 0, 0.5, 0.5]]
        points = [(0.25, 0.25), (1.5, 1.5)]
        self.assertEqual(match_count(boxes, points), 2)
        self.assertEqual(match_count(boxes[:1], points), 1)

    def test_detector_box_clipping(self):
        self.assertEqual(clip_box([-2, -1, 102, 51], 100, 50), [0.0, 0.0, 100.0, 50.0])
        with self.assertRaises(ValueError):
            clip_box([101, 1, 102, 2], 100, 50)


if __name__ == "__main__":
    unittest.main()
