"""CPU evaluator regression using synthetic local decisions equal to full cache.

All synthetic artifacts are temporary. This verifies historical controls and
is not a measurement of local-view model quality.
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from target_aware_counting.fqie_dualview import DEST, atomic_once, evaluate, freeze, read, sha


class DualViewContractTest(unittest.TestCase):
    def test_evaluator_reproduces_cached_controls(self):
        if not (DEST / "manifest.json").exists():
            self.skipTest("prepare development manifest first")
        manifest = read(DEST / "manifest.json")
        with tempfile.TemporaryDirectory(prefix="fqie-control-test-") as tmp:
            destination = Path(tmp)
            atomic_once(destination / "manifest.json", manifest)
            fingerprint = sha(destination / "manifest.json")
            rows = {x["key"]: x for x in manifest["items"]}
            # Missing output must fail instead of being interpreted as count zero.
            with self.assertRaises(RuntimeError): freeze(destination)
            for task in manifest["tasks"]:
                row = rows[task["key"]]
                record = {"task_id": task["task_id"], "manifest_sha256": fingerprint}
                ep = destination / "encodings" / (task["task_id"] + ".json")
                atomic_once(ep, record)
                atomic_once(destination / "responses" / (task["task_id"] + ".json"),
                            {**record, "query": row["query"], "encoding_sha256": sha(ep),
                             "raw_response": row["full_decision"], "output_tokens": 2,
                             "decision": row["full_decision"]})
            with contextlib.redirect_stdout(io.StringIO()):
                freeze(destination)
                evaluate(destination)
            result = read(destination / "evaluation.json")
            self.assertEqual(result["rules"]["no_op"]["lvis_tp50"], 163)
            self.assertEqual(result["rules"]["full_yes_legacy"]["lvis_tp50"], 157)
            self.assertEqual(result["rules"]["dual_no_remove"], result["rules"]["full_not_no"])
            self.assertEqual(result["rules"]["full_yes_legacy"]["fsc_top5_absent_boxes"], 12)
            self.assertEqual(result["rules"]["full_yes_legacy"]["fsc_top5_active_queries"], 4)
            self.assertEqual(result["decision"], "NOT_PROMOTED")


if __name__ == "__main__":
    unittest.main()
