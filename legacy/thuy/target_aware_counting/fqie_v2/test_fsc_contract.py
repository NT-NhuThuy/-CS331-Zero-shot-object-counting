"""End-to-end CPU contract check on frozen FSC candidates with UNKNOWN evidence."""
import json
import tempfile
import unittest
from pathlib import Path

from .evaluate import evaluate
from .run import run, sha

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "target_aware_counting/output/fqie_v2_fsc_dev_20261104_v1/candidates.json"


class FscContractTest(unittest.TestCase):
    def test_complete_noop_is_exact_a4_parity(self):
        if not MANIFEST.exists():
            self.skipTest("frozen FSC manifest has not been prepared")
        data = json.loads(MANIFEST.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            freeze = root / "freeze.json"
            evidence = root / "evidence.json"
            config = root / "config.json"
            prediction = root / "prediction.json"
            freeze.write_text(json.dumps({"status": "FROZEN_BEFORE_LABEL_JOIN", "manifest_sha256": sha(MANIFEST)}))
            evidence.write_text(json.dumps({"status": "FROZEN_GT_FREE_EVIDENCE",
                "gpu_freeze_sha256": sha(freeze), "manifest_sha256": sha(MANIFEST),
                "items": [{"candidate_id": x["candidate_id"], "caption_status": "UNKNOWN",
                           "compatibility": "UNKNOWN", "views": {}} for x in data["items"]]}))
            config.write_text(json.dumps({"tau": 0.0, "input_sha256":
                {"candidates": sha(MANIFEST), "evidence": sha(evidence)}}))
            pred = run(MANIFEST, evidence, config, prediction)
            self.assertEqual(pred["count"], len(data["items"]))
            result = evaluate(MANIFEST, freeze, evidence, prediction, root / "evaluation")
            self.assertEqual(result["metrics"]["a4"], result["metrics"]["final"])
            self.assertEqual(result["denominators"]["candidates"], 12173)


if __name__ == "__main__":
    unittest.main()
