import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from target_aware_counting.reference_correspondence.config import file_sha256,stable_hash
from target_aware_counting.reference_correspondence.hypothesis_builder import Hypothesis
from target_aware_counting.reference_correspondence.evaluate import matched_count,evaluate
from target_aware_counting.reference_correspondence.predict import predict_item,freeze_predictions
from target_aware_counting.reference_correspondence.reference_bank import build_reference_bank
from target_aware_counting.reference_correspondence.spatial_features import (
    Atom,save_atom_cache,load_atom_cache)


def write_jsonl(path,records):
    path.write_text("".join(json.dumps(r)+"\n" for r in records))


class PipelineContractTests(unittest.TestCase):
    def test_one_to_one_grounding_and_no_label_access_without_freeze(self):
        self.assertEqual(matched_count([(0,0,10,10),(0,0,10,10)],[(1,1)],point=True),1)
        self.assertEqual(matched_count([(0,0,10,10)],[(0,0,10,10)],point=False),1)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with patch("target_aware_counting.reference_correspondence.evaluate.load_labels") as labels:
                with self.assertRaises(FileNotFoundError):
                    evaluate("lvis_dev300",folder/"manifest",folder/"predictions",
                             folder/"missing_freeze",folder/"comparison",folder/"summary")
                labels.assert_not_called()

    def test_reference_duplicate_and_missing_roi_stay_unavailable(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            eval_image = folder/"eval.png"
            reference_image = folder/"reference.png"
            Image.fromarray(np.tile(np.arange(16,dtype=np.uint8)[None,:,None],(16,1,3))).save(eval_image)
            Image.fromarray(np.tile(np.arange(16,dtype=np.uint8)[:,None,None],(1,16,3))).save(reference_image)
            manifest = folder/"eval.jsonl"
            write_jsonl(manifest,[{"item_id":"e","image_path":str(eval_image),"query":"car ."}])
            query_bank = folder/"queries.jsonl"
            write_jsonl(query_bank,[{"canonical_query":"car .","aliases":[],"competitors":[]}])
            provider = folder/"provider.jsonl"
            base = {"canonical_query":"car .","semantic_role":"target",
                    "retrieval_term":"car","source_url":"https://example.org/car",
                    "retrieved_at":"2026-10-29","license":"unknown","provider":"archived_local"}
            write_jsonl(provider,[{**base,"retrieval_rank":1,"image_path":str(eval_image)},
                                  {**base,"retrieval_rank":2,"image_path":str(reference_image)}])
            config = folder/"config.json"
            config.write_text(json.dumps({"provider_manifest":str(provider),
                "provider_manifest_sha256":file_sha256(provider),
                "evaluation_manifests":[str(manifest)],"max_rank_per_term":2,
                "min_valid_target":1,"min_valid_competitor":0,
                "near_duplicate_hamming_max":0,"broken_image_policy":"invalidate"}))
            result = build_reference_bank(query_bank,config,folder/"references.jsonl")
            rows = [json.loads(s) for s in (folder/"references.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["status"],"evaluation_exact_duplicate")
            self.assertEqual(rows[1]["status"],"roi_pending")
            self.assertFalse(result["coverage"]["car ."]["reference_available"])

    def test_atom_cache_uses_image_and_config_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"atoms.npz"
            config = {"view_policy":"full","patch_size":14}
            atoms = [Atom("full:0:0",(0,0,14,14),(7,7),"full",(1.,0.))]
            save_atom_cache(path,atoms,"imagehash",config)
            self.assertEqual(load_atom_cache(path,"imagehash",config),atoms)
            with self.assertRaises(ValueError):
                load_atom_cache(path,"other_image",config)
            with np.load(path,allow_pickle=False) as original:
                values = {key:original[key].copy() for key in original.files}
            values["views"] = values["views"][:0]
            corrupt = Path(temp)/"corrupt.npz"
            np.savez(corrupt,**values)
            with self.assertRaisesRegex(ValueError,"corrupt atom cache"):
                load_atom_cache(corrupt,"imagehash",config)

    def test_missing_reference_falls_back_but_valid_hypothesis_can_add_target(self):
        config = {"gallery_sha256":"galleryhash","feature":{"view_policy":"full"},
                  "numerical":{"opening_cost":.2,"retention_fraction":.5,
                               "nms_iou":.5,"milp_relative_gap":0.}}
        item = {"item_id":"item","source_image_id":"source","query":"car ."}
        candidate = {"candidate_id":"raw:1","box_xyxy_original":[0,0,10,10]}
        pool = {**item,"candidates":[candidate],"m0_final_ids":[],"raw_pool_complete":True}
        common = {**item,"feature_config_sha256":stable_hash(config["feature"]),
                  "numerical_config_sha256":stable_hash(config["numerical"]),
                  "reference_snapshot_sha256":"galleryhash",
                  "global_hypotheses":[],"match_hypotheses":[],"gamma_zero_hypotheses":[],
                  "unresolved_candidate_ids":[]}
        fallback = predict_item(item,pool,{**common,"status":"reference_unavailable"},{},config,"cfg")
        self.assertEqual(fallback["methods"]["R-allocate"]["count"],0)
        self.assertEqual(fallback["methods"]["R-allocate"]["fallback_reason"],"reference_unavailable")
        h = Hypothesis("raw:1|r","raw:1","r","target",1.,{"p":1.},"synthetic",
                       1.,1.,2.,0.,0.,0.,True,0.,0.)
        live = predict_item(item,pool,{**common,"status":"ok",
                                  "global_hypotheses":[h.record()],
                                  "match_hypotheses":[h.record()],
                                  "gamma_zero_hypotheses":[h.record()]},
                            {"r":{"reference_id":"r","semantic_role":"target"}},config,"cfg")
        self.assertEqual(live["methods"]["R-allocate"]["count"],1)
        self.assertEqual(live["methods"]["R-allocate"]["final_boxes"][0]["candidate_id"],"raw:1")

    def test_freeze_refuses_incomplete_prediction_set(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            files = [folder/name for name in ("manifest.jsonl","candidate.jsonl","hypothesis.jsonl",
                                              "gallery.jsonl","checkpoint.bin","config.json",
                                              "predictions.jsonl","freeze.json")]
            manifest,candidates,hypotheses,gallery,checkpoint,config,predictions,freeze = files
            write_jsonl(manifest,[{"item_id":"a"},{"item_id":"b"}])
            for path in (candidates,hypotheses,gallery):
                path.write_text("\n")
            checkpoint.write_bytes(b"synthetic-test-only")
            config.write_text(json.dumps({"status":"protocol_locked","gallery_snapshot":str(gallery),
                "gallery_sha256":file_sha256(gallery),"model_checkpoint":str(checkpoint),
                "model_sha256":file_sha256(checkpoint),"source_split":"synthetic-test",
                "feature":{"backbone":"dinov2_vitb14_reg","patch_size":14,"layer":"x_norm_patchtokens",
                           "long_edge_pixels":28,"view_policy":"full"},
                "numerical":{"gamma":0.,"rho":1.,"opening_cost":.2,
                             "retention_fraction":.5,"nms_iou":.5,
                             "match_max_iterations":20,"match_tolerance":1e-6},
                "candidate_pool":"synthetic",
                "reference_policy":{"min_valid_target":1,"min_valid_competitor":0}}))
            write_jsonl(predictions,[{"item_id":"a"}])
            with self.assertRaisesRegex(ValueError,"incomplete"):
                freeze_predictions(manifest,candidates,hypotheses,gallery,config,predictions,freeze)
            self.assertFalse(freeze.exists())


if __name__ == "__main__":
    unittest.main()
