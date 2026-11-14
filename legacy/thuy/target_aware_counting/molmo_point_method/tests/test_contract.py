import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from target_aware_counting.molmo_point_method.native_greedy import load_existing, validate_points
from target_aware_counting.molmo_point_method.native_greedy import load_manifest, sha256
from target_aware_counting.molmo_point_method.freeze import verify_and_freeze
from target_aware_counting.molmo_point_method.evaluate import matched_points
from target_aware_counting.molmo_point_method.head_scoring import score_step, native_greedy_next
from target_aware_counting.molmo_point_method.reference_replay import replay
from target_aware_counting.molmo_point_method.point_grammar import validate_point_tokens
from target_aware_counting.molmo_point_method.proposal_mapping import proposal_patch_indices
from target_aware_counting.molmo_point_method.reference_search import native_replay, spatial_search, spatial_search_cached
from target_aware_counting.molmo_point_method.score_audit import direct_step_probability, representative_steps
from target_aware_counting.molmo_point_method.run_reference import decode_path, summarize_decoded_paths
from target_aware_counting.molmo_point_method.snapshot_lock import lock, verify
from target_aware_counting.molmo_point_method.f1_sidecar import load_f1_sidecar
from target_aware_counting.molmo_point_method.export_a_predictions import prediction_for_method
from target_aware_counting.molmo_point_method.set_risk import Hypothesis, decide, gospa


class ContractTests(unittest.TestCase):
    def test_a_export_uses_proper_fallback_only_when_f1_unavailable(self):
        branch = {"decision_status": "ready",
                  "decision": {"map_points_xy": [[1, 2]],
                               "mbr_points_xy": [],
                               "map_trace_id": "3", "mbr_trace_id": "4",
                               "bank_size_unique": 2},
                  "stats": {"complete_sequences": 5,
                            "invalid_complete_sequences": 1,
                            "active_truncated": 1}}
        row = {"native": {"status": "complete", "points_xy": [[2, 3]],
                          "output_tokens": 10},
               "proper_beam": branch, "guided_beam": None,
               "f1_raw_available": False}
        result = prediction_for_method(row, "guided_mbr")
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["trace"]["source"], "proper_fallback_no_f1")
        self.assertEqual(prediction_for_method(row, "proper_map")["count"], 1)
        row["f1_raw_available"] = True
        with self.assertRaises(ValueError):
            prediction_for_method(row, "guided_mbr")

    def test_invalid_beam_completion_is_audited_not_counted_as_empty(self):
        bounds = SimpleNamespace(patch_start=10, patch_end_without_no_more_points=11,
                                 patch_end=12, no_more_points_token_id=11,
                                 subpatch_start=12, subpatch_end=13,
                                 location_start=13, location_end=22)
        model = SimpleNamespace(
            device=torch.device("cpu"),
            model=SimpleNamespace(build_token_bounds=lambda pooling: bounds),
            extract_image_points=lambda text, pooling, mapping, sizes: [])
        processor = SimpleNamespace(
            post_process_image_text_to_text=lambda tokens, **kwargs: ["<POINT_0><POINT_1><POINT_2></point>"])
        metadata = {"token_pooling": [[0]], "subpatch_mapping": [[[0]]],
                    "image_sizes": [[100, 100]]}
        malformed = SimpleNamespace(tokens=(10, 12, 13, 2), log_probability=-0.1,
                                    complete=True)
        decoded = decode_path(model, processor, metadata, (100, 100), malformed)
        self.assertFalse(decoded["valid_for_decision"])
        self.assertIsNone(decoded["count"])
        self.assertIn("differs", decoded["decode_error"])
        summary = summarize_decoded_paths([malformed], [decoded],
                                          {"complete_sequences": 1}, 32.0)
        self.assertEqual(summary["decision_status"], "no_valid_complete_sequences")
        self.assertIsNone(summary["decision"])
        self.assertEqual(summary["stats"]["invalid_complete_sequences"], 1)
        empty = SimpleNamespace(tokens=(11, 2), log_probability=-0.2, complete=True)
        empty_decoded = decode_path(model, processor, metadata, (100, 100), empty)
        self.assertTrue(empty_decoded["valid_for_decision"])
        self.assertEqual(empty_decoded["count"], 0)
        mixed = summarize_decoded_paths([malformed, empty], [decoded, empty_decoded],
                                        {"complete_sequences": 2}, 32.0)
        self.assertEqual(mixed["decision_status"], "ready")
        self.assertEqual(mixed["decision"]["map_trace_id"], "1")

    def test_empty_set_and_assignment(self):
        self.assertEqual(gospa([], [], 10), 0)
        self.assertEqual(gospa([], [(1, 1)], 10), 50)
        self.assertEqual(gospa([(0, 0), (10, 0)], [(10, 0), (0, 0)], 10), 0)
        self.assertEqual(gospa([(0, 0)], [(100, 0)], 10), 100)

    def test_deduplicated_bank_and_empty_competition(self):
        bank = [Hypothesis((), -0.1, "empty"),
                Hypothesis(((4, 4),), -1.0, "one"),
                Hypothesis(((4, 4),), -2.0, "rediscovery")]
        result = decide(bank, 10)
        self.assertEqual(result["bank_size_unique"], 2)
        self.assertEqual(result["map_trace_id"], "empty")
        self.assertEqual(result["mbr_trace_id"], "empty")
        self.assertNotIn("rediscovery", result["weights"])

    def test_invalid_probability_and_points(self):
        with self.assertRaises(ValueError):
            decide([Hypothesis((), float("nan"), "bad")], 10)
        with self.assertRaises(ValueError):
            validate_points([[0, 0, 101, 1]], (100, 100))
        self.assertEqual(validate_points([["a", 0, 0, 100]], (100, 100)), [[0, 100]])

    def test_point_matching_is_one_to_one(self):
        self.assertEqual(matched_points([[5, 5], [5, 5]], [[0, 0, 10, 10]],
                                        "lvis_dev300", 16), 1)
        self.assertEqual(matched_points([[5, 5]], [[6, 5], [7, 5]],
                                        "fsc_primary_dev100", 2), 1)
        self.assertEqual(matched_points([[50, 50]], [[0, 0]],
                                        "fsc_primary_dev100", 2), 0)

    def test_spatial_probability_factorization_and_stop(self):
        bounds = SimpleNamespace(patch_start=3, patch_end=6, subpatch_start=6,
                                 subpatch_end=8, location_start=8, location_end=10,
                                 no_more_points_token_id=5)
        def grammar(prefix, scores):
            scores = scores.clone()
            last = int(prefix[0, -1])
            if last == 5:
                scores[:, 3:] = -float("inf")
            elif 3 <= last < 5:
                scores[:, :6] = -float("inf")
                scores[:, 8:] = -float("inf")
            elif 6 <= last < 8:
                scores[:, :8] = -float("inf")
            else:
                scores[:, 6:] = -float("inf")
            return scores
        output = SimpleNamespace(logits=torch.tensor([[[0., 0., 0., 2., -1e5,
                                                        -1e5, -1e5, -1e5, -1e5, -1e5]]]),
                                 patch_logits=torch.tensor([[[0., 1., -1.]]]),
                                 subpatch_logits=None, location_logits=None)
        options = score_step(output, torch.tensor([[0]]), grammar, bounds,
                             max_language=3, max_spatial=3)
        self.assertAlmostEqual(sum(math.exp(o.log_probability) for o in options), 1.0, places=6)
        self.assertIn(5, [o.token_id for o in options])
        self.assertEqual(next(o.kind for o in options if o.token_id == 5), "no_more_points")
        after_stop = score_step(output, torch.tensor([[0, 5]]), grammar, bounds,
                                max_language=3, max_spatial=3)
        self.assertTrue(all(o.kind == "language" for o in after_stop))
        self.assertEqual(native_greedy_next(output, torch.tensor([[0]]), grammar), 3)

    def test_direct_head_probability_matches_hand_calculation(self):
        bounds = SimpleNamespace(patch_start=3, patch_end=6, subpatch_start=6,
                                 subpatch_end=8, location_start=8, location_end=10,
                                 no_more_points_token_id=5)
        def grammar(prefix, scores):
            scores = scores.clone()
            last = int(prefix[0, -1])
            if 3 <= last < 6:
                scores[:, :6] = -float("inf")
                scores[:, 8:] = -float("inf")
            elif 6 <= last < 8:
                scores[:, :8] = -float("inf")
            else:
                scores[:, 6:] = -float("inf")
            return scores
        output = SimpleNamespace(
            logits=torch.tensor([[[0., 0., 0., -1e5, 2., -1e5,
                                   -1e5, -1e5, -1e5, -1e5]]]),
            patch_logits=torch.tensor([[[0., 1., -1.]]]),
            subpatch_logits=None, location_logits=None)
        base = torch.log_softmax(torch.tensor([0., 0., 0., 2.]), 0)
        spatial = torch.log_softmax(torch.tensor([0., 1., -1.]), 0)
        kind, direct = direct_step_probability(output, torch.tensor([[0]]), grammar, bounds, 3)
        self.assertEqual(kind, "patch")
        self.assertAlmostEqual(direct, float(base[-1]+spatial[0]), places=5)
        scored = score_step(output, torch.tensor([[0]]), grammar, bounds,
                            forced_patch_indices=(0,), max_spatial=3)
        self.assertAlmostEqual(next(o.log_probability for o in scored if o.token_id == 3),
                               direct, places=5)
        kind, stop = direct_step_probability(output, torch.tensor([[0]]), grammar, bounds, 5)
        self.assertEqual(kind, "no_more_points")
        self.assertAlmostEqual(stop, float(base[-1]+spatial[2]), places=5)
        output.logits[0, 0, 6:8] = torch.tensor([1., 2.])
        output.subpatch_logits = torch.tensor([[[1., 2.]]])
        kind, direct = direct_step_probability(output, torch.tensor([[0, 3]]), grammar, bounds, 7)
        self.assertEqual(kind, "subpatch")
        self.assertAlmostEqual(direct, float(torch.log_softmax(torch.tensor([1., 2.]), 0)[1]), places=5)
        output.logits[0, 0, 8:10] = torch.tensor([3., 0.])
        output.location_logits = torch.tensor([[[3., 0.]]])
        kind, direct = direct_step_probability(output, torch.tensor([[0, 3, 7]]), grammar, bounds, 8)
        self.assertEqual(kind, "location")
        self.assertAlmostEqual(direct, float(torch.log_softmax(torch.tensor([3., 0.]), 0)[0]), places=5)
        self.assertEqual(representative_steps([3, 6, 8, 5, 2], bounds, (2,)), [0, 1, 2, 3, 4])

    def test_reference_replay_isolates_branch_state(self):
        class FakeModel:
            def prepare_inputs_for_generation(self, prefix, **kwargs):
                self.logits_to_keep = kwargs.get("logits_to_keep")
                return {"token": int(prefix[0, -1]), "cache": kwargs.get("past_key_values"),
                        "cache_position": kwargs["cache_position"]}

            def __call__(self, token, cache, cache_position, return_dict):
                assert return_dict
                history = (cache or []) + [token]
                return SimpleNamespace(history=history, past_key_values=history)

            def _update_model_kwargs_for_generation(self, output, kwargs, is_encoder_decoder):
                assert not is_encoder_decoder
                return {**kwargs, "past_key_values": output.past_key_values,
                        "cache_position": kwargs["cache_position"][-1:] + 1}

        inputs = {"input_ids": torch.tensor([[7]])}
        model = FakeModel()
        self.assertEqual(replay(model, inputs, [10, 20]).history, [7, 10, 20])
        self.assertEqual(model.logits_to_keep, 1)
        self.assertEqual(replay(FakeModel(), inputs, [11]).history, [7, 11])

    def test_point_grammar_distinguishes_empty_from_bad_decode(self):
        bounds = SimpleNamespace(patch_start=100, patch_end_without_no_more_points=102,
                                 patch_end=103, no_more_points_token_id=102,
                                 subpatch_start=103, subpatch_end=105,
                                 location_start=105, location_end=114)
        self.assertEqual(validate_point_tokens([1, 102, 2], bounds, 0)["point_triples"], 0)
        self.assertEqual(validate_point_tokens([100, 104, 107, 8, 102, 2], bounds, 1)["point_triples"], 1)
        for tokens, count in (([100, 104], 0), ([100, 104, 107], 0),
                              ([102, 100, 104, 107], 1), ([104], 0)):
            with self.assertRaises(ValueError):
                validate_point_tokens(tokens, bounds, count)

    def test_proposal_to_native_patch_mapping_and_unmapped(self):
        metadata = {"subpatch_mapping": [[[10, 11], [12, 13]]],
                    "token_pooling": [[10, 11], [12, 13]],
                    "image_sizes": [(100, 100)]}
        candidates = [{"candidate_id": "left", "box_xyxy_original": [0, 0, 40, 40]},
                      {"candidate_id": "right", "box_xyxy_original": [60, 60, 100, 100]},
                      {"candidate_id": "bad", "box_xyxy_original": [-1, 0, 10, 10]}]
        result = proposal_patch_indices(candidates, metadata)
        self.assertEqual(result["patch_indices"], [0, 1])
        self.assertEqual(result["candidate_unmapped_ids"], ["bad"])
        self.assertEqual(result["candidate_capped"], 0)

    def test_reference_search_keeps_empty_and_one_point_paths(self):
        bounds = SimpleNamespace(patch_start=10, patch_end=11,
                                 subpatch_start=11, subpatch_end=12,
                                 location_start=12, location_end=14,
                                 no_more_points_token_id=10)
        class ToyModel:
            def __init__(self):
                self.model = SimpleNamespace(build_token_bounds=lambda pooling: bounds)
                self.generation_config = SimpleNamespace(eos_token_id=2)

            def build_logit_processor_from_inputs(self, inputs):
                return [lambda prefix, scores: scores]

            def prepare_inputs_for_generation(self, prefix, **kwargs):
                return {"token": int(prefix[0, -1]), "cache": kwargs.get("past_key_values")}

            def __call__(self, token, cache, return_dict):
                history = (cache or []) + [token]
                scores = torch.full((1, 1, 14), -float("inf"))
                if history == [7]:
                    scores[0, 0, 1], scores[0, 0, 2] = 1., 0.
                else:
                    scores[0, 0, 2] = 2.
                return SimpleNamespace(logits=scores, patch_logits=None,
                                       past_key_values=history)

            def _update_model_kwargs_for_generation(self, output, kwargs, is_encoder_decoder):
                return {**kwargs, "past_key_values": output.past_key_values}

        model = ToyModel()
        inputs = {"input_ids": torch.tensor([[7]]), "image_token_pooling": torch.zeros(1, 1, 1)}
        self.assertEqual(native_replay(model, inputs, max_new_tokens=3).tokens, (1, 2))
        paths, stats = spatial_search(model, inputs, beam_width=2, max_new_tokens=3,
                                      max_language=2, max_spatial=1)
        self.assertIn((2,), [p.tokens for p in paths])
        self.assertIn((1, 2), [p.tokens for p in paths])
        self.assertGreater(stats["replay_forward_calls"], 0)

    def test_cached_search_matches_reference_with_mutable_sibling_caches(self):
        bounds = SimpleNamespace(patch_start=10, patch_end=11,
                                 subpatch_start=11, subpatch_end=12,
                                 location_start=12, location_end=14,
                                 no_more_points_token_id=10)
        class MutableToyModel:
            def __init__(self):
                self.model = SimpleNamespace(build_token_bounds=lambda pooling: bounds)
                self.generation_config = SimpleNamespace(eos_token_id=2)

            def build_logit_processor_from_inputs(self, inputs):
                return [lambda prefix, scores: scores]

            def prepare_inputs_for_generation(self, prefix, **kwargs):
                return {"token": int(prefix[0, -1]),
                        "cache": kwargs.get("past_key_values")}

            def __call__(self, token, cache, return_dict):
                assert return_dict
                history = cache if cache is not None else []
                history.append(token)  # Simulate in-place DynamicCache update.
                scores = torch.full((1, 1, 14), -float("inf"))
                if history == [7]:
                    scores[0, 0, 1], scores[0, 0, 3] = 1., 0.5
                elif history == [7, 1]:
                    scores[0, 0, 2], scores[0, 0, 4] = 2., 1.
                elif history == [7, 3]:
                    scores[0, 0, 2], scores[0, 0, 5] = 1., 0.
                else:
                    scores[0, 0, 2] = 3.
                return SimpleNamespace(logits=scores, patch_logits=None,
                                       past_key_values=history)

            def _update_model_kwargs_for_generation(self, output, kwargs, is_encoder_decoder):
                assert not is_encoder_decoder
                return {**kwargs, "past_key_values": output.past_key_values}

        model = MutableToyModel()
        inputs = {"input_ids": torch.tensor([[7]]),
                  "image_token_pooling": torch.zeros(1, 1, 1)}
        params = dict(beam_width=2, max_new_tokens=3, max_language=2, max_spatial=1)
        reference, ref_stats = spatial_search(model, inputs, **params)
        cached, cached_stats = spatial_search_cached(model, inputs, **params)
        self.assertEqual([(p.tokens, p.log_probability) for p in cached],
                         [(p.tokens, p.log_probability) for p in reference])
        self.assertEqual(ref_stats["complete_sequences"], cached_stats["complete_sequences"])
        self.assertLess(cached_stats["cached_forward_calls"], ref_stats["replay_forward_calls"])

    def test_cached_search_tail_mass_stop_has_explicit_bound(self):
        bounds = SimpleNamespace(patch_start=3, patch_end=4,
                                 subpatch_start=4, subpatch_end=5,
                                 location_start=5, location_end=6)
        class Toy:
            model = SimpleNamespace(build_token_bounds=lambda pooling: bounds)
            generation_config = SimpleNamespace(eos_token_id=2)
            def build_logit_processor_from_inputs(self, inputs):
                return [lambda prefix, scores: scores]
            def prepare_inputs_for_generation(self, prefix, **kwargs):
                return {"cache": kwargs.get("past_key_values")}
            def __call__(self, cache, return_dict):
                return SimpleNamespace(logits=torch.tensor([[[float("-inf"), -20., 0.,
                                                               float("-inf"), float("-inf"),
                                                               float("-inf")]]]),
                                       patch_logits=None,
                                       past_key_values=[7] if cache is None else cache)
            def _update_model_kwargs_for_generation(self, output, kwargs, is_encoder_decoder):
                return {**kwargs, "past_key_values": output.past_key_values}
        paths, stats = spatial_search_cached(
            Toy(), {"input_ids": torch.tensor([[7]]),
                    "image_token_pooling": torch.zeros(1, 1, 1)},
            beam_width=2, max_new_tokens=100, max_language=2, max_spatial=1,
            tail_mass_epsilon=1e-6)
        self.assertEqual([path.tokens for path in paths], [(2,)])
        self.assertTrue(stats["stopped_by_mass_bound"])
        self.assertLess(stats["remaining_mass_ratio_bound"], 1e-6)
        self.assertEqual(stats["cached_forward_calls"], 1)

    def test_snapshot_lock_detects_mutated_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "config.json").write_text(json.dumps({"model_type": "molmo_point"}))
            (path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "model-00001.safetensors"}}))
            (path / "modeling_molmo_point.py").write_text("# source\n")
            (path / "tokenizer.json").write_text("{}")
            (path / "model-00001.safetensors").write_bytes(b"weight")
            revision = "a"*40
            self.assertEqual(lock(path, revision)["repo"], "allenai/MolmoPoint-8B")
            self.assertEqual(verify(path, revision)["revision"], revision)
            (path / "model-00001.safetensors").write_bytes(b"other!")
            with self.assertRaises(ValueError):
                verify(path, revision)

    def test_f1_guidance_uses_full_pool_and_marks_absent_unavailable(self):
        for cohort, expected_available in (("lvis_dev300", True),
                                           ("fsc_primary_dev100", True),
                                           ("fsc_verified_absent111", False)):
            item = load_manifest(cohort)[1][0]
            path, candidates, available = load_f1_sidecar(cohort, item)
            self.assertEqual(available, expected_available)
            self.assertTrue(all(c["forward_id"] == "F1_raw" for c in candidates))
            if cohort == "fsc_primary_dev100":
                self.assertTrue(path.name.endswith("_full.jsonl"))

    def test_resume_rejects_mixed_config_and_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.jsonl"
            row = {"item_id": "a", "contract": {"model": "x"}}
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(len(load_existing(path, row["contract"], {"a"})), 1)
            with self.assertRaises(ValueError):
                load_existing(path, {"model": "y"}, {"a"})
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_existing(path, row["contract"], {"a"})

    def test_freeze_requires_full_complete_identity(self):
        manifest, items = load_manifest("lvis_dev300")
        contract = {"cohort": "lvis_dev300", "manifest_sha256": sha256(manifest)}
        rows = [{"item_id": item["item_id"], "source_image_id": item["source_image_id"],
                 "query": item["query"], "contract": contract, "status": "complete",
                 "count": 0, "points_xy": []} for item in items]
        with tempfile.TemporaryDirectory() as folder:
            predictions = Path(folder) / "predictions.jsonl"
            frozen = Path(folder) / "freeze.json"
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(verify_and_freeze("lvis_dev300", predictions, frozen)["items"], 300)
            self.assertTrue(frozen.exists())
            rows[0]["status"] = "truncated"
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaises(ValueError):
                verify_and_freeze("lvis_dev300", predictions, Path(folder) / "bad.json")


if __name__ == "__main__":
    unittest.main()
