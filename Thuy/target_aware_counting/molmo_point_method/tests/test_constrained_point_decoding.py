import unittest
from types import SimpleNamespace

import torch

from target_aware_counting.molmo_point_method.constrained_point_decoding import (
    StrictPointGrammarProcessor,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text == '<points coords="':
            return [10, 11]
        return [20 + int(character) for character in text]


class StrictPointGrammarTests(unittest.TestCase):
    def setUp(self):
        self.bounds = SimpleNamespace(
            patch_start=100, patch_end_without_no_more_points=103,
            no_more_points_token_id=103, patch_end=104,
            subpatch_start=104, subpatch_end=106,
            location_start=106, location_end=108)
        self.processor = StrictPointGrammarProcessor(
            self.bounds, prompt_length=2, tokenizer=FakeTokenizer(), eos_token_id=9)

    def allowed(self, generated):
        scores = torch.zeros((1, 128))
        input_ids = torch.tensor([[1, 2, *generated]])
        result = self.processor(input_ids, scores)
        return set(torch.isfinite(result[0]).nonzero().flatten().tolist())

    def test_forces_finite_canonical_serialization(self):
        self.assertEqual(self.allowed([]), {10})
        self.assertEqual(self.allowed([10]), {11})
        self.assertEqual(self.allowed([10, 11]), {100, 101, 102, 103})
        self.assertEqual(self.allowed([10, 11, 100]), {104, 105})
        self.assertEqual(self.allowed([10, 11, 100, 104]), {106, 107})
        self.assertEqual(self.allowed([10, 11, 100, 104, 106]), {21})
        self.assertEqual(self.allowed([10, 11, 100, 104, 106, 21]),
                         {100, 101, 102, 103})
        self.assertEqual(self.allowed([10, 11, 103]), {9})

    def test_rejects_prose_escape(self):
        with self.assertRaisesRegex(ValueError, "patch or terminator"):
            self.allowed([10, 11, 42])


if __name__ == "__main__":
    unittest.main()
