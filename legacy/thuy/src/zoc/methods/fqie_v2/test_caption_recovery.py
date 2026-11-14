import unittest
from .caption_recovery import extract


class CaptionRecoveryTest(unittest.TestCase):
    def test_descriptive_sentence(self):
        self.assertEqual(extract("<think>\n</think>\nA flamingo with a long, curved neck.", 32)["descriptor"], "flamingo")
        self.assertEqual(extract("The object is a tree.", 12)["descriptor"], "tree")
        self.assertEqual(extract("A red fire truck parked nearby.", 14)["descriptor"], "red fire truck")
        self.assertEqual(extract("A circular, metallic bottle cap with a logo.", 18)["descriptor"], "circular metallic bottle cap")
        self.assertEqual(extract("A dark purple, almost black grape with glossy skin.", 18)["descriptor"], "dark purple almost black grape")

    def test_safety(self):
        for text in ("Several birds standing together.", "A part of a chair.", "A dog and a cat.",
                     "A dog, possibly a toy.", "An object.", "A region is visible.", "<think>A dog", "A circular.", "A dark purple."):
            self.assertEqual(extract(text, 20)["status"], "UNKNOWN", text)
        self.assertEqual(extract("A cat.", 96)["status"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
