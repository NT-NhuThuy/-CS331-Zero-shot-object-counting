import unittest
from candidate_quality import admissible
class QualityTests(unittest.TestCase):
    def test_two_views_rejected(self): self.assertFalse(admissible({'support_count':2})[0])
    def test_three_views_admitted(self): self.assertTrue(admissible({'support_count':3,'box_xyxy':[1,2,3,4]})[0])
    def test_unknown_support_not_inferred(self): self.assertFalse(admissible({'support_views':[]})[0])
if __name__=='__main__': unittest.main()
