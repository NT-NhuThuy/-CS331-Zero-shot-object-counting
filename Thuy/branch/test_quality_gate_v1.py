import unittest
from quality_gate_v1 import admit
class QualityGateV1Tests(unittest.TestCase):
 def base(self):return {'support_count':3,'box_xyxy':[1,2,10,20],'semantic':'YES','whole_instance':'YES'}
 def test_admits_only_complete_evidence(self):self.assertEqual(admit(self.base()),(True,None))
 def test_unknown_rejected(self):
  x=self.base();x['semantic']='UNKNOWN';self.assertFalse(admit(x)[0])
 def test_part_rejected(self):
  x=self.base();x['whole_instance']='NO';self.assertFalse(admit(x)[0])
