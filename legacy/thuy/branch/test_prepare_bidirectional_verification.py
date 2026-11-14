import unittest
from prepare_bidirectional_verification import build_tasks

class PrepareBidirectionalTests(unittest.TestCase):
 def test_emits_baseline_and_rescue_without_gt(self):
  pred={'annotations':[{'id':9,'image_id':1,'bbox':[1,2,3,4]}]}
  public={1:{'source_image_id':77,'query_caption':'cat .'}}
  rescue=[{'candidate_id':'c1','query_id':1,'box_xyxy':[2,3,6,8]}]
  x=build_tasks(pred,public,rescue)
  self.assertEqual([z['role'] for z in x],['baseline','rescue'])
  self.assertTrue(all(z['decision']=='UNKNOWN' for z in x))
  self.assertTrue(all('gt' not in z for z in x))
