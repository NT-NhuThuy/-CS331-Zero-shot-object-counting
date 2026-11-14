import unittest
from bidirectional_evidence_controller import reconcile

class ControllerTests(unittest.TestCase):
 def test_unknown_preserves_baseline(self):
  x=reconcile(1,[{'instance_id':'a'},{'instance_id':'b'}],[])
  self.assertEqual((x['count'],x['count_lower_bound'],x['count_upper_bound'],x['count_status']),(2,0,2,'partial'))
 def test_explicit_rejection_is_only_removal(self):
  x=reconcile(1,[{'instance_id':'a'},{'instance_id':'b'}],[{'matched_baseline_id':'a','decision':'REJECTED'}])
  self.assertEqual(x['count'],1);self.assertEqual(len(x['rejected_regions']),1);self.assertEqual(len(x['unknown_regions']),1)
 def test_unmatched_acceptance_is_rescue(self):
  x=reconcile(1,[{'instance_id':'a'}],[{'instance_id':'r','decision':'ACCEPTED'}])
  self.assertEqual(x['count'],2);self.assertEqual(x['count_lower_bound'],1);self.assertEqual(x['count_upper_bound'],2)
 def test_unknown_never_removes(self):
  x=reconcile(1,[{'instance_id':'a'}],[{'matched_baseline_id':'a','decision':'UNKNOWN'}])
  self.assertEqual(x['count'],1)
