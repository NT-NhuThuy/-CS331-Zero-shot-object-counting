import unittest
from fqie_contracts import validate_count
class FullOutputValidatorTests(unittest.TestCase):
 def test_partial_unknown_count_is_contract_valid(self):
  x={'count':0,'count_lower_bound':0,'count_upper_bound':2,'count_status':'partial','accepted_instances':[],'rejected_regions':[],'unknown_regions':[{'status':'UNKNOWN'}],'evidence_trace':[],'query_abstained':False}
  self.assertEqual(validate_count(x),(True,None))
