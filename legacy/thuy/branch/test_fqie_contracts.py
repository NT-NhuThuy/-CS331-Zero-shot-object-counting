#!/usr/bin/env python3
import unittest
from fqie_contracts import validate_candidate,validate_count

def candidate(**overrides):
    x={'candidate_id':'q1_c1','query_id':1,'box_xyxy':[1,2,10,20],
       'source_image_id':7,'view_id':'tile0','view_provenance':['tile0'],
       'proposal_score':0.8,'source_model':'countgd++',
       'semantic':'UNKNOWN','whole_instance':'UNKNOWN','identity':'UNKNOWN',
       'evidence_refs':['ev-1'],'artifact_hashes':{'proposal':'a'*64},'decision':'UNKNOWN'}
    x.update(overrides); return x

class ContractTests(unittest.TestCase):
    def test_candidate_unknown_is_valid_but_not_accepted(self):
        x=candidate()
        self.assertEqual(validate_candidate(x),(True,None))
    def test_accepted_requires_two_positive_checks(self):
        x=candidate(semantic='YES',whole_instance='NO',decision='ACCEPTED')
        self.assertFalse(validate_candidate(x)[0])
    def test_same_rejected_conflict(self):
        x=candidate(semantic='YES',whole_instance='YES',identity='SAME',decision='REJECTED')
        self.assertFalse(validate_candidate(x)[0])

    def test_candidate_requires_provenance_and_hashes(self):
        x=candidate(); del x['artifact_hashes']
        self.assertEqual(validate_candidate(x),(False,"missing:['artifact_hashes']"))
    def test_partial_count(self):
        x={'count':3,'count_lower_bound':3,'count_upper_bound':4,'count_status':'partial','accepted_instances':[],'rejected_regions':[],'unknown_regions':[],'evidence_trace':[],'query_abstained':False}
        self.assertEqual(validate_count(x),(True,None))
    def test_exact_requires_equal_bounds(self):
        x={'count':3,'count_lower_bound':3,'count_upper_bound':4,'count_status':'exact','accepted_instances':[],'rejected_regions':[],'unknown_regions':[],'evidence_trace':[],'query_abstained':False}
        self.assertFalse(validate_count(x)[0])
    def test_abstained_requires_flag(self):
        x={'count':3,'count_lower_bound':3,'count_upper_bound':4,'count_status':'abstained','accepted_instances':[],'rejected_regions':[],'unknown_regions':[],'evidence_trace':[],'query_abstained':False}
        self.assertFalse(validate_count(x)[0])

if __name__=='__main__':unittest.main()
