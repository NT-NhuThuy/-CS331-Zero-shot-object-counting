#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from build_evidence_output import main


class EvidenceOutputTests(unittest.TestCase):
    def test_cli_emits_contract_validated_output(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'consensus_candidates.jsonl').write_text(json.dumps({
                'candidate_id':'q1_c1','query_id':1,'source_image_id':42,
                'box_xyxy':[1,2,10,20],'support_views':['v0','v1'],
                'proposal_score':0.9,'source_model':'fixture-model'})+'\n')
            (root/'identity_unary_decisions.jsonl').write_text(json.dumps({
                'candidate_id':'q1_c1','semantic':'YES','whole':'YES'})+'\n')
            (root/'identity_pair_decisions.jsonl').write_text('')
            out=root/'evidence.json'
            import sys
            old=sys.argv
            try:
                sys.argv=['build_evidence_output.py','--experiment',str(root),'--out',str(out)]
                main()
            finally:
                sys.argv=old
            payload=json.loads(out.read_text())
            self.assertEqual(payload['schema'],'fqie_verified_instance_count_v1')
            self.assertEqual(payload['queries']['1']['count'],1)
            self.assertEqual(payload['queries']['1']['count_status'],'exact')
            accepted=payload['queries']['1']['accepted_instances'][0]
            self.assertEqual(accepted['box_xyxy'],[1,2,10,20])
            self.assertEqual(accepted['semantic'],'YES')
            self.assertEqual(accepted['whole_instance'],'YES')
            self.assertIn('artifact_hashes',accepted)
            manifest=json.loads((root/'evidence.input_manifest.json').read_text())
            self.assertEqual(manifest['schema'],'fqie_evidence_input_manifest_v1')
            self.assertEqual(set(manifest['files']),{
                'consensus_candidates.jsonl','identity_unary_decisions.jsonl','identity_pair_decisions.jsonl'})


if __name__=='__main__':
    unittest.main()
