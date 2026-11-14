import json,tempfile,unittest
from pathlib import Path
from assemble_full_cohort_output import main

class AssembleTests(unittest.TestCase):
 def test_full_coverage_and_unknown_baseline(self):
  with tempfile.TemporaryDirectory() as td:
   h=Path(td);(h/'pool').mkdir();(h/'full_p0').mkdir()
   (h/'pool/evaluator_manifest.json').write_text(json.dumps({'rows':[{'id':1},{'id':2}]}))
   (h/'full_p0/predictions.json').write_text(json.dumps({'annotations':[{'image_id':2,'bbox':[1,2,3,4],'id':7}]}))
   ev=h/'ev.json';ev.write_text(json.dumps({'queries':{'1':{'count':1,'count_lower_bound':1,'count_upper_bound':1,'count_status':'exact','accepted_instances':[{'component_id':'c1'}],'rejected_regions':[],'unknown_regions':[],'evidence_trace':[],'query_abstained':False}}}))
   out=h/'out.json';import sys; old=sys.argv;sys.argv=['x','--experiment',str(h),'--evidence',str(ev),'--out',str(out)]
   try: main()
   finally: sys.argv=old
   x=json.loads(out.read_text());self.assertEqual(x['coverage']['output_queries'],2);self.assertEqual(x['queries']['2']['count_status'],'partial');self.assertEqual(x['queries']['2']['count'],1);self.assertEqual(x['queries']['2']['count_upper_bound'],1)
   self.assertEqual(x['queries']['1']['evidence_source'],'rescue_verified')
   self.assertEqual(x['queries']['2']['evidence_source'],'baseline_unverified')
