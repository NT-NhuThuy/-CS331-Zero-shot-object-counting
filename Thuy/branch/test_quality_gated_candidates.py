import json,tempfile,unittest
from pathlib import Path
from build_quality_gated_candidates import main
class GateTests(unittest.TestCase):
 def test_gate(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);i=p/'i.jsonl';o=p/'o.jsonl';i.write_text(json.dumps({'candidate_id':'a','support_count':2})+'\n'+json.dumps({'candidate_id':'b','support_count':3})+'\n')
   import sys;old=sys.argv;sys.argv=['x','--input',str(i),'--out',str(o)]
   try:main()
   finally:sys.argv=old
   self.assertEqual(len(o.read_text().splitlines()),1);self.assertEqual(json.loads(o.read_text())['candidate_id'],'b')
if __name__=='__main__':unittest.main()
