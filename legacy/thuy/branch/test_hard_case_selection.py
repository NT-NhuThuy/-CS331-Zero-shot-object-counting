import json,tempfile,unittest
from pathlib import Path
from freeze_hard_case_selection import main
class SelectionTests(unittest.TestCase):
 def test_freeze(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);m=p/'m.json';o=p/'o.json';m.write_text(json.dumps({'status':'DEV_ONLY_GT_STRATIFIED_BEFORE_HUMAN_ANNOTATION','rows':[{'case_type':'zero_present'}]}))
   import sys;old=sys.argv;sys.argv=['x','--manifest',str(m),'--out',str(o)]
   try:main()
   finally:sys.argv=old
   self.assertEqual(json.loads(o.read_text())['case_counts']['zero_present'],1)
if __name__=='__main__':unittest.main()
