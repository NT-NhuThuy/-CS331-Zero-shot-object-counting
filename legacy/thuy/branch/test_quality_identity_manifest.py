import json,tempfile,unittest
from pathlib import Path
from prepare_quality_identity_manifest import main
class ManifestTests(unittest.TestCase):
 def test_filters_support(self):
  with tempfile.TemporaryDirectory() as d:
   h=Path(d);(h/'images').mkdir();(h/'images/a.jpg').write_bytes(b'jpg');(h/'consensus_lock.json').write_text(json.dumps({'status':'V10_CROSS_VIEW_CONSENSUS_LOCKED_BEFORE_GT'}));(h/'consensus_candidates.jsonl').write_text(json.dumps({'candidate_id':'a','query_id':1,'box_xyxy':[1,1,3,3],'support_count':2})+'\n'+json.dumps({'candidate_id':'b','query_id':1,'box_xyxy':[1,1,3,3],'support_count':3})+'\n');(h/'pool').mkdir();(h/'pool/inference_coco.json').write_text(json.dumps({'images':[{'id':1,'file_name':'a.jpg','query_name':'x','width':10,'height':10}]}));o=h/'o.json';import sys;old=sys.argv;sys.argv=['x','--experiment',str(h),'--out',str(o)]
   try:main()
   finally:sys.argv=old
   self.assertEqual(len(json.loads(o.read_text())['items']),1)
if __name__=='__main__':unittest.main()
