import json,tempfile,unittest
from pathlib import Path
from export_annotation_templates import main
class TemplateTests(unittest.TestCase):
 def test_template_fields(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);m=p/'m.json';o=p/'o.jsonl';m.write_text(json.dumps({'rows':[{'query_id':1,'source_image_id':2,'query_name':'apple','case_type':'zero_present','baseline_count':0}]}))
   import sys;old=sys.argv;sys.argv=['x','--manifest',str(m),'--out',str(o)]
   try:main()
   finally:sys.argv=old
   x=json.loads(o.read_text());self.assertEqual(x['annotator_status'],'needs_review');self.assertIn('negative_regions',x)
if __name__=='__main__':unittest.main()
