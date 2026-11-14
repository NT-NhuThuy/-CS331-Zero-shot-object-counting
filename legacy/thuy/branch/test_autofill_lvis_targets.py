import json,tempfile,unittest
from pathlib import Path
from autofill_lvis_targets import main
class AutofillTests(unittest.TestCase):
 def test_fills_target_fields(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);t=p/'t.jsonl';pool=p/'pool.json';o=p/'o.jsonl';t.write_text(json.dumps({'query_id':1})+'\n');pool.write_text(json.dumps({'images':[{'id':1,'source_image_id':5,'lvis_category_id':7,'width':100,'height':80}]}))
   import autofill_lvis_targets as m
   old=m.ROOT
   try:
    # Test only wiring through a temporary synthetic LVIS root.
    m.ROOT=p; (p/'data/lvis_v1_val_source').mkdir(parents=True); (p/'data/lvis_v1_val_source/lvis_v1_val.json').write_text(json.dumps({'annotations':[{'id':3,'image_id':5,'category_id':7,'bbox':[1,2,10,20]}]}))
    import sys;oldargv=sys.argv;sys.argv=['x','--template',str(t),'--pool',str(pool),'--out',str(o)];main();sys.argv=oldargv
   finally:m.ROOT=old
   x=json.loads(o.read_text());self.assertEqual(x['gt_count'],1);self.assertEqual(x['target_instances'][0]['instance_id'],'gt_3')
if __name__=='__main__':unittest.main()
