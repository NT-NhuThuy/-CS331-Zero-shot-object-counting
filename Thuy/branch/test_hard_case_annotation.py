import unittest
from hard_case_annotation import validate_case
class AnnotationTests(unittest.TestCase):
 def base(self):return {'query_id':1,'image_width':100,'image_height':80,'query':'apple','gt_count':1,'target_instances':[{'instance_id':'i1','box_xyxy':[1,2,20,30],'region_type':'whole_target'}],'negative_regions':[],'ambiguous_regions':[],'annotator_status':'complete'}
 def test_valid(self):self.assertEqual(validate_case(self.base()),(True,None))
 def test_count_mismatch(self):
  x=self.base();x['gt_count']=2;self.assertFalse(validate_case(x)[0])
 def test_duplicate_instance(self):
  x=self.base();x['target_instances'].append({'instance_id':'i1','box_xyxy':[30,2,40,30],'region_type':'whole_target'});x['gt_count']=2;self.assertFalse(validate_case(x)[0])
 def test_invalid_negative_type(self):
  x=self.base();x['negative_regions']=[{'region_id':'n1','box_xyxy':[2,3,4,5],'region_type':'whole_target'}];self.assertFalse(validate_case(x)[0])
if __name__=='__main__':unittest.main()
