import unittest
from cgground.infer import selected_points

class NativeTextMaskTests(unittest.TestCase):
    def test_max_logit_alone_does_not_override_native_content_mask(self):
        item={'boxes_cxcywh_normalized':[[.2,.3,.1,.1],[.4,.5,.1,.1]],'native_scores':[.9,.8],
              'native_text_pass':[False,True],'grounding_scores':[.99,.99],'usable_rois':[True,True],
              'width':100,'height':80}
        for mode in ['baseline','verifier']:
            points,indices=selected_points(item,mode)
            self.assertEqual(indices.tolist(),[1]);self.assertEqual(len(points),1)
