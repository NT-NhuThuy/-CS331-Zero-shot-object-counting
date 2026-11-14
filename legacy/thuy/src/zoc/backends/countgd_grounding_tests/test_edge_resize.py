import unittest
import torch
from cgground.backend import CountGDBackend

class ResizeEdgeTests(unittest.TestCase):
    def test_image_edge_rois_do_not_fail_float32_resize(self):
        backend=CountGDBackend.__new__(CountGDBackend)
        backend.cfg={'grid_size':3,'context_scale':1.5};backend.device=torch.device('cpu');backend.strides=[8,16]
        features=[torch.zeros(1,16,100,123),torch.zeros(1,16,50,62)]
        tokens=backend.tokens(features,[[0,0,469,384]],(977,800),(469,384))
        self.assertEqual(tokens.shape,(1,36,16));self.assertTrue(torch.isfinite(tokens).all())
