import unittest
import torch
from cgground.model import RegionTextVerifier,grounding_loss

class MLPTests(unittest.TestCase):
    def test_forward_gradient_and_roundtrip(self):
        model=RegionTextVerifier(dim=16,heads=2,layers=1,grid_size=3,architecture='mlp',dropout=0)
        logits=model(torch.randn(2,36,16),torch.randn(2,2,16),torch.ones(2,2,dtype=torch.bool))
        loss,_=grounding_loss(logits,torch.tensor([1.,0.]),torch.tensor([0,0]));loss.backward()
        self.assertTrue(torch.isfinite(model.head[-1].weight.grad).all())
        restored=RegionTextVerifier(**model.settings);restored.load_state_dict(model.state_dict(),strict=True)
        self.assertEqual(len(restored.blocks),0)
