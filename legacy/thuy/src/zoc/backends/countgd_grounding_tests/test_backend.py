"""Exercise the actual adapter methods with a small native-interface model.

This verifies hooks/text masking/coordinate behavior, NOT real checkpoint accuracy.
"""
import types
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from cgground.backend import CountGDBackend


class Tokens(dict):
    def __getattr__(self,key):return self[key]
    def to(self,device):return Tokens({k:v.to(device) for k,v in self.items()})


class Tokenizer:
    all_special_ids=[0,101,102]
    def __call__(self,captions,**kwargs):
        ids=torch.tensor([[101,11,12,1012,102]])
        return Tokens(input_ids=ids,attention_mask=torch.ones_like(ids),token_type_ids=torch.zeros_like(ids))


class Backbone(nn.Module):
    def forward(self,tensor):
        features=[types.SimpleNamespace(tensors=F.avg_pool2d(tensor,stride,stride,ceil_mode=True)) for stride in (8,16)]
        return features,None


class Bert(nn.Module):
    def forward(self,input_ids,**kwargs):
        hidden=torch.zeros(*input_ids.shape,16)
        hidden[:,:,0]=input_ids.float()
        return {'last_hidden_state':hidden}


class NativeInterface(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=Backbone()
        self.input_proj=nn.ModuleList([nn.Conv2d(3,16,1),nn.Conv2d(3,16,1)])
        self.tokenizer=Tokenizer();self.bert=Bert();self.feat_map=nn.Identity()
        self.specical_tokens=[101,102,1012];self.max_text_len=256;self.sub_sentence_present=True
        self.fail=False

    def forward(self,image,exemplar,boxes,neg_images,neg_boxes,**kwargs):
        features,_=self.backbone(image)
        values=[proj(f.tensors) for proj,f in zip(self.input_proj,features)]
        if self.fail:raise RuntimeError('deliberate test failure')
        # Simulate downstream in-place fusion; hooks must retain clean pre-fusion tensors.
        for value in values:value.add_(5)
        logit=sum(v.mean() for v in values)
        logits=torch.full((1,3,256),float('-inf'))
        logits[:,:,:5]=logit
        return {'pred_boxes':torch.full((1,3,4),.2),'pred_logits':logits}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.backend=CountGDBackend.__new__(CountGDBackend)
        self.backend.cfg={'grid_size':3,'context_scale':1.5}
        self.backend.device=torch.device('cpu');self.backend.levels=[0,1];self.backend.strides=[8,16]
        self.backend.model=NativeInterface().eval().requires_grad_(False)
        self.tensor=torch.rand(3,43,67)
        self.backend._input=lambda path:(self.tensor,(67,43))
        self.backend.nested_fn=lambda values:torch.stack(values)
        self.backend.mask_fn=lambda tokens,special,tokenizer:(torch.ones(1,5,5,dtype=torch.bool),torch.arange(5)[None],[])

    def test_readonly_hooks_match_plain_forward_and_pre_fusion(self):
        pred=self.backend.predict('unused','apples',check_hooks=True)
        standalone,_,_=self.backend.features('unused')
        for a,b in zip(pred['features'],standalone):torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertEqual(len(pred['native_scores']),3)
        for module in self.backend.model.input_proj:self.assertEqual(len(module._forward_hooks),0)

    def test_hooks_removed_after_exception(self):
        self.backend.model.fail=True
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            self.backend.predict('unused','apples')
        for module in self.backend.model.input_proj:self.assertEqual(len(module._forward_hooks),0)

    def test_text_features_exclude_special_tokens(self):
        text=self.backend.encode_text('apples')
        self.assertEqual(text.shape,(2,16))
        torch.testing.assert_close(text[:,0],torch.tensor([11.,12.]))
        self.assertFalse(text.requires_grad)

    def test_roi_coordinates_and_proposal_features(self):
        pred=self.backend.predict('unused','apples')
        roi=self.backend.proposal_tokens(pred,torch.tensor([0,2]))
        self.assertEqual(roi.shape,(2,36,16));self.assertTrue(torch.isfinite(roi).all())


if __name__=='__main__':unittest.main()
