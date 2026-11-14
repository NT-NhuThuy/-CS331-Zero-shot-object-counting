from __future__ import annotations

import copy
import contextlib
import io
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
from torch import nn

from cgground.io import write_json, write_jsonl, read_json, read_jsonl, digest, load_tensor_file
from cgground.fsc import FSC147
from cgground.protocol import make_manifest, make_region_review, checked_region_labels
from cgground.geometry import region_tokens, expand_boxes, cxcywh_to_xyxy
from cgground.model import RegionTextVerifier, grounding_loss
from cgground.cache import build_cache, RegionDataset, collate_regions
from cgground.train import train_verifier, validate_training_caches
from cgground.infer import predict_manifest, score_proposals, selected_points
from cgground.metrics import point_match, evaluate
from cgground.review import render_review
from cgground.cli import main

# CPU tests are deterministic and fast, independent of native CountGD CUDA extensions.
torch.set_num_threads(1)


def fixture(root):
    root = Path(root)
    images = root / 'images_384_VarV2'
    images.mkdir()
    names = ['train_a.jpg', 'train_b.jpg', 'val_a.jpg', 'val_b.jpg', 'test_a.jpg', 'test_b.jpg']
    classes = ['apples', 'pears', 'strawberries', 'raspberries', 'cars', 'buses']
    annotations = {}
    for i, name in enumerate(names):
        Image.new('RGB', (120, 80), (i * 30, 20, 20)).save(images / name, quality=100, subsampling=0)
        annotations[name] = {'H': 800, 'W': 1200, 'ratio_h': 0.1, 'ratio_w': 0.1,
             'points': [[20.0, 30.0], [70.0, 30.0]],
             'box_examples_coordinates': [[[10,20],[30,20],[30,40],[10,40]],
                                         [[60,20],[80,20],[80,40],[60,40]]]}
    write_json(root / 'annotation_FSC147_384.json', annotations)
    write_json(root / 'Train_Test_Val_FSC_147.json',
               {'train': names[:2], 'val': names[2:4], 'test': names[4:], 'val_coco': names[2:3]})
    (root / 'ImageClasses_FSC147.txt').write_text('\n'.join(f'{n}\t{c}' for n,c in zip(names,classes))+'\n')
    cfg = {'data_root': str(root), 'images_dir': str(images), 'seed': 42, 'device': 'cpu',
           'grid_size': 3, 'context_scale': 1.5, 'roi_chunk': 2, 'resize_short': 80, 'resize_max': 120,
           'baseline_threshold': 0.23, 'pool_threshold': 0.23, 'grounding_threshold': 0.5,
           'point_radius_relative': 0.01, 'model': {'dim':16,'heads':2,'layers':1,'dropout':0.0}}
    return cfg, FSC147(cfg)


class FakeBackend:
    """Synthetic image encoder, NOT a replacement for the native GPU smoke test."""
    def __init__(self, cfg, queries):
        self.cfg, self.device = cfg, torch.device('cpu')
        self.queries = queries
        self.strides = [8,16]
        self.spec = {'grid_size': cfg['grid_size'], 'context_scale': cfg['context_scale'],
                     'resize_short': cfg['resize_short'], 'resize_max': cfg['resize_max'],
                     'implementation': 'TEST ONLY deterministic synthetic encoder'}
        self.spec['id'] = digest(self.spec)
        self.calls = 0

    def encode_text(self, query):
        text = torch.zeros(2,16)
        text[:, self.queries.index(query)] = 3
        return text

    def features(self, path):
        with Image.open(path) as img:
            i = int(round(img.getpixel((0,0))[0]/30))
        features = [torch.zeros(1,16,10,15), torch.zeros(1,16,5,8)]
        for f in features:
            f[:, i] = 3
        return features, (120,80), (120,80)

    def tokens(self, features, boxes, resized, original):
        return region_tokens(features,self.strides,torch.tensor(boxes,dtype=torch.float32),resized,
                             self.cfg['grid_size'],self.cfg['context_scale'])

    def predict(self, path, query, check_hooks=False):
        self.calls += 1
        features, resized, original = self.features(path)
        boxes = torch.tensor([[20/120,30/80,20/120,20/80], [20/120,30/80,20/120,20/80],
                              [70/120,30/80,20/120,20/80], [100/120,60/80,10/120,10/80]])
        return {'features':features,'resized_size':resized,'original_size':original,
                'boxes_cxcywh_normalized':boxes,'native_scores':torch.tensor([.9,.8,.4,.3])}

    def proposal_tokens(self, prediction, selected):
        b = cxcywh_to_xyxy(prediction['boxes_cxcywh_normalized'][selected]).clamp(0,1)*torch.tensor([120,80,120,80])
        return self.tokens(prediction['features'], b.tolist(), (120,80),(120,80))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg,self.ds = fixture(self.root)

    def tearDown(self): self.temp.cleanup()

    def test_coordinate_convention_and_original_metadata(self):
        np.testing.assert_equal(self.ds.boxes('train_a.jpg')[0], [10,20,30,40])
        self.assertEqual(self.ds.size('train_a.jpg'), (120,80))
        self.assertEqual(self.ds.audit()['images_checked'], 6)

    def test_split_overlap_fails(self):
        aliases = self.root/'aliases.json'
        write_json(aliases, {'strawberries':'apples'})
        with self.assertRaisesRegex(ValueError,'overlap'):
            FSC147(dict(self.cfg, class_aliases=str(aliases)))

    def test_unknown_is_not_absent(self):
        path = self.root/'manifest.jsonl'
        summary = make_manifest(self.ds,'val',path,all_queries=True)
        self.assertEqual(summary, {'rows':4,'present':2,'absent':0,'unknown':2})
        for row in read_jsonl(path):
            if row['status']=='unknown':
                self.assertIsNone(row['gt_count']);self.assertIsNone(row['gt_points'])

    def test_overrides_verified_absence_and_present(self):
        override = self.root/'override.jsonl'
        write_jsonl(override,[{'image':'val_a.jpg','query':'raspberries','status':'absent', 'evidence':'synthetic fixture verified'},
                             {'image':'val_b.jpg','query':'strawberries','status':'present','points':[[100,60]],'evidence':'synthetic fixture supplemental'}])
        result=make_manifest(self.ds,'val',self.root/'m.jsonl',all_queries=True,overrides=override)
        self.assertEqual((result['present'],result['absent'],result['unknown']), (3,1,0))

    def test_heldout_query_cannot_be_region_train_negative(self):
        path=self.root/'regions.jsonl';make_region_review(self.ds,'train',path)
        row=list(read_jsonl(path))[0]
        row['labels'].append({'query':'cars','label':0,'evidence':'wrong split'})
        with self.assertRaisesRegex(ValueError,'outside train'):
            checked_region_labels(self.ds,row,'train')

    def test_pending_negative_ignored(self):
        path=self.root/'regions.jsonl';make_region_review(self.ds,'train',path)
        labels=checked_region_labels(self.ds,next(read_jsonl(path)),'train')
        self.assertEqual(len(labels),1);self.assertEqual(labels[0]['label'],1)

    def test_no_fabricated_absent_override(self):
        path=self.root/'override.jsonl'
        write_jsonl(path,[{'image':'val_a.jpg','query':'raspberries','status':'absent'}])
        with self.assertRaisesRegex(ValueError,'evidence'):
            make_manifest(self.ds,'val',self.root/'m.jsonl',overrides=path)

    def test_wrong_image_variant_rejected(self):
        Image.new('RGB',(12,8)).save(self.ds.image_path('train_a.jpg'))
        with self.assertRaisesRegex(ValueError,'outside actual image'):
            self.ds.validate_image('train_a.jpg')

    def test_regions_review_html_and_query_review_html(self):
        region=self.root/'r.jsonl';make_region_review(self.ds,'train',region)
        result=render_review(self.ds,region,self.root/'regions.html',limit=2)
        self.assertEqual(result['displayed'],2)
        text=(self.root/'regions.html').read_text()
        self.assertIn('data:image/jpeg;base64,',text);self.assertNotIn('__PAYLOAD__',text)
        manifest=self.root/'m.jsonl';make_manifest(self.ds,'val',manifest,all_queries=True)
        render_review(self.ds,manifest,self.root/'queries.html',kind='queries')
        self.assertIn('query_overrides.jsonl',(self.root/'queries.html').read_text())

    def test_cli_audit_and_manifest(self):
        cfg=dict(self.cfg,repo='.',checkpoint='missing.pth',bert_dir='.',output_dir='out')
        config=self.root/'config.json';write_json(config,cfg)
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            main(['--config',str(config),'audit'])
        self.assertEqual(json.loads(capture.getvalue())['images_checked'],6)


class GeometryModelTests(unittest.TestCase):
    def test_roi_position_not_swapped(self):
        y,x=torch.meshgrid(torch.arange(8),torch.arange(12),indexing='ij')
        feat=(x+100*y).float()[None,None]
        boxes=torch.tensor([[2.,1.,6.,5.]])
        tokens=region_tokens([feat],[1],boxes,(12,8),grid=1,context_scale=1)
        # aligned ROI center=(x=3.5,y=2.5) in feature indices -> 253.5
        torch.testing.assert_close(tokens[0,:,0],torch.tensor([253.5,253.5]))

    def test_actual_stride_with_padded_feature_size(self):
        y,x=torch.meshgrid(torch.arange(6),torch.arange(9),indexing='ij')
        feat=(x+100*y).float()[None,None]
        tokens=region_tokens([feat],[8],torch.tensor([[16.,8.,48.,40.]]),(67,43),grid=1,context_scale=1)
        torch.testing.assert_close(tokens[0,:,0],torch.tensor([253.5,253.5]))

    def test_context_clamps_to_image(self):
        out=expand_boxes(torch.tensor([[0.,0.,10.,10.]]),2,12,12)
        torch.testing.assert_close(out,torch.tensor([[0.,0.,12.,12.]]))

    def test_center_inside_not_nearest_other_box(self):
        model=RegionTextVerifier(dim=16,heads=2,layers=1,grid_size=3,dropout=0)
        self.assertEqual(model.center_indices.tolist(),[4,22])

    def test_padding_does_not_change_score(self):
        torch.manual_seed(2)
        model=RegionTextVerifier(dim=16,heads=2,layers=2,grid_size=3,dropout=0).eval()
        v=torch.randn(1,36,16); t=torch.randn(1,2,16)
        with torch.no_grad():
            a=model(v,t,torch.ones(1,2,dtype=torch.bool))
            b=model(v,torch.cat([t,torch.randn(1,3,16)*100],dim=1),torch.tensor([[1,1,0,0,0]],dtype=torch.bool))
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)

    def test_gradient_finite_and_cross_prompt_loss(self):
        logits=torch.tensor([2.,-2.,0.3,-0.4],requires_grad=True)
        labels=torch.tensor([1.,0.,1.,0.]);groups=torch.tensor([0,0,1,1])
        loss,terms=grounding_loss(logits,labels,groups)
        self.assertGreater(terms['counterfactual'].item(),0)
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        # Opposite labels in DIFFERENT regions must not become a counterfactual pair.
        _,terms=grounding_loss(logits[:2],labels[:2],torch.tensor([0,1]))
        self.assertEqual(terms['counterfactual'].item(),0)

    def test_empty_text_and_invalid_labels_fail(self):
        model=RegionTextVerifier(dim=16,heads=2,layers=1,grid_size=3)
        with self.assertRaisesRegex(ValueError,'content token'):
            model(torch.randn(1,36,16),torch.randn(1,2,16),torch.zeros(1,2,dtype=torch.bool))
        with self.assertRaisesRegex(ValueError,'0/1'):
            grounding_loss(torch.zeros(1),torch.tensor([-1.]),torch.zeros(1))

    def test_learns_matching_on_synthetic_pairs(self):
        torch.manual_seed(3)
        model=RegionTextVerifier(dim=16,heads=2,layers=1,grid_size=3,dropout=0)
        # Same physical ROI, two prompts; visual and text encode independent identities.
        visual=torch.zeros(8,36,16);text=torch.zeros(8,1,16)
        labels=[]
        for i,(a,b) in enumerate(itertools.product(range(2),repeat=2)):
            for j in range(2):
                idx=2*i+j;visual[idx,:,a]=3;text[idx,:,b]=3;labels.append(float(a==b))
        labels=torch.tensor(labels);groups=torch.tensor([0,0,0,0,1,1,1,1])
        opt=torch.optim.AdamW(model.parameters(),lr=.005)
        mask=torch.ones(8,1,dtype=torch.bool)
        first=None
        for step in range(60):
            logits=model(visual,text,mask);loss,_=grounding_loss(logits,labels,groups)
            if first is None:first=loss.item()
            opt.zero_grad();loss.backward();opt.step()
        self.assertLess(loss.item(),first*.3)
        self.assertTrue(((logits.detach()>0)==labels.bool()).all())


class MetricTests(unittest.TestCase):
    def test_duplicate_is_fp(self):
        self.assertEqual(point_match([[0,0],[0,0]],[[0,0]],1),(1,1,0))

    def test_radius_first_maximum_matching(self):
        # Plain minimum-distance assignment picks distances 0 + .8, losing one TP;
        # maximum-cardinality picks .6 + .6, giving two valid matches.
        p=[[0,0],[1/15,math.sqrt(.36-(1/15)**2)]]
        g=[[0,0],[.6,0]]
        self.assertEqual(point_match(p,g,.7),(2,0,0))

    def test_empty_and_absent(self):
        self.assertEqual(point_match([],[[1,1]],1),(0,0,1))
        self.assertEqual(point_match([[1,1]],[],1),(0,1,0))
        self.assertEqual(point_match([],[],1),(0,0,0))

    def test_perfect_count_wrong_positions(self):
        self.assertEqual(point_match([[50,50],[60,60]],[[1,1],[2,2]],1),(0,2,2))

    def test_agrees_with_exhaustive_max_cardinality(self):
        rng=np.random.default_rng(2)
        for _ in range(30):
            p=rng.random((3,2));g=rng.random((3,2));r=.4
            valid=np.linalg.norm(p[:,None]-g[None],axis=-1)<=r
            expected=max(sum(valid[i,j] for i,j in enumerate(perm)) for perm in itertools.permutations(range(3)))
            self.assertEqual(point_match(p,g,r)[0],expected)


class FullPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.cfg,self.ds=fixture(self.root)
        self.backend=FakeBackend(self.cfg,['apples','pears','strawberries','raspberries','cars','buses'])

    def tearDown(self):self.tmp.cleanup()

    def cache(self,split,negative=True):
        path=self.root/(split+'.jsonl');make_region_review(self.ds,split,path,negatives=1)
        rows=list(read_jsonl(path))
        if negative:
            for row in rows:
                row['labels'][1].update(label=0,evidence='synthetic fixture verified')
            write_jsonl(path,rows)
        out=self.root/(split+'_cache');build_cache(self.ds,self.backend,path,out,split)
        return out

    def test_cache_pending_guard_and_fingerprint(self):
        train=self.cache('train',False);val=self.cache('val')
        with self.assertRaisesRegex(ValueError,'positive AND negative'):
            validate_training_caches(RegionDataset(train),RegionDataset(val))
        index=read_json(val/'index.json');index['entries'][0]['labels'][0]['label']=0;write_json(val/'index.json',index)
        with self.assertRaisesRegex(ValueError,'modified'):
            RegionDataset(val)

    def test_train_save_resume_and_inference_evaluation(self):
        train,val=self.cache('train'),self.cache('val')
        ckpt=self.root/'ckpt'
        train_verifier(self.cfg,train,val,ckpt,epochs=2,batch_regions=2,lr=.001)
        before=load_tensor_file(ckpt/'last.pt');self.assertEqual(before['epoch'],1)
        train_verifier(self.cfg,train,val,ckpt,epochs=3,batch_regions=2,lr=.001,resume=ckpt/'last.pt')
        self.assertEqual(load_tensor_file(ckpt/'last.pt')['epoch'],2)
        manifest=self.root/'manifest.jsonl';make_manifest(self.ds,'test',manifest,all_queries=True)
        predictions=self.root/'predictions'
        predict_manifest(self.ds,self.backend,manifest,predictions,verifier=ckpt/'best.pt')
        result=evaluate(self.ds,manifest,predictions,self.root/'reports')
        self.assertEqual(result['metrics']['baseline']['positive_mae'],2)
        self.assertEqual(result['metrics']['baseline']['unknown_excluded_from_accuracy'],2)
        self.assertIsNone(result['metrics']['baseline']['absent_mae'])
        self.assertEqual(result['metrics']['baseline']['tp'],4)
        self.assertEqual(result['metrics']['baseline']['fp'],4)
        calls=self.backend.calls
        result=predict_manifest(self.ds,self.backend,manifest,predictions,verifier=ckpt/'best.pt')
        self.assertEqual(result['reused'],4);self.assertEqual(self.backend.calls,calls)

    def test_resume_matches_uninterrupted_epoch_training(self):
        train,val=self.cache('train'),self.cache('val')
        a,b=self.root/'a',self.root/'b'
        train_verifier(self.cfg,train,val,a,epochs=3,batch_regions=2,lr=.001)
        train_verifier(self.cfg,train,val,b,epochs=1,batch_regions=2,lr=.001)
        train_verifier(self.cfg,train,val,b,epochs=3,batch_regions=2,lr=.001,resume=b/'last.pt')
        sa,sb=load_tensor_file(a/'last.pt'),load_tensor_file(b/'last.pt')
        for key in sa['state_dict']:
            torch.testing.assert_close(sa['state_dict'][key],sb['state_dict'][key],rtol=0,atol=0)

    def test_partial_evaluation_requires_explicit_flag(self):
        manifest=self.root/'manifest.jsonl';make_manifest(self.ds,'val',manifest,all_queries=True)
        predict_manifest(self.ds,self.backend,manifest,self.root/'predictions',limit=1)
        with self.assertRaisesRegex(ValueError,'missing'):
            evaluate(self.ds,manifest,self.root/'predictions',self.root/'report')
        report=evaluate(self.ds,manifest,self.root/'predictions',self.root/'report',allow_partial=True)
        self.assertTrue(report['partial']);self.assertEqual(report['missing_queries'],3)

    def test_absent_accuracy_requires_verified_annotations(self):
        override=self.root/'override.jsonl'
        write_jsonl(override,[{'image':'val_a.jpg','query':'raspberries','status':'absent','evidence':'verified synthetic'}])
        manifest=self.root/'manifest.jsonl';make_manifest(self.ds,'val',manifest,all_queries=True,overrides=override)
        predict_manifest(self.ds,self.backend,manifest,self.root/'predictions')
        report=evaluate(self.ds,manifest,self.root/'predictions',self.root/'report')
        self.assertEqual(report['metrics']['baseline']['verified_absent'],1)
        self.assertEqual(report['metrics']['baseline']['absent_mae'],4)
        self.assertEqual(report['metrics']['baseline']['absent_false_count_rate'],1)


if __name__=='__main__':unittest.main()
