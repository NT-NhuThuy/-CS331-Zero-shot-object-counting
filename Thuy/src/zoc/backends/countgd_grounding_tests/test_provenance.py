import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from PIL import Image
import torch
from test_pipeline import fixture, FakeBackend
from cgground.io import read_json, write_json, read_jsonl, write_jsonl, load_tensor_file, atomic_torch_save
from cgground.fsc import FSC147
from cgground.protocol import make_manifest, make_region_review
from cgground.cache import build_cache, RegionDataset
from cgground.train import train_verifier
from cgground.infer import predict_manifest
from cgground.metrics import evaluate


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.cfg,self.ds=fixture(self.root)
        self.backend=FakeBackend(self.cfg,['apples','pears','strawberries','raspberries','cars','buses'])
    def tearDown(self):self.tmp.cleanup()

    def cache(self,split):
        review=self.root/(split+'.jsonl');make_region_review(self.ds,split,review,negatives=1)
        rows=list(read_jsonl(review))
        for row in rows:row['labels'][1].update(label=0,evidence='verified synthetic')
        write_jsonl(review,rows);out=self.root/(split+'_cache')
        with contextlib.redirect_stdout(io.StringIO()):build_cache(self.ds,self.backend,review,out,split)
        return out

    def test_floating_point_boundary_tolerance(self):
        p=self.root/'annotation_FSC147_384.json';a=read_json(p)
        a['train_a.jpg']['points'][0]=[120+1e-13,30];write_json(p,a)
        ds=FSC147(self.cfg);ds.validate_image('train_a.jpg')
        make_manifest(ds,'train',self.root/'manifest.jsonl')

    def test_feature_corruption_rejected(self):
        cache=self.cache('train');data=RegionDataset(cache)
        p=cache/data.entries[0]['feature'];saved=load_tensor_file(p);saved['visual']+=1
        atomic_torch_save(p,saved)
        with self.assertRaisesRegex(ValueError,'checksum'):data[0]

    def test_image_modified_after_inference_rejected(self):
        manifest=self.root/'manifest.jsonl';make_manifest(self.ds,'val',manifest)
        with contextlib.redirect_stdout(io.StringIO()):predict_manifest(self.ds,self.backend,manifest,self.root/'pred')
        Image.new('RGB',(120,80),(0,0,0)).save(self.ds.image_path('val_a.jpg'))
        with self.assertRaisesRegex(ValueError,'Image bytes changed'):
            evaluate(self.ds,manifest,self.root/'pred',self.root/'report')

    def test_wrong_configuration_and_heldout_cache_cannot_train(self):
        train,val=self.cache('train'),self.cache('val')
        with self.assertRaisesRegex(ValueError,'Expected train'):
            train_verifier(self.cfg,val,train,self.root/'run')
        wrong=dict(self.cfg,context_scale=2)
        with self.assertRaisesRegex(ValueError,'Config/cache mismatch'):
            train_verifier(wrong,train,val,self.root/'run')

    def test_resume_configuration_change_rejected(self):
        train,val=self.cache('train'),self.cache('val')
        with contextlib.redirect_stdout(io.StringIO()):
            train_verifier(self.cfg,train,val,self.root/'run',epochs=1,batch_regions=2)
        with self.assertRaisesRegex(ValueError,'Resume configuration'):
            train_verifier(self.cfg,train,val,self.root/'run',epochs=2,batch_regions=2,lr=.002,resume=self.root/'run/last.pt')

    def test_inference_run_directory_cannot_mix_weights(self):
        manifest=self.root/'manifest.jsonl';make_manifest(self.ds,'val',manifest)
        out=self.root/'pred'
        with contextlib.redirect_stdout(io.StringIO()):predict_manifest(self.ds,self.backend,manifest,out)
        self.backend.spec=dict(self.backend.spec,id='changed-spec')
        with self.assertRaisesRegex(ValueError,'different run'):
            predict_manifest(self.ds,self.backend,manifest,out)


if __name__=='__main__':unittest.main()
