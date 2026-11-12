"""独立深度和用户语义模型适配器的离线契约回归。"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agent.paths import PROJECT_ROOT
from agent.perception_config import (DepthAnythingConfig,load_perception_config)
from percection_tool.depth_anything import DepthAnythingEstimator


class AnalysisToolsTests(unittest.TestCase):
    def test_lazy_isolated_imports_and_configuration(self):
        code='''import sys
from percection_tool.factory import create_depth_estimator,create_semantic_segmenter
assert 'percection_tool.depth_anything.inference' not in sys.modules
a=create_depth_estimator()
b=create_semantic_segmenter()
assert a.model is None and b.processor is None
assert 'torch' not in sys.modules
assert 'agent.map_preprocess.graph' not in sys.modules
assert 'agent.vlm.client' not in sys.modules
'''
        subprocess.run([sys.executable,'-c',code],cwd=PROJECT_ROOT,check=True)
        settings=load_perception_config('agent/perception_tools.yaml')
        self.assertIn('large',settings.depth.weights_dir)
        self.assertEqual(settings.segearth_ov3.stride,1200)
        for change in ({'input_size':512},{'max_input_pixels':0},{'device':'invalid'}):
            with self.assertRaises(ValueError):
                DepthAnythingConfig(**change)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.yaml'
            p.write_text('depth:\n  unknown: true\n')
            with self.assertRaises(ValueError):
                load_perception_config(str(p))

    def test_depth_geotiff_keeps_georeference(self):
        import rasterio
        from rasterio.transform import from_origin
        from agent import run_depth_geotiff as tool
        seen={}
        class Fake:
            def __init__(self,config):
                seen['config']=config
            def predict(self,image,valid_mask=None):
                return {'depth':np.arange(48,dtype=np.float32).reshape(6,8),'metadata':{}}
            def close(self):
                seen['closed']=True
        transform=from_origin(8.38,48.98,1e-6,5e-7)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with rasterio.open(root/'in.tif','w',driver='GTiff',width=8,height=6,count=3,dtype='uint8',crs='EPSG:4326',transform=transform) as target:
                target.write(np.zeros((3,6,8),np.uint8))
            with patch.object(tool,'create_depth_estimator',Fake):
                path=tool.run(['--image',str(root/'in.tif'),'--output',str(root/'out.tif')])
            with rasterio.open(path) as result:
                self.assertEqual(result.crs.to_epsg(),4326)
                self.assertEqual(result.transform,transform)
                self.assertEqual((result.width,result.height,result.count,result.dtypes[0]),(8,6,1,'float32'))
                np.testing.assert_array_equal(result.read(1),np.arange(48,dtype=np.float32).reshape(6,8))
            with rasterio.open(root/'plain.tif','w',driver='GTiff',width=8,height=6,count=3,dtype='uint8') as target:
                target.write(np.zeros((3,6,8),np.uint8))
            with self.assertRaises(ValueError):
                tool.run(['--image',str(root/'plain.tif'),'--output',str(root/'x.tif')])
        self.assertTrue(seen['closed'])
        self.assertEqual(seen['config'].max_input_pixels,16_000_000)
        self.assertEqual(seen['config'].input_size,tool.DEFAULT_INPUT_SIZE)

    def semantic(self,model):
        tool=SemanticSegmenter()
        tool.model=model
        tool.device=torch.device('cpu')  # 测试替身不执行真实 CUDA 导出模型。
        tool.metadata={'classes':['road','building','plant'],'palette_rgb':[[255,255,255],[0,0,255],[0,255,0]],'precision':'test'}
        tool.source={'files':{'segmentation.pt':{'sha256':'test'}}}
        return tool

    def depth(self):
        tool=DepthAnythingEstimator()
        tool.device=torch.device('cpu')
        tool.processor=lambda **kwargs:{'pixel_values':torch.ones((1,3,14,14))}
        tool.model=lambda **kwargs:SimpleNamespace(predicted_depth=torch.arange(196,dtype=torch.float32).reshape(1,14,14))
        tool.load_report={'missing_keys':[]}
        return tool

    def test_depth_original_grid_relative_units_and_invalid_pixels(self):
        tool=self.depth()
        rgb=np.zeros((70,95,3),np.uint8)
        valid=np.ones((70,95),bool)
        valid[0]=False
        out=tool.predict(rgb,valid)
        self.assertEqual(out['depth'].shape,(70,95))
        self.assertEqual(out['depth'].dtype,np.float32)
        self.assertTrue(np.isnan(out['depth'][0]).all())
        self.assertTrue(np.isfinite(out['depth'][valid]).all())
        self.assertEqual(out['metadata']['representation'],'relative_inverse_depth')
        self.assertFalse(out['metadata']['is_metric'])
        self.assertFalse(out['metadata']['is_elevation'])
        self.assertEqual(out['metadata']['larger_is'],'nearer')
        tool.model=lambda **kwargs:SimpleNamespace(predicted_depth=torch.full((1,14,14),float('nan')))
        with self.assertRaisesRegex(RuntimeError,'非有限'):
            tool.predict(rgb)

    def test_input_limits_and_missing_weights_do_not_start_models(self):
        for tool in (DepthAnythingEstimator(DepthAnythingConfig(max_input_pixels=10)),):
            with self.assertRaisesRegex(ValueError,'max_input_pixels'):
                tool.predict(np.zeros((5,5,3),np.uint8))
            with self.assertRaises(ValueError):
                tool.predict(np.zeros((3,3,3),np.float32))
            self.assertIsNone(tool.model)
        with tempfile.TemporaryDirectory() as tmp:
            tool=DepthAnythingEstimator(DepthAnythingConfig(weights_dir=tmp))
            with self.assertRaisesRegex(ValueError,'校验失败'):
                tool.predict(np.zeros((20,20,3),np.uint8))
        tool=DepthAnythingEstimator()
        with self.assertRaisesRegex(ValueError,'处理网格'):
            tool.predict(np.zeros((1,10000,3),np.uint8))

    def test_depth_segmented_download_resumes_and_rejects_wrong_ranges(self):
        import hashlib,io
        from percection_tool.depth_anything.download import _download_large,verify
        data=(bytes(range(256))*9000)
        class Response(io.BytesIO):
            status=206
            def __init__(inner,content,header):
                super().__init__(content)
                inner.headers={'Content-Range':header}
        def response(request,timeout):
            start,stop=map(int,request.get_header('Range').split('=')[1].split('-'))
            return Response(data[start:stop+1],f'bytes {start}-{stop}/{len(data)}')
        with tempfile.TemporaryDirectory() as tmp:
            part=Path(tmp)/'model.part'
            part.write_bytes(data[:3333])
            # 旧的较大分段缓存，只缺最后123字节，转换后应从已下载处续传。
            old=part.with_name(part.name+'.3333.chunk')
            old.write_bytes(data[3333:-123])
            with patch('urllib.request.urlopen',side_effect=response) as network:
                _download_large('https://example.invalid/model?download=true',part,len(data))
            self.assertLessEqual(network.call_count,2)
            self.assertEqual(part.read_bytes(),data)
            self.assertTrue(verify(part,{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}))
            part.write_bytes(b'')
            with patch('urllib.request.urlopen',side_effect=lambda *a,**k:Response(b'', 'bytes 1-2/3')):
                with self.assertRaisesRegex(ValueError,'分段范围'):
                    _download_large('https://example.invalid/model?download=true',part,10)
            self.assertEqual(part.read_bytes(),b'')


if __name__=='__main__':
    unittest.main()
