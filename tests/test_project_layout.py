"""目录迁移契约：旧产物可读，新的入口与服务路径仍指向项目根。"""
from pathlib import Path
import os
import subprocess
import sys
import unittest
from unittest import mock

from agent.paths import PROJECT_ROOT, resolve_path
from agent.map_preprocess.config import load_config
from agent.vlm.runtime import VLMRuntime


class ProjectLayoutTests(unittest.TestCase):
    def test_legacy_relative_and_absolute_paths_resolve_to_same_resource(self):
        for old, new in (
            ('map_preprocess/output/a/manifest.json', 'output/map_preprocess/a/manifest.json'),
            ('percection_tool/weights/model.ckpt', 'percection_tool/sam_road/weights/model.ckpt'),
            ('map_preprocess/vendor/sam_road/model.py', 'percection_tool/sam_road/vendor/model.py'),
            ('map_preprocess/weights/model.ckpt', 'percection_tool/sam_road/weights/model.ckpt'),
            ('vlm/runtime/a/state.json', 'output/vlm/runtime/a/state.json'),
            ('kitti_raw_0930_0027.tif', 'data/input/kitti_raw_0930_0027.tif'),
            ('data/河西遥感/hexi_dom_0.05m.tif', 'input_geo_img/hexi_dom.tif'),
            ('data/河西遥感/hexi_dsm.tif', 'input_geo_img/hexi_dsm.tif'),
            ('data/Kitti遥感数据/kitti_raw_1003_0034.tif', 'input_geo_img/kitti_1003_0034.tif'),
        ):
            with self.subTest(old=old):
                self.assertEqual(resolve_path(old), PROJECT_ROOT/new)
                self.assertEqual(resolve_path(PROJECT_ROOT/old), PROJECT_ROOT/new)
        self.assertEqual(resolve_path('/tmp/map_preprocess/a'), Path('/tmp/map_preprocess/a'))
        self.assertEqual(resolve_path('qwen3827b-exl3-dflash2'), PROJECT_ROOT/'qwen3827b-exl3-dflash2')

    def test_configs_use_perception_weights_and_unified_outputs(self):
        for path in ('agent/map_preprocess/pipeline.yaml', None):
            settings = load_config(path)
            self.assertEqual(settings.segmenter, 'magroad_scene')
            self.assertTrue(settings.output_dir.startswith('output/'))
        self.assertFalse((PROJECT_ROOT/'percection_tool'/'pathfinder').exists())

    def test_perception_import_does_not_load_agent_graph_or_vlm(self):
        code = '''import sys
import percection_tool.factory
assert 'agent.map_preprocess.graph' not in sys.modules
assert 'agent.vlm.client' not in sys.modules
assert 'exllamav3' not in sys.modules
'''
        subprocess.run([sys.executable, '-c', code], cwd=PROJECT_ROOT, check=True)

    def test_factory_imports_only_selected_model(self):
        code = """import sys
from percection_tool.factory import create_graph_extractor
from agent.map_preprocess import GlobalPreprocessConfig
assert 'percection_tool.magroad.graph_adapter' not in sys.modules
create_graph_extractor(GlobalPreprocessConfig(segmenter='dummy'), 1.0, '/tmp')
assert 'percection_tool.magroad.graph_adapter' not in sys.modules
assert 'torch' not in sys.modules
"""
        subprocess.run([sys.executable, '-c', code], cwd=PROJECT_ROOT, check=True)

    def test_runtime_uses_exl3_model_project(self):
        with mock.patch.dict(os.environ, {'VLM_MAX_NUM_SEQS': '1'}, clear=True):
            runtime = VLMRuntime('http://127.0.0.1:8011/v1', 'local-qwen')
            try:
                with mock.patch('importlib.util.find_spec', return_value=object()):
                    command, env = runtime._server_command(8011)
            finally:
                runtime.close()
        model_root = PROJECT_ROOT/'qwen3827b-exl3-dflash2'
        self.assertEqual(command[0], sys.executable)   # Agent 与 VLM 服务共用同一个环境
        self.assertEqual(command[1], str(PROJECT_ROOT/'agent/vlm/exl3_server.py'))
        self.assertEqual(command[command.index('--target')+1], str(model_root/'models/qwen38-27b-exl3'))
        self.assertEqual(command[command.index('--draft')+1], str(model_root/'models/dflash2-exl3'))
        self.assertEqual(runtime.log_dir, PROJECT_ROOT/'output/vlm/log')
        self.assertNotIn('PYTHONPATH', env)

if __name__ == '__main__':
    unittest.main()
