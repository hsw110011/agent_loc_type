"""run_until：按配置跑到指定节点；pipeline.yaml 能读取。"""
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path

from PIL import Image

from agent.map_preprocess.config import load_config
from agent.map_preprocess.global_config import GlobalPreprocessConfig
from agent.map_preprocess.graph import run_pipeline
from agent.map_preprocess.io import resolve_path
from agent.map_preprocess.utils import road_topo

PIPELINE = Path(__file__).resolve().parents[1]/'pipeline.yaml'


class RunUntilTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        Image.new('RGB', (600, 600), (90, 90, 90)).save(self.root/'image.png')
        self.config = GlobalPreprocessConfig(segmenter='dummy', output_dir=str(self.root/'out'), cache_dir=str(self.root/'cache'), fallback_resolution_m_per_px=.5,
                                             min_edge_score=0, dedup_enabled=False, tile_grid=1)

    def snapshots(self, config) -> tuple[dict, list[str]]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            state = run_pipeline(str(self.root/'image.png'), config)
        return state, sorted(p.stem for p in (resolve_path(state['output_dir'])/'state'/'node_states').glob('*.json'))

    def test_stops_after_chosen_node(self):
        state, nodes = self.snapshots(replace(self.config, run_until='clean_road_graph'))
        self.assertEqual(nodes, ['01_load_image_and_resolution', '02_extract_road_graph', '03_clean_road_graph'])
        self.assertNotIn('manifest_path', state)

    def test_dom_discovers_paired_dsm_and_explicit_path_overrides(self):
        import numpy as np
        from agent.map_preprocess.tests.test_block_features import write_raster

        image = self.root/'scene_dom.tif'
        Image.new('RGB', (20, 20), (90, 90, 90)).save(image)
        height = np.full((20, 20), 10., np.float32)
        paired = write_raster(self.root/'scene_dsm.tif', height)
        other = write_raster(self.root/'override.tif', height)
        config = replace(self.config, run_until='load_image_and_resolution')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            state = run_pipeline(str(image), config)
            self.assertEqual(resolve_path(state['image']['dsm']['path']), paired)
            state = run_pipeline(str(image), config, dsm_path=str(other))
            self.assertEqual(resolve_path(state['image']['dsm']['path']), other)
            paired.unlink()
            state = run_pipeline(str(image), config)
            self.assertNotIn('dsm', state['image'])

    def test_stop_at_blocks_still_refreshes_manifest(self):
        state, nodes = self.snapshots(replace(self.config, run_until='build_blocks_and_groups'))
        self.assertEqual(nodes[-2:], ['04_build_blocks_and_groups', '05_refresh_global_manifest'])
        self.assertIn('manifest_path', state)
        # 算法版本写入 state 与 manifest，skill 据此不复用旧版本的结果
        from agent.map_preprocess.global_config import PIPELINE_VERSION
        from agent.map_preprocess.io import read_json
        self.assertEqual(state['pipeline_version'], PIPELINE_VERSION)
        self.assertEqual(read_json(state['manifest_path'])['pipeline_version'], PIPELINE_VERSION)
        # 停在 Node 4：没有判断退化，最终结论必须是不完整，而不是无退化
        self.assertEqual(state['final_summary']['conclusion'], 'incomplete')
        self.assertEqual(read_json(state['manifest_path'])['final_summary'], state['final_summary'])
        self.assertEqual(read_json(resolve_path(state['output_dir'])/'state'/'state.json')['final_summary'], state['final_summary'])

    def test_vlm_client_is_shared_only_when_safe(self):
        from unittest import mock
        from agent.map_preprocess.graph import shared_vlm_client
        full = replace(self.config, block_features_enabled=True, vlm_review_enabled=True, vlm_expansion_enabled=True,
                       block_size_m=64, degradation_fusion_enabled=True, vlm_localization_enabled=True)
        with mock.patch('agent.vlm.client.VLMClient') as client:
            self.assertEqual(shared_vlm_client(full), (client.return_value, True))
            injected = object()
            self.assertEqual(shared_vlm_client(full, vlm_client=injected), (injected, False))
            # 只有一个 VLM 节点、不缓存语义、注入了分割模型、跑不到 VLM 节点时都不共用
            self.assertEqual(shared_vlm_client(replace(full, vlm_expansion_enabled=False, degradation_fusion_enabled=False,
                                                       vlm_localization_enabled=False)), (None, False))
            self.assertEqual(shared_vlm_client(replace(full, cache_enabled=False)), (None, False))
            self.assertEqual(shared_vlm_client(full, semantic_segmenter=object()), (None, False))
            self.assertEqual(shared_vlm_client(replace(full, run_until='vlm_review_blocks')), (None, False))

    def test_no_limit_runs_everything_enabled(self):
        _, nodes = self.snapshots(self.config)
        self.assertEqual(nodes[-1], '05_refresh_global_manifest')

    def test_unknown_node_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'run_until'):
            replace(self.config, run_until='no_such_node')


class PipelineFileTests(unittest.TestCase):
    def test_pipeline_yaml_loads_with_full_flow_and_topology(self):
        config = load_config(PIPELINE)
        self.assertEqual(config.run_until, 'recommend_localization')
        self.assertTrue(config.road_topo_enabled and config.vlm_expansion_enabled and config.degradation_fusion_enabled and config.vlm_localization_enabled)
        self.assertEqual(road_topo.load_config(PIPELINE).preprocess.road_thresh, .18)

    def test_each_listed_option_is_valid(self):
        options = [line.split(':')[1].split('#')[0].strip() for line in PIPELINE.read_text(encoding='utf-8').splitlines() if 'run_until:' in line and line.lstrip('# ').startswith('run_until')]
        self.assertGreaterEqual(len(options), 6)
        for option in options:
            GlobalPreprocessConfig(run_until=option)

    def test_two_uncommented_lines_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'c.yaml'
            path.write_text('segmenter: dummy\nrun_until: clean_road_graph\nrun_until: extract_road_graph\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '只能保留一行'):
                load_config(path)


if __name__ == '__main__':
    unittest.main()
