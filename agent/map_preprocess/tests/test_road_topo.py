"""掩码拓扑：算法的合成场景断言，以及接入 Node 3 后输出给下游的数据格式。"""
from __future__ import annotations

import io
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml
from PIL import Image

from agent.map_preprocess.global_config import GlobalPreprocessConfig
from agent.map_preprocess.io import read_json, relative_path
from agent.map_preprocess.nodes.global_pipeline import clean_road_graph
from agent.map_preprocess.tests import road_topo_scene as synthetic
from agent.map_preprocess.utils import road_topo as rt

PIPELINE = Path(__file__).resolve().parents[1]/'pipeline.yaml'


class SyntheticSceneTest(unittest.TestCase):
    def test_every_scenario_passes(self):
        scene = synthetic.make_scene()
        semantic, road = scene.semantic, scene.R
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            cfg = rt.load_config(overrides={'classes': {'parking_ids': [synthetic.PARKING]}, 'preprocess': {'road_thresh': .35}})
            graph, records, _ = rt.run_arrays(road, semantic, synthetic.GSD, folder, cfg)
            lines = synthetic.check(graph, records)
        self.assertEqual([line for line in lines if line.startswith('FAIL')], [])
        self.assertGreaterEqual(len(lines), 40)


class CleanRoadGraphTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        self.size = 800
        Image.new('RGB', (self.size, self.size), (90, 90, 90)).save(root/'image.png')
        (root/'cache').mkdir()
        # 概率图是 1/4 原图的粗网格，检查上采样；竖路和横路在语义里都是道路
        coarse = np.zeros((self.size//4, self.size//4), np.float32)
        cv2.line(coarse, (10, 100), (190, 100), 1., 5)
        cv2.line(coarse, (100, 10), (100, 190), 1., 5)
        np.save(root/'cache'/'road_probability.npy', coarse)
        self.labels = np.full((self.size, self.size), 2, np.uint8)
        self.labels[cv2.resize(coarse, (self.size, self.size), interpolation=cv2.INTER_NEAREST) > .5] = 0
        output = root/'task'
        output.mkdir()
        self.state = {'task_id': 't', 'output_dir': relative_path(output), 'image_path': relative_path(root/'image.png'), 'resolution_m_per_px': .5,
                      'image': {'path': relative_path(root/'image.png'), 'width': self.size, 'height': self.size},
                      'road_graph_cache': {'path': relative_path(root/'cache'/'raw_graph.json')}}
        self.config = GlobalPreprocessConfig(segmenter='dummy', output_dir=str(root/'out'), cache_dir=str(root/'cache'), road_topo_enabled=True, max_tile_pixels=10**9,
                                             fallback_resolution_m_per_px=.5, road_max_shared_gap_m=10.)

    def run_node(self) -> dict:
        with patch('agent.map_preprocess.nodes.semantic_roads.whole_image_semantic', return_value=(self.labels, None, {'cache_hit': True})), redirect_stdout(io.StringIO()):
            return clean_road_graph(self.state, config=self.config)

    def test_output_matches_downstream_contract(self):
        update = self.run_node()
        cleaned = read_json(update['clean_road_graph_path'])
        nodes = {n['node_id']: n for n in cleaned['nodes']}
        self.assertEqual(sorted(n['node_type'] for n in nodes.values()), ['endpoint']*4+['junction'])
        self.assertEqual(len(cleaned['edges']), 4)
        for edge in cleaned['edges']:
            self.assertEqual(edge['polyline_xy'][0], nodes[edge['u']]['xy'])
            self.assertEqual(edge['polyline_xy'][-1], nodes[edge['v']]['xy'])
            self.assertTrue(all(0 <= v <= self.size for p in edge['polyline_xy'] for v in p))
            self.assertGreater(edge['length_m'], 0)
            self.assertIn(edge['road_topo_source'], ('core', 'filled'))
            self.assertIn('edge_score', edge)
            self.assertIsInstance(edge['edge_id'], str)
        junction = next(n for n in nodes.values() if n['node_type'] == 'junction')
        self.assertLess(max(abs(junction['xy'][0]-402), abs(junction['xy'][1]-402)), 6)
        self.assertIn('road_topo', cleaned['filter_stats'])
        task = Path(self.folder.name)/'task'
        self.assertTrue((task/'visualization'/'01b_road_topo'/'summary.png').is_file())
        self.assertTrue((task/'state'/'road_topo'/'graph.pkl').is_file())

    def test_low_score_filter_does_not_remove_topology_edges(self):
        self.config = GlobalPreprocessConfig(**{**self.config.to_dict(), 'min_edge_score': 1.})
        cleaned = read_json(self.run_node()['clean_road_graph_path'])
        self.assertEqual(len(cleaned['edges']), 4)

    def test_missing_probability_is_a_clear_error(self):
        (Path(self.folder.name)/'cache'/'road_probability.npy').unlink()
        with self.assertRaisesRegex(ValueError, 'road_probability'):
            self.run_node()


    def run_cached(self, task: str) -> tuple[dict, int]:
        """带 MaGRoad/语义缓存键运行一次，返回 (节点输出, run_arrays 实际调用次数)。"""
        output = Path(self.folder.name)/task
        output.mkdir()
        self.state = {**self.state, 'output_dir': relative_path(output), 'road_graph_cache': {**self.state['road_graph_cache'], 'key': 'magroad', 'enabled': True}}
        with patch('agent.map_preprocess.nodes.semantic_roads.semantic_cache_key', return_value=('semantic', 'model', (self.size, self.size), 1.)), \
                patch.object(rt, 'run_arrays', wraps=rt.run_arrays) as runs:
            update = self.run_node()
        return update, runs.call_count

    def test_topology_cache_skips_recompute(self):
        first, calls = self.run_cached('a')
        self.assertEqual(calls, 1)
        second, calls = self.run_cached('b')
        self.assertEqual(calls, 0)   # 第二次直接读缓存
        one, two = read_json(first['clean_road_graph_path']), read_json(second['clean_road_graph_path'])
        self.assertEqual(one['edges'], two['edges'])
        self.assertEqual((one['filter_stats']['road_topo']['cache_hit'], two['filter_stats']['road_topo']['cache_hit']), (False, True))
        task = Path(self.folder.name)/'b'
        self.assertTrue((task/'visualization'/'01b_road_topo'/'summary.png').is_file())
        self.assertTrue((task/'state'/'road_topo'/'graph.pkl').is_file())

    def test_topology_cache_invalidates_on_settings_and_respects_switch(self):
        self.run_cached('a')
        # road_topo 参数变化：键不同，重新计算
        changed = Path(self.folder.name)/'topo.yaml'
        values = yaml.safe_load(PIPELINE.read_text(encoding='utf-8'))
        values['road_topo']['steps']['refine_curvature'] = not values['road_topo']['steps']['refine_curvature']
        changed.write_text(yaml.safe_dump(values, allow_unicode=True), encoding='utf-8')
        self.config = GlobalPreprocessConfig(**{**self.config.to_dict(), 'road_topo_config': str(changed)})
        self.assertEqual(self.run_cached('b')[1], 1)
        # 关闭缓存：每次都算，也不读旧缓存
        self.config = GlobalPreprocessConfig(**{**self.config.to_dict(), 'cache_enabled': False})
        self.assertEqual(self.run_cached('c')[1], 1)
        self.assertEqual(self.run_cached('d')[1], 1)

    def test_corrupted_topology_cache_is_recomputed(self):
        self.run_cached('a')
        for path in (Path(self.folder.name)/'cache'/'road_topo').glob('*/result.pkl'):
            path.write_bytes(b'broken')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertEqual(self.run_cached('b')[1], 1)


class AreaRegionTests(unittest.TestCase):
    def test_distance_transform_matches_disk_dilation(self):
        import math
        from types import SimpleNamespace
        rng = np.random.default_rng(0)
        mask = np.zeros((600, 600), np.uint8)
        for _ in range(12):
            x, y = (int(v) for v in rng.integers(0, 600, 2))
            cv2.line(mask, (x, 0), (600-y, 599), 1, int(rng.integers(5, 120)))
        gsd, width = .05, 4.
        cfg = SimpleNamespace(area=SimpleNamespace(max_road_width_m=width), classes=SimpleNamespace(parking_ids=[]))
        labels, count = rt.mark_area_regions(mask, np.zeros_like(mask), cfg, gsd)
        half = width/2/gsd
        seed = (cv2.distanceTransform(mask, cv2.DIST_L2, 5) > half).astype(np.uint8)
        size = 2*int(math.ceil(half))+1
        old = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) & mask
        self.assertGreater(count, 0)
        # 只在圆盘离散边缘差零星像元（实测约 0.07%）
        self.assertLess(np.count_nonzero((labels > 0) != (old > 0)), old.sum()*.005)

if __name__ == '__main__':
    unittest.main()
