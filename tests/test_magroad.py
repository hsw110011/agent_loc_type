"""MaGRoad 适配的离线检查：源码来源、模型结构、滑窗与取点逻辑。不需要权重或 GPU。"""
from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np

from agent.paths import PROJECT_ROOT
from percection_tool.magroad.inference import (CHECKPOINTS, extract_points, load_config, neighbor_queries,
                                               nms_points, patch_starts, upstream_stride)

VENDOR = PROJECT_ROOT/'percection_tool/magroad/vendor'


class VendorTest(unittest.TestCase):
    def test_source_hashes(self):
        record = json.loads((VENDOR/'SOURCE.json').read_text())
        for name, digest in record['source_hashes'].items():
            self.assertEqual(hashlib.sha256((VENDOR/name).read_bytes()).hexdigest(), digest, name)

    def test_generated_model_is_current(self):
        from percection_tool.magroad.vendor.build_inference_model import generate
        self.assertEqual((VENDOR/'model.py').read_text(encoding='utf-8'), generate())

    def test_models_build_with_geometric_features(self):
        import torch
        from percection_tool.magroad.vendor.model import MaGRoad
        torch.manual_seed(0)
        for name, spec in CHECKPOINTS.items():
            config = load_config(spec)
            model = MaGRoad(config).eval()
            keys = model.state_dict().keys()
            # 三个官方权重的 MaGTopoNet 都含 geo_proj；spacenet 由显式覆盖打开，且公开权重带点特征。
            self.assertIn('topo_net.geo_proj.weight', keys, name)
            self.assertEqual('topo_net.node_proj.weight' in keys, name == 'spacenet', name)
            size = config.PATCH_SIZE
            if size > 256:
                continue
            with torch.inference_mode():
                embedding, logits, scores = model.infer_masks_and_img_features(torch.rand(1, size, size, 3)*255)
                points = torch.tensor([[[10., 10.], [40., 12.], [80., 30.]]])
                pairs = torch.tensor([[[[0, 1], [0, 2]], [[1, 0], [1, 2]], [[2, 1], [2, 0]]]])
                valid = torch.tensor([[[True, True], [True, False], [True, True]]])
                topo = model.infer_toponet(embedding, points, pairs, valid, logits)
            self.assertEqual(tuple(scores.shape), (1, size, size, 2))
            self.assertEqual(tuple(topo.shape), (1, 3, 2, 1))
            self.assertTrue(torch.isfinite(topo).all())

    def test_threshold_overrides(self):
        from agent.run_magroad import parse_variant
        from percection_tool.magroad import MaGRoadExtractor
        self.assertEqual(parse_variant('spacenet'), ('spacenet', 'spacenet', {}))
        name, label, overrides = parse_variant('globalscale:ROAD=0.1,itsc_threshold=0.05')
        self.assertEqual((name, label, overrides), ('globalscale', 'globalscale_road0.1_itsc0.05', {'ROAD': 0.1, 'ITSC': 0.05}))
        tool = MaGRoadExtractor(name, threshold_overrides=overrides)  # 不加载权重
        self.assertEqual((tool.config.ROAD_THRESHOLD, tool.config.ITSC_THRESHOLD, tool.config.TOPO_THRESHOLD), (0.1, 0.05, 0.455))
        self.assertEqual(tool.official_thresholds['ROAD_THRESHOLD'], 0.404)
        self.assertEqual(load_config(CHECKPOINTS['globalscale']).ROAD_THRESHOLD, 0.404)
        for bad in ({'ROAD': 1.5}, {'NMS': 0.2}):
            with self.assertRaises(ValueError):
                MaGRoadExtractor('globalscale', threshold_overrides=bad)

    def test_unknown_config_key_reads_false(self):
        config = load_config(CHECKPOINTS['wildroad'])
        self.assertFalse(config.SOME_MISSING_KEY)
        self.assertEqual(config.PATCH_SIZE, 1024)


class SlidingWindowTest(unittest.TestCase):
    def test_upstream_strides(self):
        expected = {'wildroad': 204.8, 'globalscale': 102.4, 'spacenet': 9.6}
        for name, stride in expected.items():
            spec = CHECKPOINTS[name]
            self.assertAlmostEqual(upstream_stride(spec, load_config(spec)), stride)

    def test_patch_starts_cover_and_respect_stride(self):
        for length, size, stride in ((726, 512, 102.4), (2420, 1024, 204.8), (256, 256, 9.6), (735, 256, 9.6)):
            starts = patch_starts(length, size, stride)
            self.assertEqual(starts[0], 0)
            self.assertEqual(starts[-1], length-size)
            self.assertLessEqual(max(np.diff(starts), default=0), stride+1)
        with self.assertRaises(ValueError):
            patch_starts(100, 256, 9.6)


class PointExtractionTest(unittest.TestCase):
    def test_nms_keeps_forced_points(self):
        points = np.array([[0, 0], [3, 0], [20, 0]])
        kept = nms_points(points, np.array([0.5, 2.0, 0.4]), 8)
        self.assertEqual({tuple(p) for p in kept}, {(3, 0), (20, 0)})
        kept = nms_points(points, np.array([0.9, 0.5, 0.4]), 8)
        self.assertEqual({tuple(p) for p in kept}, {(0, 0), (20, 0)})

    def test_extract_points_prefers_intersections(self):
        config = load_config(CHECKPOINTS['globalscale'])
        keypoint = np.zeros((64, 64), np.uint8)
        road = np.zeros((64, 64), np.uint8)
        road[30, 5:60] = 200
        keypoint[30, 32] = 250
        points = extract_points(keypoint, road, config)
        self.assertIn((32, 30), {tuple(p) for p in points})
        self.assertTrue(all(abs(p[0]-32) > config.ROAD_NMS_RADIUS or tuple(p) == (32, 30) for p in points))

    def test_neighbor_queries_exclude_self_and_radius(self):
        points = np.array([[0, 0], [10, 0], [100, 0]], dtype=np.float32)
        pairs, valid = neighbor_queries(points, 64, 4)
        self.assertEqual(pairs.shape, (3, 4, 2))
        self.assertEqual(valid.sum(axis=1).tolist(), [1, 1, 0])
        self.assertEqual(pairs[0, 0].tolist(), [0, 1])


if __name__ == '__main__':
    unittest.main()
