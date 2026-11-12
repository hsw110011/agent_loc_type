"""SegEarth-OV3 适配器离线回归：不加载 SAM 3，用假分数核对滑窗融合、类别映射和配置分发。"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from agent.map_preprocess.nodes.block_features import _model_fingerprint
from agent.map_preprocess.utils.block_features import semantic_labels
from agent.perception_config import SegEarthOV3Config, load_perception_config
from percection_tool.segearth_ov3.import_model import sha256
from percection_tool.segearth_ov3.inference import (BUILDING, PLANT, PROMPT_LABELS, PROMPTS, ROAD, SegEarthOV3Segmenter, blend_weights,
                                                    expand_prompts, score_to_confidence, window_starts)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERIES, ROWS = expand_prompts()
ROAD_QUERY, GRASS_QUERY = QUERIES.index('road'), QUERIES.index('grass')


def tool_with_scores(function, size=64, stride=48, threshold=.1):
    tool = SegEarthOV3Segmenter(SegEarthOV3Config(tile_size=size, stride=stride, score_threshold=threshold))
    tool.processor, tool.device = object(), torch.device('cpu')
    tool._view_scores = function
    return tool


def run_cpu(tool, image, valid=None):
    with mock.patch('torch.cuda.reset_peak_memory_stats'), mock.patch('torch.cuda.max_memory_allocated', return_value=0):
        return tool.predict(image, valid_mask=valid)


def constant(query, value=.9):
    def scores(tile):
        out = np.zeros((len(QUERIES), *tile.shape[:2]), np.float32)
        out[query] = value
        return out
    return scores


class SegEarthOV3PredictionTest(unittest.TestCase):
    def test_every_prompt_maps_to_one_of_three_classes(self):
        self.assertEqual(len(PROMPTS), len(PROMPT_LABELS))
        self.assertEqual(set(PROMPT_LABELS), {ROAD, BUILDING, PLANT})
        self.assertNotIn('background', ' '.join(PROMPTS))
        self.assertNotIn('water', ' '.join(PROMPTS))
        image = np.full((20, 20, 3), 100, np.uint8)
        for query, row in enumerate(ROWS):
            mask = run_cpu(tool_with_scores(constant(query)), image)['mask']
            self.assertTrue((mask == PROMPT_LABELS[row]).all(), QUERIES[query])

    def test_synonyms_take_maximum_before_comparing_rows(self):
        # 单词 tree 和 forest 各 0.4，取 max 后仍是 0.4；road 0.5 应获胜（求和时 plant 会误胜）。
        def scores(tile):
            out = np.zeros((len(QUERIES), *tile.shape[:2]), np.float32)
            out[QUERIES.index('tree')] = out[QUERIES.index('forest')] = .4
            out[ROAD_QUERY] = .5
            return out
        mask = run_cpu(tool_with_scores(scores), np.full((16, 16, 3), 9, np.uint8))['mask']
        self.assertTrue((mask == 0).all())

    def test_low_score_still_takes_best_of_three_classes(self):
        image = np.full((16, 16, 3), 9, np.uint8)
        low = run_cpu(tool_with_scores(constant(ROAD_QUERY, .03)), image)
        self.assertTrue((low['mask'] == ROAD).all())
        self.assertEqual(low['metadata']['classes'], ['road', 'building', 'plant'])

    def test_confidence_range_never_triggers_pipeline_other(self):
        self.assertAlmostEqual(float(score_to_confidence(np.array(.03), .1)), .5)
        self.assertAlmostEqual(float(score_to_confidence(np.array(.1), .1)), .5)
        self.assertAlmostEqual(float(score_to_confidence(np.array(1.), .1)), 1.)
        output = run_cpu(tool_with_scores(constant(ROAD_QUERY, .01)), np.full((16, 16, 3), 9, np.uint8))
        self.assertTrue((output['confidence'] >= .5).all())
        labels = semantic_labels(output['mask'], output['confidence'], .5)
        self.assertEqual(set(np.unique(labels).tolist()), {ROAD})

    def test_invalid_and_pure_black_pixels_are_ignored(self):
        image = np.full((16, 16, 3), 9, np.uint8)
        image[0, 0] = 0
        valid = np.ones((16, 16), bool)
        valid[5, 5] = False
        output = run_cpu(tool_with_scores(constant(GRASS_QUERY)), image, valid)
        self.assertEqual(int(output['mask'][0, 0]), 255)
        self.assertEqual(int(output['mask'][5, 5]), 255)
        self.assertEqual(float(output['confidence'][5, 5]), 0.)
        self.assertEqual(output['metadata']['valid_pixels'], 16*16-2)

    def test_small_input_is_padded_with_zeros_to_window_size(self):
        seen = []
        def scores(tile):
            seen.append(tile.copy())
            return constant(ROAD_QUERY)(tile)
        run_cpu(tool_with_scores(scores), np.full((20, 30, 3), 7, np.uint8))
        self.assertEqual(seen[0].shape, (64, 64, 3))
        self.assertEqual(int(seen[0][19, 29, 0]), 7)
        self.assertEqual(int(seen[0][20:].max()), 0)

    def test_windows_without_valid_pixels_are_skipped(self):
        calls = []
        def scores(tile):
            calls.append(1)
            return constant(ROAD_QUERY)(tile)
        image = np.full((64, 160, 3), 9, np.uint8)
        valid = np.zeros((64, 160), bool)
        valid[:, :10] = True
        output = run_cpu(tool_with_scores(scores), image, valid)
        self.assertEqual(len(calls), output['metadata']['tile_count'])
        self.assertLess(len(calls), len(window_starts(160, 64, 48)))

    def test_matches_segment_geotiff_blending(self):
        """按脚本的滑窗融合单独计算一遍，核对重叠区融合后的类别（三类、无背景阈值）。"""
        rng = np.random.default_rng(0)
        height, width, size, stride = 150, 170, 64, 48
        image = rng.integers(1, 255, (height, width, 3), dtype=np.uint8)
        fields = rng.random((len(QUERIES), 5, 5), dtype=np.float32)*.6

        def scores(tile):
            digest = int(tile[:, :, 0].sum()) % 1000
            ramp = np.linspace(.2, 1., tile.shape[0], dtype=np.float32)[None, :, None]
            out = np.stack([np.kron(fields[q], np.ones((13, 13), np.float32))[:tile.shape[0], :tile.shape[1]]
                            for q in range(len(QUERIES))])*ramp
            return out*(.5+digest/2000)

        output = run_cpu(tool_with_scores(scores, size, stride), image)
        weights = blend_weights(size, stride)
        total, count = np.zeros((len(PROMPTS), height, width), np.float32), np.zeros((height, width), np.float32)
        last_x, last_y = max(0, width-size), max(0, height-size)
        for y in [*range(0, last_y, stride), last_y]:
            for x in [*range(0, last_x, stride), last_x]:
                rows, columns = min(size, height-y), min(size, width-x)
                tile = np.pad(image[y:y+rows, x:x+columns], ((0, size-rows), (0, size-columns), (0, 0)))
                view = scores(tile)
                merged = np.zeros((len(PROMPTS), rows, columns), np.float32)
                for query, row in enumerate(ROWS):
                    np.maximum(merged[row], view[query, :rows, :columns], out=merged[row])
                total[:, y:y+rows, x:x+columns] += merged*weights[:rows, :columns]
                count[y:y+rows, x:x+columns] += weights[:rows, :columns]
        np.testing.assert_array_equal(output['mask'], np.asarray(PROMPT_LABELS, np.uint8)[total.argmax(axis=0)])

    def test_rejects_bad_input_and_nonfinite_scores_and_restores_threads(self):
        tool = tool_with_scores(constant(ROAD_QUERY))
        with self.assertRaises(ValueError):
            tool.predict(np.zeros((4, 4, 3), np.float32))
        def bad(tile):
            out = constant(ROAD_QUERY)(tile)
            out[0, 0, 0] = np.nan
            return out
        before = torch.get_num_threads()
        with self.assertRaisesRegex(RuntimeError, '非有限'):
            run_cpu(tool_with_scores(bad), np.full((16, 16, 3), 9, np.uint8))
        self.assertEqual(torch.get_num_threads(), before)

    def test_sha_mismatch_stops_before_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'sam3.pt').write_bytes(b'changed')
            (root/'SOURCE.json').write_text(json.dumps({'files': {'sam3.pt': {'sha256': '0'*64}}}), encoding='utf-8')
            tool = SegEarthOV3Segmenter(SegEarthOV3Config(weights_dir=directory))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                tool._load()
            self.assertIsNone(tool.processor)


class SegEarthOV3ConfigTest(unittest.TestCase):
    def test_pipeline_uses_segearth(self):
        config = load_perception_config('agent/perception_tools.yaml')
        self.assertIsInstance(config.pipeline_semantic_config, SegEarthOV3Config)
        self.assertEqual((config.segearth_ov3.tile_size, config.segearth_ov3.stride), (1600, 1200))

    def test_invalid_settings_are_rejected(self):
        for kwargs in ({'device': 'cpu'}, {'stride': 1600}, {'score_threshold': 1.}, {'tile_size': 0}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                SegEarthOV3Config(**kwargs)

    def test_cache_identity_changes_with_threshold_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'SOURCE.json'
            path.write_text(json.dumps({'files': {'sam3.pt': {'sha256': 'a'*64}}}), encoding='utf-8')
            first = _model_fingerprint(None, SegEarthOV3Config(weights_dir=directory))
            self.assertEqual(first['model'], 'segearth_ov3')
            self.assertEqual(first['prompts'], list(PROMPTS))
            self.assertNotEqual(first, _model_fingerprint(None, SegEarthOV3Config(weights_dir=directory, score_threshold=.2)))
            path.write_text(json.dumps({'files': {'sam3.pt': {'sha256': 'b'*64}}}), encoding='utf-8')
            self.assertNotEqual(first, _model_fingerprint(None, SegEarthOV3Config(weights_dir=directory)))

    def test_factory_dispatch_is_lazy(self):
        code = """
import sys
from agent.perception_config import SegEarthOV3Config
from percection_tool.factory import create_semantic_segmenter
assert 'percection_tool.segearth_ov3.inference' not in sys.modules
tool = create_semantic_segmenter(SegEarthOV3Config())
assert type(tool).__name__ == 'SegEarthOV3Segmenter'
assert type(create_semantic_segmenter()).__name__ == 'SegEarthOV3Segmenter'  # 不传配置时的默认工具
assert 'sam3' not in sys.modules and 'torch' not in sys.modules
"""
        result = subprocess.run([sys.executable, '-c', code], cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_import_model_refuses_to_overwrite_different_weights(self):
        from percection_tool.segearth_ov3.import_model import TRACKED, import_model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root/'source', root/'target'
            (source/'weights/sam3').mkdir(parents=True)
            (source/'weights/sam3/sam3.pt').write_bytes(b'weights')
            for name in TRACKED:
                (source/name).parent.mkdir(parents=True, exist_ok=True)
                (source/name).write_bytes(name.encode())
            record = json.loads(import_model(source, target).read_text(encoding='utf-8'))
            self.assertEqual(record['files']['sam3.pt']['sha256'], sha256(source/'weights/sam3/sam3.pt'))
            (target/'sam3.pt').write_bytes(b'other')
            with self.assertRaisesRegex(ValueError, '拒绝覆盖'):
                import_model(source, target)


if __name__ == '__main__':
    unittest.main()
