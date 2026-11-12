"""Node 1b（VLM 整图场景判断 → MaGRoad 权重选择）的离线回归：假 VLM、假道路提取器，不加载任何模型。"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from PIL import Image

from agent.map_preprocess import GlobalPreprocessConfig, run_pipeline
from agent.map_preprocess.io import read_json, resolve_path
from agent.vlm.scene_type import decide_scene, load_scene_system
from percection_tool.dummy import DummyRoadGraphExtractor


def answer(scene='结构化', confidence='高'):
    return {'schema_version': 2,
            'observations': {'road_surface': '铺装为主', 'road_layout': '规整连通路网',
                             'building_density': '成片密集', 'land_cover': '建成区为主'},
            'scene_type': scene, 'scene_type_confidence': confidence, 'evidence': '成片住宅与规整铺装路网。'}


class FakeResult:
    def __init__(self, text):
        self.text, self.usage, self.latency_seconds, self.reasoning = text, {}, 0.1, ''


class FakeVLM:
    def __init__(self, reply):
        self.reply, self.calls, self.closed = reply, [], 0

    def infer(self, prompt, images=(), **kwargs):
        self.calls.append({'prompt': prompt, 'images': list(images), **kwargs})
        if isinstance(self.reply, Exception):
            raise self.reply
        return FakeResult(json.dumps(self.reply, ensure_ascii=False))

    def close(self):
        self.closed += 1


class SceneDecisionTest(unittest.TestCase):
    def test_prompt_system_loads(self):
        system = load_scene_system()
        self.assertEqual(system['schema_version'], 2)
        self.assertIn('非结构化', system['prompt'])

    def test_decide_scene(self):
        self.assertEqual(decide_scene(answer('结构化'), 'wild')['scene'], 'urban')
        self.assertEqual(decide_scene(answer('非结构化', '中'), 'urban'), {'scene': 'wild', 'source': 'vlm', 'reason': 'vlm'})
        self.assertEqual(decide_scene(answer('结构化', '低'), 'wild')['reason'], 'low_confidence')
        self.assertEqual(decide_scene(answer('不确定'), 'wild')['scene'], 'wild')
        self.assertEqual(decide_scene(None, 'urban'), {'scene': 'urban', 'source': 'fallback', 'reason': 'vlm_error'})

    def test_config_validation(self):
        config = GlobalPreprocessConfig(segmenter='magroad_scene')
        self.assertEqual(config.magroad_urban_checkpoint, 'globalscale:ITSC=0.18,ROAD=0.18,TOPO=0.455')
        for bad in ({'magroad_wild_checkpoint': 'nope'}, {'magroad_urban_checkpoint': 'globalscale:ROAD=2'},
                    {'magroad_urban_checkpoint': 'globalscale:NMS=0.2'}, {'scene_type_override': 'forest'},
                    {'scene_type_fallback': 'x'}, {'scene_vlm_thinking_budget': 4000}):
            with self.assertRaises(ValueError, msg=bad):
                GlobalPreprocessConfig(segmenter='magroad_scene', **bad)


class ScenePipelineTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = self.root/'map.png'
        yy, xx = np.indices((700, 900))
        rgb = np.stack((xx % 256, yy % 256, (xx+yy) % 256), axis=-1).astype(np.uint8)
        rgb[:, :120] = 0                 # 纯黑外缘
        Image.fromarray(rgb).save(self.image)
        self.config = GlobalPreprocessConfig(segmenter='magroad_scene',
                                             dedup_enabled=False, output_dir=str(self.root/'out'),
                                             cache_dir=str(self.root/'cache'), scene_vlm_image_px=256)

    def run_scene(self, config, client):
        chosen = []

        def fake_factory(cfg, resolution, output_dir, *, magroad_checkpoint=None):
            chosen.append(magroad_checkpoint)
            return DummyRoadGraphExtractor()
        with patch('agent.map_preprocess.nodes.global_pipeline.create_graph_extractor', fake_factory), \
                warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            state = run_pipeline(str(self.image), config, vlm_client=client)
        return state, chosen

    def test_urban_selects_globalscale_thresholds_and_closes_vlm(self):
        client = FakeVLM(answer('结构化'))
        state, chosen = self.run_scene(self.config, client)
        self.assertEqual(chosen, ['globalscale:ITSC=0.18,ROAD=0.18,TOPO=0.455'])
        self.assertEqual(state['scene_type']['scene'], 'urban')
        self.assertEqual(state['scene_type']['source'], 'vlm')
        self.assertEqual(client.closed, 1)
        call = client.calls[0]
        self.assertEqual(len(call['images']), 1)
        self.assertEqual(call['image_labels'], ['整图缩略图'])
        self.assertEqual(call['temperature'], 0.0)
        self.assertTrue(all(Path(p).is_file() for p in call['images']))
        record = read_json(state['scene_type']['record_path'])
        self.assertEqual(record['output']['scene_type'], '结构化')
        self.assertLess(record['inputs']['valid_fraction_overview'], 0.9)   # 黑边被识别为无效
        snapshots = sorted(p.name for p in (resolve_path(state['output_dir'])/'state/node_states').iterdir())
        self.assertTrue(snapshots[1].endswith('classify_scene_type.json'), snapshots)

    def test_wild_and_fallbacks(self):
        _, chosen = self.run_scene(self.config, FakeVLM(answer('非结构化')))
        self.assertEqual(chosen, ['wildroad'])
        state, chosen = self.run_scene(replace(self.config, scene_type_fallback='urban'), FakeVLM(RuntimeError('boom')))
        self.assertEqual((state['scene_type']['source'], state['scene_type']['reason']), ('fallback', 'vlm_error'))
        self.assertEqual(chosen, ['globalscale:ITSC=0.18,ROAD=0.18,TOPO=0.455'])

    def test_override_skips_vlm(self):
        client = FakeVLM(answer('非结构化'))
        state, chosen = self.run_scene(replace(self.config, scene_type_override='urban'), client)
        self.assertEqual(client.calls, [])
        self.assertEqual(state['scene_type']['source'], 'override')
        self.assertEqual(chosen, ['globalscale:ITSC=0.18,ROAD=0.18,TOPO=0.455'])

    def test_known_image_skips_vlm_and_new_image_uses_it(self):
        client = FakeVLM(answer('非结构化'))
        state, chosen = self.run_scene(replace(self.config, scene_type_known={'map': 'urban'}), client)
        self.assertEqual(client.calls, [])
        self.assertEqual(state['scene_type']['source'], 'known_image')
        self.assertEqual(chosen, ['globalscale:ITSC=0.18,ROAD=0.18,TOPO=0.455'])
        state, chosen = self.run_scene(replace(self.config, scene_type_known={'other': 'urban'}), client)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual((state['scene_type']['source'], chosen), ('vlm', ['wildroad']))
        with self.assertRaises(ValueError):
            GlobalPreprocessConfig(segmenter='magroad_scene', scene_type_known={'map': 'forest'})

    def test_dom_suffix_matches_known_scene_and_exact_name_takes_priority(self):
        named = self.root/'map_dom.tif'
        self.image.rename(named)
        self.image = named
        client = FakeVLM(answer('非结构化'))
        for known, expected in (({'map': 'urban'}, 'urban'), ({'map': 'urban', 'map_dom': 'wild'}, 'wild')):
            state, _ = self.run_scene(replace(self.config, scene_type_known=known), client)
            self.assertEqual((state['scene_type']['source'], state['scene_type']['scene']), ('known_image', expected))
        self.assertEqual(client.calls, [])

    def test_cache_key_depends_on_selected_checkpoint(self):
        first, _ = self.run_scene(self.config, FakeVLM(answer('结构化')))
        again, _ = self.run_scene(self.config, FakeVLM(answer('结构化')))
        other, _ = self.run_scene(self.config, FakeVLM(answer('非结构化')))
        self.assertTrue(again['road_graph_cache']['hit'])
        self.assertEqual(first['road_graph_cache']['key'], again['road_graph_cache']['key'])
        self.assertNotEqual(first['road_graph_cache']['key'], other['road_graph_cache']['key'])


if __name__ == '__main__':
    unittest.main()
