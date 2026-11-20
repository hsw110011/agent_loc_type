"""Node 9 辅助定位方式推荐离线回归：假 VLM + 合成图，不加载 Qwen、不需要 GPU。

这些用例验证规则与接口，假 VLM 的答案是预设的，不能证明真实模型的推荐质量。
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from jsonschema import ValidationError
from PIL import Image

from agent.map_preprocess.global_config import GlobalPreprocessConfig, NODE_ORDER
from agent.map_preprocess.graph import _run_node, build_graph
from agent.map_preprocess.io import read_json, write_json
from agent.map_preprocess.nodes import degradation_fusion as df
from agent.map_preprocess.nodes import localization_recommendation as lr
from agent.map_preprocess.nodes.global_pipeline import refresh_global_manifest
from agent.map_preprocess.run_logging import NodeProgress
from agent.map_preprocess.utils.patch_fusion import fuse_patches
from agent.paths import resolve_path
from agent.vlm.common import VLMError
from agent.vlm.localization_recommendation import (InvalidRecommendation, METHODS, PROMPT_NAME, ROOT, STATUSES, apply_program_rules, bind_region,
                                                   check_recommendation, load_recommendation_system, overclaims)

BACKGROUND = (90, 120, 150)
SYSTEM = load_recommendation_system()


def answer(region_id: str, method: str = 'gps', status: str = 'provisional', **overrides) -> dict:
    assessment = {key: 'suitable' if method == key else 'unsuitable' for key in ('lidar_inertial_slam', 'wheel_imu_odometry')}
    assessment.update(structure_density='dense' if method == 'lidar_inertial_slam' else 'sparse_or_repetitive',
                      sky_view='open', obstruction_or_reflection='absent')
    return {'region_id': region_id, 'visual_evidence': [f'{region_id} 道路两侧可见农田'], 'recommended_method': method,
            'assessment': assessment,
            'reason': f'{region_id} 周边开阔', 'limitations': [f'{region_id} 局部可能有遮挡'], 'recommendation_status': status, **overrides}


class FakeVLM:
    """按 region_id 返回预设输出，一个 region_id 可给多次回答（用于重试）。"""

    def __init__(self, answers: dict):
        self.answers, self.calls, self.lock = {k: list(v) if isinstance(v, list) else [v] for k, v in answers.items()}, [], threading.Lock()

    def infer(self, prompt, images=(), *, system, max_tokens=None, temperature=0., json_output=False, output_schema=None,
              image_labels=None, enable_thinking=False, thinking_token_budget=None):
        region_id = prompt.split('请为区域 ')[1].split(' ')[0]
        with self.lock:
            self.calls.append({'region_id': region_id, 'prompt': prompt, 'system': system, 'images': [str(p) for p in images], 'labels': image_labels,
                               'schema': output_schema, 'thinking': (enable_thinking, thinking_token_budget), 'max_tokens': max_tokens})
            queue = self.answers[region_id]
            item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return SimpleNamespace(text=text, usage={'prompt_tokens': 10, 'completion_tokens': 5}, latency_seconds=.1, reasoning='')


class SystemTests(unittest.TestCase):
    def test_prompt_system_files_and_division(self):
        self.assertTrue((ROOT/PROMPT_NAME).is_file())
        for heading in ('Role', 'Objective', 'Input', 'Assessment Criteria', 'Decision Procedure', 'Constraints', 'Uncertainty Handling', 'Output Requirements'):
            self.assertIn(heading, SYSTEM['prompt'])
        self.assertNotIn('隧道', SYSTEM['prompt'])  # 适用条件只在 Rule 中定义
        self.assertIn('隧道', SYSTEM['rule'])

    def test_localization_priority_and_gps_last_resort(self):
        selection = SYSTEM['rules']['selection']
        self.assertEqual({k: v['priority'] for k, v in SYSTEM['rules']['methods'].items()},
                         {'lidar_inertial_slam': 'P0', 'wheel_imu_odometry': 'P1', 'gps': 'P2'})
        self.assertEqual(selection['last_resort']['method'], 'gps')
        self.assertIn('持续开阔', selection['last_resort']['when'])

    def test_vlm_is_not_told_degradation_or_asked_for_numbers(self):
        for text in (SYSTEM['prompt'], SYSTEM['skill'], SYSTEM['rule']):
            for word in ('退化类别', '纵向退化', 'degradation_type', '掩膜', '语义'):
                self.assertNotIn(word, text)
        for field in SYSTEM['schema']['properties'].values():
            self.assertNotIn(field.get('type'), ('number', 'integer'))

    def test_program_section_is_not_sent_to_model(self):
        self.assertNotIn('overclaim_phrases', SYSTEM['rule'])
        self.assertNotIn('mandatory_limitations', SYSTEM['rule'])

    def test_unified_enums(self):
        fields = SYSTEM['schema']['properties']
        self.assertEqual(tuple(fields['recommended_method']['enum']), METHODS)
        self.assertEqual(set(METHODS), {'lidar_inertial_slam', 'wheel_imu_odometry', 'gps'})
        self.assertEqual(tuple(fields['recommendation_status']['enum']), STATUSES)

    def test_model_cannot_return_geometry_or_degradation(self):
        fields = SYSTEM['schema']['properties']
        self.assertFalse(SYSTEM['schema']['additionalProperties'])
        for name in ('bbox', 'degradation_type', 'union_polygon'):
            self.assertNotIn(name, fields)

    def test_evidence_is_generated_before_method(self):
        order = list(SYSTEM['schema']['properties'])
        self.assertLess(order.index('visual_evidence'), order.index('recommended_method'))

    def test_loader_rejects_inconsistent_enums(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            for path in ROOT.iterdir():
                if path.is_file() and path.suffix != '.py':
                    (base/path.name).write_text(path.read_text(encoding='utf-8'), encoding='utf-8')
            schema = json.loads((base/'schema.json').read_text(encoding='utf-8'))
            schema['properties']['recommended_method']['enum'] = ['gps', 'vio']
            (base/'schema.json').write_text(json.dumps(schema), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '枚举'):
                load_recommendation_system(base)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.schema = bind_region(SYSTEM['schema'], 'Merged_1')

    def test_valid_output_passes(self):
        check_recommendation(answer('Merged_1'), self.schema, SYSTEM['rules'])

    def test_gps_requires_both_odometry_methods_unsuitable_and_open_sky_without_obstruction(self):
        for key, value in (('lidar_inertial_slam', 'suitable'), ('wheel_imu_odometry', 'uncertain'),
                           ('sky_view', 'obstructed'), ('obstruction_or_reflection', 'present')):
            output = answer('Merged_1')
            output['assessment'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(InvalidRecommendation, 'GPS 要求'):
                check_recommendation(output, self.schema, SYSTEM['rules'])

    def test_lio_precedes_suitable_wheel_even_with_gps_obstruction(self):
        output = answer('Merged_1', 'wheel_imu_odometry')
        output['assessment'].update(lidar_inertial_slam='suitable', sky_view='obstructed',
                                    obstruction_or_reflection='present')
        with self.assertRaisesRegex(InvalidRecommendation, 'lidar_inertial_slam'):
            check_recommendation(output, self.schema, SYSTEM['rules'])
        output['recommended_method'] = 'lidar_inertial_slam'
        check_recommendation(output, self.schema, SYSTEM['rules'])

    def test_dense_structure_requires_lidar_inertial_slam(self):
        output = answer('Merged_1', 'wheel_imu_odometry')
        output['assessment'].update(structure_density='dense', lidar_inertial_slam='uncertain', wheel_imu_odometry='suitable')
        with self.assertRaisesRegex(InvalidRecommendation, 'structure_density=dense'):
            check_recommendation(output, self.schema, SYSTEM['rules'])
        output['assessment']['lidar_inertial_slam'] = 'unsuitable'
        output['recommended_method'] = 'lidar_inertial_slam'
        with self.assertRaisesRegex(InvalidRecommendation, 'unsuitable'):
            check_recommendation(output, self.schema, SYSTEM['rules'])
        output['assessment']['lidar_inertial_slam'] = 'suitable'
        check_recommendation(output, self.schema, SYSTEM['rules'])

    def test_uncertain_odometry_requires_validation(self):
        output = answer('Merged_1', 'wheel_imu_odometry')
        output['assessment']['wheel_imu_odometry'] = 'uncertain'
        with self.assertRaisesRegex(InvalidRecommendation, '暂定'):
            check_recommendation(output, self.schema, SYSTEM['rules'])
        output.update(recommendation_status='requires_validation')
        check_recommendation(output, self.schema, SYSTEM['rules'])

    def test_illegal_enum_and_wrong_id_and_extra_field(self):
        for bad in (answer('Merged_1', method='vio'), answer('Merged_2'), answer('Merged_1', status='confirmed'), answer('Merged_1', bbox=[0, 0, 1, 1])):
            with self.assertRaises(ValidationError):
                check_recommendation(bad, self.schema, SYSTEM['rules'])

    def test_blank_text_is_rejected(self):
        with self.assertRaises(InvalidRecommendation):
            check_recommendation(answer('Merged_1', reason='   '), self.schema, SYSTEM['rules'])

    def test_overclaim_is_rejected_but_negation_is_allowed(self):
        self.assertEqual(overclaims(answer('M', reason='开阔，可以保证定位成功'), SYSTEM['rules']), ['保证定位'])
        self.assertEqual(overclaims(answer('M', reason='开阔，但不能保证定位成功'), SYSTEM['rules']), [])
        self.assertEqual(overclaims(answer('M', reason='道路平直，轮速计不会漂移'), SYSTEM['rules']), ['不会漂移'])
        with self.assertRaises(InvalidRecommendation):
            check_recommendation(answer('Merged_1', reason='GPS 绝对可靠'), self.schema, SYSTEM['rules'])

    def test_reliability_numbers_are_rejected(self):
        for text in ('GPS 适用性 95%', 'LiDAR 成功率约 85', '置信度为 0.9', '有百分之九十把握', '遮挡约占 30％'):
            with self.assertRaisesRegex(InvalidRecommendation, '定性', msg=text):
                check_recommendation(answer('Merged_1', reason=text), self.schema, SYSTEM['rules'])
        with self.assertRaisesRegex(InvalidRecommendation, '定性'):
            check_recommendation(answer('Merged_1', limitations=['成功率 80%']), self.schema, SYSTEM['rules'])
        check_recommendation(answer('Merged_1', reason='道路两侧有 2 排建筑，可能提供几何约束'), self.schema, SYSTEM['rules'])

    def test_unsupported_signal_claim_is_rejected(self):
        self.assertEqual(overclaims(answer('M', reason='GPS 信号可穿透树冠'), SYSTEM['rules']), ['穿透树冠'])

    def test_program_rules_only_tighten(self):
        wheel = apply_program_rules(answer('M', 'wheel_imu_odometry', 'provisional'), SYSTEM['rules'])
        self.assertEqual(wheel['recommendation_status'], 'requires_validation')
        self.assertTrue(any('累积' in item for item in wheel['limitations']))
        self.assertTrue(any('全局位置校正' in item for item in wheel['limitations']))
        self.assertEqual(apply_program_rules(answer('M', 'gps', 'requires_validation'), SYSTEM['rules'])['recommendation_status'], 'requires_validation')
        self.assertEqual(apply_program_rules(answer('M', 'gps', 'provisional'), SYSTEM['rules'])['recommendation_status'], 'provisional')

    def test_model_limitations_are_kept_first(self):
        limits = apply_program_rules(answer('M', 'gps'), SYSTEM['rules'])['limitations']
        self.assertEqual(limits[0], 'M 局部可能有遮挡')
        self.assertEqual(len(limits), 2)


class GeometryTests(unittest.TestCase):
    def test_crop_box_clips_and_rejects_outside(self):
        self.assertEqual(lr.crop_box([-5, 10, 50, 900], 100, 100), [0, 10, 50, 100])
        self.assertEqual(lr.crop_box([10.2, 10.7, 20.1, 30.5], 100, 100), [10, 10, 21, 31])
        self.assertIsNone(lr.crop_box([200, 200, 300, 300], 100, 100))

    def test_image_size_keeps_aspect_and_never_upscales(self):
        self.assertEqual(lr.image_size([0, 0, 4000, 2000], 1024), (1024, 512))
        self.assertEqual(lr.image_size([0, 0, 100, 50], 1024), (100, 50))

    def test_image_rings_map_global_to_crop(self):
        region = {'union_polygon': {'type': 'Polygon', 'coordinates': [[[100, 200], [300, 200], [300, 400], [100, 400], [100, 200]]]}}
        self.assertEqual(lr.image_rings(region, [100, 200, 300, 400], (100, 100))[0][0][2], (100., 100.))


class NodeTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        records = self.root/'task'/'state'
        records.mkdir(parents=True)
        self.manifest_path = records/'manifest.json'
        self.regions_path = records/'degradation_regions.json'
        image = Image.new('RGB', (1000, 800), BACKGROUND)
        image.save(self.root/'image.png')
        self.manifest = {'format_version': 2, 'workflow': 'global_blocks', 'task_id': 't', 'input_sha256': 'x',
                         'image': {'path': str(self.root/'image.png'), 'width': 1000, 'height': 800}, 'resolution_m_per_px': .5, 'resolution_source': 'geotiff',
                         'tiles': [], 'nodes': [], 'edges': [], 'blocks': [], 'groups': [], 'extensions': {},
                         'config': {'pillow_max_image_pixels': 10**9}, 'files': {'degradation_regions': str(self.regions_path)}}
        self.config = GlobalPreprocessConfig(segmenter='dummy', block_size_m=64, block_features_enabled=True, vlm_review_enabled=True, vlm_expansion_enabled=True,
                                             degradation_fusion_enabled=True, vlm_localization_enabled=True)

    def write_regions(self, patches: list[dict]) -> list[dict]:
        regions = fuse_patches(patches)['regions']
        write_json(self.manifest_path, self.manifest)
        write_json(self.regions_path, {'regions': regions})
        return regions

    def two_regions(self) -> list[dict]:
        return self.write_regions([{'patch_id': 'A', 'bbox': [100, 100, 200, 200], 'degradation_type': 'degradation'},
                                   {'patch_id': 'B', 'bbox': [150, 150, 250, 250], 'degradation_type': 'degradation'},
                                   {'patch_id': 'C', 'bbox': [600, 400, 700, 500], 'degradation_type': 'longitudinal_degradation'}])

    def run_node(self, client, **kwargs) -> dict:
        return lr.run_localization_recommendation(self.manifest_path, kwargs.pop('config', self.config), client=client, **kwargs)

    def output(self) -> dict:
        return read_json(self.manifest_path.parent/'localization_recommendations.json')

    # 用例编号对应 Node 9 的测试要求；假 VLM 的回答是预设的，验证的是接口与规则
    def test_case1_gps_notes_realtime_validation(self):
        self.write_regions([{'patch_id': 'A', 'bbox': [100, 100, 200, 200], 'degradation_type': 'degradation'}])
        self.run_node(FakeVLM({'Merged_1': answer('Merged_1', 'gps')}))
        region = self.output()['regions'][0]
        self.assertEqual(region['recommended_method'], 'gps')
        self.assertTrue(any('实时信号质量' in item for item in region['limitations']))

    def test_case2_lidar_notes_online_validation(self):
        self.write_regions([{'patch_id': 'A', 'bbox': [100, 100, 200, 200], 'degradation_type': 'degradation'}])
        self.run_node(FakeVLM({'Merged_1': answer('Merged_1', 'lidar_inertial_slam', 'requires_validation')}))
        region = self.output()['regions'][0]
        self.assertEqual(region['recommended_method'], 'lidar_inertial_slam')
        self.assertTrue(any('车载 LiDAR' in item for item in region['limitations']))

    def test_case3_wheel_imu_states_drift_and_short_term(self):
        self.write_regions([{'patch_id': 'A', 'bbox': [100, 100, 200, 200], 'degradation_type': 'degradation'}])
        self.run_node(FakeVLM({'Merged_1': answer('Merged_1', 'wheel_imu_odometry', 'provisional')}))
        region = self.output()['regions'][0]
        self.assertEqual(region['recommendation_status'], 'requires_validation')
        text = ''.join(region['limitations'])
        self.assertIn('累积', text)
        self.assertIn('短时', text)

    def test_case4_mixed_environment_needs_validation(self):
        self.write_regions([{'patch_id': 'A', 'bbox': [100, 100, 200, 200], 'degradation_type': 'degradation'}])
        self.run_node(FakeVLM({'Merged_1': answer('Merged_1', 'lidar_inertial_slam', 'requires_validation',
                                                 reason='北半部开阔、南半部密集建筑，单一方案难以覆盖全区域')}))
        region = self.output()['regions'][0]
        self.assertNotIn('review_required', region)
        self.assertEqual(region['recommendation_status'], 'requires_validation')
        self.assertIn('混合', SYSTEM['prompt'])

    def test_case5_each_region_is_independent_and_matched(self):
        regions = self.two_regions()
        client = FakeVLM({'Merged_1': answer('Merged_1', 'lidar_inertial_slam'), 'Merged_2': answer('Merged_2', 'gps')})
        self.run_node(client)
        self.assertEqual(sorted(call['region_id'] for call in client.calls), ['Merged_1', 'Merged_2'])
        for call in client.calls:
            other = 'Merged_2' if call['region_id'] == 'Merged_1' else 'Merged_1'
            self.assertNotIn(other, call['prompt'])
        results = self.output()['regions']
        self.assertEqual([r['region_id'] for r in results], [r['merged_id'] for r in regions])
        self.assertEqual([r['recommended_method'] for r in results], ['lidar_inertial_slam', 'gps'])
        self.assertEqual([r['reason'] for r in results], ['Merged_1 周边开阔', 'Merged_2 周边开阔'])
        self.assertEqual([r['degradation_type'] for r in results], ['degradation', 'longitudinal_degradation'])

    def test_case6_bbox_is_identical_to_node8_and_node8_file_unchanged(self):
        regions = self.two_regions()
        before = self.regions_path.read_bytes()
        self.run_node(FakeVLM({'Merged_1': answer('Merged_1'), 'Merged_2': answer('Merged_2')}))
        results = self.output()['regions']
        self.assertEqual([r['bbox'] for r in results], [r['bbox'] for r in regions])
        self.assertEqual([r['union_polygon'] for r in results], [r['union_polygon'] for r in regions])
        self.assertEqual([r['source_patch_ids'] for r in results], [r['source_patch_ids'] for r in regions])
        self.assertEqual(self.regions_path.read_bytes(), before)

    def test_case7_empty_regions_make_no_vlm_call(self):
        self.write_regions([])
        client = FakeVLM({})
        summary = self.run_node(client)
        self.assertEqual(client.calls, [])
        self.assertEqual(self.output()['regions'], [])
        self.assertEqual((summary['status'], summary['region_count']), ('completed', 0))

    def test_case8_invalid_outputs_never_become_recommendations(self):
        for bad in ('{不是 JSON', json.dumps(answer('Merged_1', method='vio')), json.dumps(answer('Merged_2')), json.dumps(answer('Merged_1', reason='可以保证定位成功'))):
            self.write_regions([{'patch_id': 'A', 'bbox': [100, 100, 200, 200], 'degradation_type': 'degradation'}])
            summary = self.run_node(FakeVLM({'Merged_1': bad}), config=replace(self.config, vlm_localization_max_retries=0))
            document = self.output()
            self.assertEqual(document['regions'], [], bad)
            self.assertEqual([r['region_id'] for r in document['failed_regions']], ['Merged_1'])
            self.assertEqual((summary['status'], summary['failed_count']), ('partial', 1))

    def test_retry_feeds_back_and_recovers(self):
        self.write_regions([{'patch_id': 'A', 'bbox': [100, 100, 200, 200], 'degradation_type': 'degradation'}])
        client = FakeVLM({'Merged_1': [json.dumps(answer('Merged_1', reason='GPS 绝对可靠')), answer('Merged_1')]})
        summary = self.run_node(client)
        self.assertEqual(len(client.calls), 2)
        self.assertNotIn('上一次输出未通过校验', client.calls[0]['prompt'])
        self.assertIn('绝对可靠', client.calls[1]['prompt'])
        self.assertEqual((summary['status'], summary['recommended_count']), ('completed', 1))

    def test_retries_are_bounded(self):
        self.write_regions([{'patch_id': 'A', 'bbox': [100, 100, 200, 200], 'degradation_type': 'degradation'}])
        client = FakeVLM({'Merged_1': '{坏'})
        self.run_node(client, config=replace(self.config, vlm_localization_max_retries=2))
        self.assertEqual(len(client.calls), 3)
        record = read_json(self.manifest_path.parent/'localization_recommendation'/'Merged_1.json')
        self.assertEqual((record['status'], len(record['attempts'])), ('error', 3))

    def test_transient_service_error_is_retried(self):
        self.write_regions([{'patch_id': 'A', 'bbox': [100, 100, 200, 200], 'degradation_type': 'degradation'}])
        client = FakeVLM({'Merged_1': [VLMError('VLM 服务已退出或不可用'), answer('Merged_1')]})
        self.assertEqual(self.run_node(client)['recommended_count'], 1)
        self.assertNotIn('上一次输出未通过校验', client.calls[1]['prompt'])

    def test_unexpected_exception_is_recorded_without_retry(self):
        self.write_regions([{'patch_id': 'A', 'bbox': [100, 100, 200, 200], 'degradation_type': 'degradation'}])
        client = FakeVLM({'Merged_1': KeyError('bug')})
        self.assertEqual(self.run_node(client)['failed_count'], 1)
        self.assertEqual(len(client.calls), 1)

    def test_one_failure_does_not_affect_other_regions(self):
        self.two_regions()
        with self.assertLogs('map_preprocess.localization_recommendation', level='WARNING'):
            summary = self.run_node(FakeVLM({'Merged_1': '{坏', 'Merged_2': answer('Merged_2')}))
        document = self.output()
        self.assertEqual([r['region_id'] for r in document['regions']], ['Merged_2'])
        failed = document['failed_regions'][0]
        self.assertEqual(failed['region_id'], 'Merged_1')
        self.assertNotIn('recommended_method', failed)
        self.assertEqual(failed['bbox'], [100, 100, 250, 250])
        self.assertEqual((summary['status'], summary['failed_region_ids']), ('partial', ['Merged_1']))

    def test_every_node8_region_appears_exactly_once(self):
        regions = self.two_regions()
        self.run_node(FakeVLM({'Merged_1': '{坏', 'Merged_2': answer('Merged_2')}))
        document = self.output()
        self.assertEqual(sorted(r['region_id'] for r in document['regions']+document['failed_regions']), sorted(r['merged_id'] for r in regions))

    def test_region_outside_image_fails_without_vlm(self):
        self.write_regions([{'patch_id': 'A', 'bbox': [2000, 2000, 2100, 2100], 'degradation_type': 'degradation'}])
        client = FakeVLM({})
        self.run_node(client)
        self.assertEqual(client.calls, [])
        self.assertEqual(self.output()['failed_regions'][0]['error'], 'bbox_outside_image')

    def test_model_input_is_rgb_id_and_scale_only(self):
        self.two_regions()
        client = FakeVLM({'Merged_1': answer('Merged_1'), 'Merged_2': answer('Merged_2')})
        self.run_node(client)
        call = next(item for item in client.calls if item['region_id'] == 'Merged_1')
        self.assertNotIn('[100, 100, 250, 250]', call['prompt'])
        self.assertNotIn('bbox', call['prompt'].split('Skill：')[0])
        self.assertIn('75 米', call['prompt'])
        self.assertIn('米/图像像素', call['prompt'])
        self.assertIn('潜在退化区域', call['prompt'])
        self.assertEqual(len(call['images']), 1)
        self.assertEqual(call['labels'], ['第 1 张：原始 RGB 影像'])
        self.assertEqual(call['schema']['properties']['region_id']['const'], 'Merged_1')
        self.assertEqual(call['system'], SYSTEM['prompt'])
        self.assertEqual(call['thinking'], (False, None))
        for item in client.calls:  # Merged_2 是 longitudinal_degradation，模型同样不应知道
            for word in ('degradation_type', 'longitudinal', '纵向退化', '退化类别'):
                self.assertNotIn(word, item['prompt'])
                self.assertNotIn(word, read_json(Path(item['images'][0]).parent/'input.json')['image_description'])
        self.assertEqual([r['degradation_type'] for r in self.output()['regions']], ['degradation', 'longitudinal_degradation'])

    def test_image_keeps_roi_aspect_and_marks_true_region(self):
        self.write_regions([{'patch_id': 'A', 'bbox': [100, 100, 200, 200], 'degradation_type': 'degradation'},
                            {'patch_id': 'B', 'bbox': [150, 150, 350, 250], 'degradation_type': 'degradation'}])
        client = FakeVLM({'Merged_1': answer('Merged_1')})
        self.run_node(client)
        rgb_path = client.calls[0]['images'][0]
        with Image.open(rgb_path) as image:
            self.assertEqual(image.size, (250, 150))
            rgb = image.convert('RGB')
            self.assertEqual(rgb.getpixel((100, 25)), lr.BOUNDARY_RGB)     # 全局 (200,125)：Patch A 右边界
            self.assertEqual(rgb.getpixel((200, 50)), lr.BOUNDARY_RGB)     # 全局 (300,150)：Patch B 上边界
            self.assertEqual(rgb.getpixel((200, 25)), BACKGROUND)          # 全局 (300,125)：bbox 内、并集外，保留原始 RGB
            self.assertEqual(rgb.getpixel((25, 25)), BACKGROUND)           # 区域内部不被着色
        metadata = read_json(Path(rgb_path).parent/'metadata.json')
        self.assertEqual(metadata['crop_global_bbox'], [100, 100, 350, 250])
        self.assertEqual(metadata['global_px_per_image_px'], [1., 1.])

    def test_large_region_is_downscaled_consistently(self):
        Image.new('RGB', (4000, 2000), BACKGROUND).save(self.root/'image.png')
        self.manifest['image'].update(width=4000, height=2000)
        self.write_regions([{'patch_id': 'A', 'bbox': [0, 0, 4000, 2000], 'degradation_type': 'degradation'}])
        client = FakeVLM({'Merged_1': answer('Merged_1')})
        self.run_node(client)
        with Image.open(client.calls[0]['images'][0]) as image:
            self.assertEqual(image.size, (1024, 512))
        metadata = read_json(Path(client.calls[0]['images'][0]).parent/'metadata.json')
        self.assertAlmostEqual(metadata['global_px_per_image_px'][0], 4000/1024)
        self.assertEqual(self.output()['regions'][0]['bbox'], [0, 0, 4000, 2000])

    def test_description_states_when_whole_frame_is_the_region(self):
        self.two_regions()  # Merged_1 为两个重叠 Patch，不规则；Merged_2 为单 Patch，边界即画面
        client = FakeVLM({'Merged_1': answer('Merged_1'), 'Merged_2': answer('Merged_2')})
        self.run_node(client)
        prompts = {call['region_id']: call['prompt'] for call in client.calls}
        self.assertNotIn('没有线外部分', prompts['Merged_1'])
        self.assertIn('线外只是周边环境', prompts['Merged_1'])
        self.assertIn('没有线外部分', prompts['Merged_2'])

    def test_hole_boundary_is_drawn(self):
        bars = [[100, 100, 200, 120], [180, 100, 200, 200], [100, 180, 200, 200], [100, 100, 120, 200]]
        self.write_regions([{'patch_id': f'R{i}', 'bbox': bar, 'degradation_type': 'degradation'} for i, bar in enumerate(bars)])
        client = FakeVLM({'Merged_1': answer('Merged_1')})
        self.run_node(client)
        with Image.open(client.calls[0]['images'][0]) as image:
            rgb = image.convert('RGB')
            self.assertEqual(rgb.getpixel((20, 50)), lr.BOUNDARY_RGB)   # 全局 (120,150)：洞的边界
            self.assertEqual(rgb.getpixel((50, 50)), BACKGROUND)        # 洞内保留原始 RGB

    def test_outputs_summary_and_manifest(self):
        self.two_regions()
        summary = self.run_node(FakeVLM({'Merged_1': answer('Merged_1', 'gps'), 'Merged_2': answer('Merged_2', 'wheel_imu_odometry')}))
        self.assertEqual(summary['method_counts'], {'gps': 1, 'wheel_imu_odometry': 1})
        self.assertNotIn('union_polygon', summary['regions'][0])
        manifest = read_json(self.manifest_path)
        self.assertEqual(manifest['extensions']['localization_recommendation'], summary)
        self.assertIn('localization_recommendations', manifest['files'])
        self.assertTrue((self.manifest_path.parents[1]/'visualization'/'14_localization_recommendation.png').is_file())
        self.assertEqual(self.output()['coordinate_system'], 'global_pixel')
        self.assertEqual(json.loads(json.dumps(self.output())), self.output())

    def test_dry_run_has_no_vlm_or_manifest_change(self):
        self.two_regions()
        client = FakeVLM({})
        before = self.manifest_path.read_bytes()
        summary = self.run_node(client, dry_run=True)
        self.assertEqual((summary['status'], client.calls), ('dry_run', []))
        self.assertEqual(self.manifest_path.read_bytes(), before)
        self.assertTrue((resolve_path(summary['results_dir'])/'Merged_1_prompt.txt').is_file())
        self.assertFalse((self.manifest_path.parent/'localization_recommendations.json').exists())

    def test_requires_node8_output(self):
        self.manifest['files'] = {}
        write_json(self.manifest_path, self.manifest)
        with self.assertRaisesRegex(ValueError, 'Node 8'):
            self.run_node(FakeVLM({}))

    def test_duplicate_merged_ids_are_rejected(self):
        regions = self.two_regions()
        write_json(self.regions_path, {'regions': [regions[0], deepcopy(regions[0])]})
        with self.assertRaisesRegex(ValueError, '不唯一'):
            self.run_node(FakeVLM({}))

    def test_node_skips_when_node8_skipped(self):
        update = lr.recommend_localization({'degradation_fusion': {'status': 'skipped'}}, config=self.config)
        self.assertEqual(update, {'localization_recommendation': {'status': 'skipped', 'reason': 'node8_skipped'}})

    def test_integration_node8_to_node9_to_manifest(self):
        extent = lambda block, xy: {'block_id': block, 'global_xy': xy, 'review_type': 'semantic', 'status': 'completed', 'skip_reason': None, 'max_size_m': 64,
                                    'tiers': [{'size_m': 64, 'state': '退化', 'degradation_type': '退化'}]}
        extents_path = self.manifest_path.parent/'degradation_extents.json'
        self.manifest['files'] = {'degradation_extents': str(extents_path)}
        write_json(self.manifest_path, self.manifest)
        write_json(extents_path, {'resolution_m_per_px': .5, 'extents': [extent('E0_B0', [200, 200]), extent('E0_B1', [250, 200]), extent('E5_B0', [800, 600])]})
        state = {'manifest_path': str(self.manifest_path), 'task_id': 't', 'output_dir': str(self.manifest_path.parents[1]), 'vlm_expansion': {'status': 'completed'}}
        progress = NodeProgress()
        update = _run_node(state, df.fuse_degraded_patches, progress, progress.add('fuse_degraded_patches'), config=self.config)
        self.assertEqual(update['degradation_fusion']['region_count'], 2)
        state = {**state, **update}
        client = FakeVLM({'Merged_1': answer('Merged_1', 'lidar_inertial_slam'), 'Merged_2': answer('Merged_2', 'gps')})
        update = _run_node(state, lr.recommend_localization, progress, progress.add('recommend_localization'), config=self.config, client=client)
        self.assertEqual(list(update), ['localization_recommendation'])
        self.assertEqual([r['recommended_method'] for r in update['localization_recommendation']['regions']], ['lidar_inertial_slam', 'gps'])
        state = {**state, **update}
        refresh_global_manifest(state)
        manifest = read_json(self.manifest_path)
        self.assertIn('fuse_degraded_patches', ''.join(manifest['files']['node_states']))
        self.assertIn('recommend_localization', ''.join(manifest['files']['node_states']))
        fused = read_json(self.manifest_path.parent/'degradation_regions.json')['regions']
        self.assertEqual([r['bbox'] for r in self.output()['regions']], [r['bbox'] for r in fused])

    def test_graph_order_and_run_until(self):
        nodes = list(build_graph(config=self.config).get_graph().nodes)
        self.assertLess(nodes.index('fuse_degraded_patches'), nodes.index('recommend_localization'))
        self.assertLess(nodes.index('recommend_localization'), nodes.index('refresh_global_manifest'))
        self.assertEqual(NODE_ORDER[-1], 'recommend_localization')
        stopped = list(build_graph(config=replace(self.config, run_until='fuse_degraded_patches')).get_graph().nodes)
        self.assertNotIn('recommend_localization', stopped)

    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, 'degradation_fusion_enabled'):
            GlobalPreprocessConfig(segmenter='dummy', vlm_localization_enabled=True)
        for name, value in (('vlm_localization_workers', 0), ('vlm_localization_max_retries', -1), ('vlm_localization_image_px', 64),
                            ('vlm_localization_dry_run', 'yes'), ('vlm_localization_thinking_budget', -1)):
            with self.assertRaisesRegex(ValueError, name):
                GlobalPreprocessConfig(segmenter='dummy', **{name: value})

    def test_cli_dry_run_exits_zero(self):
        from agent.map_preprocess.run_localization_recommendation import main as cli_main
        self.two_regions()
        self.assertEqual(cli_main(['--manifest', str(self.manifest_path), '--dry-run']), 0)


if __name__ == '__main__':
    unittest.main()
