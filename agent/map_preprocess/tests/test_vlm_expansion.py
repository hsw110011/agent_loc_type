"""Node 7 退化区域尺寸扩展离线回归：假 VLM + 合成图/DSM，不加载 Qwen、不需要 GPU。"""
from __future__ import annotations

import json
import threading
import unittest
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from jsonschema import Draft202012Validator, ValidationError
from PIL import Image

from agent.map_preprocess import load_manifest
from agent.map_preprocess.global_config import GlobalPreprocessConfig
from agent.map_preprocess.io import read_json, resolve_path, write_json
from agent.map_preprocess.nodes import vlm_expansion as ve
from agent.map_preprocess.nodes.vlm_review import run_vlm_review
from agent.map_preprocess.tests import test_block_features as node5
from agent.map_preprocess.tests.test_vlm_review import FakeVLM, output_for
from agent.map_preprocess.utils import vlm_expansion_inputs as ei
from agent.map_preprocess.utils import vlm_review_inputs as vi
from agent.vlm.block_expansion import (NOT_EVALUATED, bind_expansion, check_name, derive_tier, load_expansion_system,
                                       tier_key, walk_tiers)
from agent.vlm.block_review import load_review_system


def tier_for(kind: str, size: float, state: str = '退化') -> dict:
    """单档模型输出；state=退化/不退化/未知/纵向退化。"""
    def confident(**fields):
        return {**fields, **{f'{k}_confidence': '高' for k in fields}}
    longitudinal = confident(road_geometry='非直线', longitudinal_structure='有变化')
    if state == '纵向退化':
        longitudinal = confident(road_geometry='明显直线' if kind == 'semantic' else '近似直线', longitudinal_structure='重复')
    if kind == 'semantic':
        weak = state == '退化'
        check = {'dominant_class': '植被', **confident(dominance='明显占优' if weak else '无单类明显占优',
                                                      road_boundary='模糊' if weak else '清晰',
                                                      road_intrusion='存在' if weak else '无')}
    else:
        weak = state == '退化'
        check = {'relief': '平坦', **confident(raised_structure='无明显高出结构' if weak else '清晰',
                                              height_distinctiveness='低' if weak else '高'),
                 'height_comparisons': output_for('height', 'B')['height_check']['height_comparisons']}
    check.update({'evidence': 'x', **longitudinal, 'uncertainty': '无'})
    tier = {'size_m': round(size), 'new_ring': '环带', 'evidence': f'{size:g} 证据',
            'data_check': {'evidence': 'x', 'status': '可用', 'data_confidence': '高', 'uncertainty': '无'},
            check_name(kind): check}
    if kind == 'semantic':
        check.update(road_comparisons=[{'position': '外侧', 'rgb_basis': '直接可见', 'rgb_evidence': '可见铺装',
                                       'semantic_road': '缺失' if weak else '完整'}])
    check['longitudinal_comparisons'] = [{'position': '前段', 'observation': '左侧结构'},
                                        {'position': '后段', 'observation': '左侧结构'}] if state == '纵向退化' else []
    refs = ['A1', 'A2'] if state == '纵向退化' else ['R1' if kind == 'semantic' else 'H']
    check['verdict'] = {'result': '证据不足' if state == '未知' else state, 'reason': '外侧对照',
                        'evidence_refs': [] if state == '未知' else ['N', *refs]}
    if state == '未知':
        tier['data_check'].update(status='不确定')
    return tier


def answer_for(kind: str, block_id: str, states: dict) -> dict:
    return {'block_id': block_id, 'review_type': kind, 'schema_version': 4,
            'tiers': {tier_key(size): tier_for(kind, size, state) for size, state in states.items()}}


class ExpansionVLM(FakeVLM):
    def infer(self, prompt, images=(), *, system, max_tokens=None, temperature=0., json_output=False, output_schema=None,
              image_labels=None, enable_thinking=False, thinking_token_budget=None):
        block_id, _, kind = prompt.split('请评估区域 ')[1].split(' ')[:3]
        with self.lock:
            self.calls.append({'block_id': block_id, 'type': kind, 'system': system, 'prompt': prompt, 'schema': output_schema,
                               'image_paths': list(images), 'image_labels': image_labels, 'max_tokens': max_tokens})
        answer = self.answers[(block_id, kind)]
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(text=json.dumps(answer, ensure_ascii=False), usage={}, latency_seconds=.1, reasoning='')


class WalkTests(unittest.TestCase):
    sizes = [80, 100]

    def walk(self, degraded, valid=(80, 100), blocked=None):
        return walk_tiers(64, self.sizes, list(valid), degraded, blocked)

    def test_consecutive_degraded_reaches_largest_tier(self):
        best, stop, states = self.walk({80: True, 100: True})
        self.assertEqual((best, stop), (100, 'max_tier_reached'))
        self.assertEqual([states[s]['state'] for s in self.sizes], ['退化', '退化'])

    def test_first_tier_not_degraded_ignores_later_output(self):
        best, stop, states = self.walk({80: False, 100: True})
        self.assertEqual((best, stop), (64, 'no_longer_degraded'))
        self.assertEqual(states[80]['state'], '不退化')
        self.assertEqual(states[100], {'state': NOT_EVALUATED, 'not_evaluated_reason': 'after_stop'})

    def test_first_tier_unknown_stops_without_skipping(self):
        best, stop, states = self.walk({80: None, 100: True})
        self.assertEqual((best, stop), (64, 'insufficient_evidence'))
        self.assertEqual(states[80]['state'], '未知')
        self.assertEqual(states[100]['state'], NOT_EVALUATED)

    def test_second_tier_decides_largest_size(self):
        self.assertEqual(self.walk({80: True, 100: None})[:2], (80, 'insufficient_evidence'))
        self.assertEqual(self.walk({80: True, 100: False})[:2], (80, 'no_longer_degraded'))

    def test_boundary_removes_tiers_without_inventing_results(self):
        best, stop, states = self.walk({}, valid=())
        self.assertEqual((best, stop), (64, 'map_boundary'))
        self.assertEqual({s['not_evaluated_reason'] for s in states.values()}, {'map_boundary'})
        best, stop, states = self.walk({80: True}, valid=(80,))
        self.assertEqual((best, stop), (80, 'map_boundary'))
        self.assertEqual(states[100]['state'], NOT_EVALUATED)
        # 上一档已停止时，边界不改写停止原因
        self.assertEqual(self.walk({80: False}, valid=(80,))[:2], (64, 'no_longer_degraded'))

    def test_blocked_keeps_base(self):
        best, stop, states = self.walk({80: True, 100: True}, blocked='base_not_comparable')
        self.assertEqual((best, stop), (64, 'base_not_comparable'))
        self.assertEqual({s['state'] for s in states.values()}, {NOT_EVALUATED})


class PromptSystemTests(unittest.TestCase):
    def test_core_rule_is_reused_not_copied(self):
        for kind in ('semantic', 'height'):
            system = load_expansion_system(kind)
            review = load_review_system(kind)
            self.assertIs(type(system['core_rules']), dict)
            self.assertEqual(system['core_rules'], review['rules'])
            self.assertNotIn('degradation_decision', system['rules'])
            self.assertNotIn('\ndegradation_decision:', system['core_rule'])
            self.assertNotIn('drivable', system['core_rule'].split('degradation_decision')[0].replace('not_drivable', ''))
            for text in (system['prompt'], system['skill']):
                self.assertNotIn('{', text.replace('`', ''))

    def test_only_current_branch_expansion_checks_are_sent(self):
        import yaml
        for kind, other, expected in (('semantic', 'height', 'same_position'),
                                      ('height', 'semantic', 'height_layout')):
            system = load_expansion_system(kind)
            model_rules = yaml.safe_load(system['rule'])
            self.assertIn(expected, model_rules['branch_checks'][kind])
            self.assertNotIn(other, model_rules['branch_checks'])
            self.assertIn('证据不足', model_rules['stop'])

    def test_standalone_cli_preserves_expansion_enabled_for_fusion_manifest(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        from agent.map_preprocess import run_vlm_expansion as cli

        config = {'block_features_enabled': True, 'vlm_review_enabled': True,
                  'vlm_expansion_enabled': True, 'degradation_fusion_enabled': True, 'block_size_m': 75,
                  'vlm_expansion_image_px': 128}
        with patch.object(cli, 'load_manifest', return_value={'config': config}), \
                patch.object(ve, 'run_vlm_expansion', return_value={'status': 'dry_run'}) as run, \
                redirect_stdout(StringIO()):
            self.assertEqual(cli.main(['--manifest', 'manifest.json', '--dry-run']), 0)
        settings = run.call_args.args[1]
        self.assertTrue(settings.vlm_expansion_enabled)
        self.assertTrue(settings.degradation_fusion_enabled)
        self.assertEqual(tuple(settings.vlm_expansion_sizes_m), (100.,))
        self.assertEqual(settings.vlm_expansion_image_px, GlobalPreprocessConfig().vlm_expansion_image_px)

    def test_bound_schema_uses_node6_fields_and_valid_tiers_only(self):
        for kind in ('semantic', 'height'):
            system = load_expansion_system(kind)
            name = check_name(kind)
            schema = bind_expansion(system, 'E1_B0', kind, [80])
            Draft202012Validator.check_schema(schema)
            self.assertNotIn('$defs', schema)
            tiers = schema['properties']['tiers']['properties']
            self.assertEqual(list(tiers), ['size_80'])
            self.assertEqual(list(tiers['size_80']['properties']), ['size_m', 'new_ring', 'data_check', name, 'evidence'])
            for field in ('data_check', name):
                self.assertEqual(tiers['size_80']['properties'][field], system['review_schema']['properties'][field])
            good = answer_for(kind, 'E1_B0', {80: '退化'})
            Draft202012Validator(schema).validate(good)
            for bad in (answer_for(kind, 'E1_B0', {80: '退化', 100: '退化'}), answer_for(kind, 'E2_B0', {80: '退化'})):
                with self.assertRaises(ValidationError):
                    Draft202012Validator(schema).validate(bad)
            both = list(bind_expansion(system, 'E1_B0', kind, [80, 100])['properties']['tiers']['required'])
            self.assertEqual(both, ['size_80', 'size_100'])

    def test_model_cannot_output_conclusions(self):
        system = load_expansion_system('height')
        schema = bind_expansion(system, 'E1_B0', 'height', [80])
        answer = answer_for('height', 'E1_B0', {80: '退化'})
        for extra in ('max_size_m', 'degraded', 'degradation_type'):
            polluted = deepcopy(answer)
            polluted['tiers']['size_80'][extra] = 1
            with self.assertRaises(ValidationError):
                Draft202012Validator(schema).validate(polluted)

    def test_tier_uses_same_rule_as_node6(self):
        for kind in ('semantic', 'height'):
            rules = load_review_system(kind)['rules']
            self.assertIs(derive_tier(tier_for(kind, 80), kind, '道路', rules)['degraded'], True)
            self.assertIs(derive_tier(tier_for(kind, 80, '不退化'), kind, '道路', rules)['degraded'], False)
            self.assertIsNone(derive_tier(tier_for(kind, 80, '未知'), kind, '道路', rules)['degraded'])

    def test_reason_can_change_between_tiers(self):
        """原原因消失而其它条件成立时仍退化：基准是侵入，当前档只有纵向重复。"""
        rules = load_review_system('semantic')['rules']
        derived = derive_tier(tier_for('semantic', 80, '纵向退化'), 'semantic', '道路', rules)
        self.assertIs(derived['degraded'], True)
        self.assertEqual(derived['degradation_type'], '纵向退化')
        self.assertEqual(derived['matched_reasons'], ['vlm_verdict'])

    def test_legacy_height_tier_without_verdict_uses_confidence(self):
        rules = load_review_system('height')['rules']
        tier = tier_for('height', 80, '不退化')
        del tier['height_check']['verdict']
        tier['height_check']['height_distinctiveness_confidence'] = '低'
        self.assertIsNone(derive_tier(tier, 'height', '道路', rules)['degraded'])
        tier = tier_for('height', 80, '不退化')
        del tier['height_check']['verdict']
        tier['data_check']['data_confidence'] = '低'
        self.assertIsNone(derive_tier(tier, 'height', '道路', rules)['degraded'])

    def test_inherited_area_type_participates_in_consistency(self):
        """广场继承“不适用”合法；道路填“不适用”被屏蔽，新版仍以模型结论为准。"""
        rules = load_review_system('semantic')['rules']
        tier = tier_for('semantic', 80, '不退化')
        tier['degradation_check'].update(road_geometry='不适用', longitudinal_structure='不适用')
        self.assertIs(derive_tier(tier, 'semantic', '铺装广场', rules)['degraded'], False)
        derived = derive_tier(tier, 'semantic', '道路', rules)
        self.assertIs(derived['degraded'], False)
        self.assertTrue(derived['consistency_failures'])


class CoverageTests(unittest.TestCase):
    def test_only_edge_connected_blank_is_outside(self):
        blank = np.zeros((20, 20), bool)
        blank[:, :3] = True        # 地图外：连通栅格边缘
        blank[10:12, 10:12] = True  # 地图内部空洞
        outside = ei._outside(blank)
        self.assertTrue(outside[:, :3].all())
        self.assertFalse(outside[10:12, 10:12].any())

    def test_undeclared_black_and_white_fill_is_blank(self):
        rgb = np.full((4, 4, 3), 90, np.uint8)
        rgb[0] = 0
        rgb[1] = 255
        rgb[2, 0] = (0, 0, 9)
        rgb[3, 0] = (255, 255, 200)
        blank = ei.rgb_blank(rgb, np.ones((4, 4), bool))
        self.assertTrue(blank[:2].all())
        self.assertFalse(blank[2:].any())
        mask = np.ones((4, 4), bool)
        mask[3, 3] = False
        self.assertTrue(ei.rgb_blank(rgb, mask)[3, 3])

    def test_uncovered_checks_branch_sources_only(self):
        full = np.ones((10, 10), bool)
        holes = full.copy()
        holes[0, 0] = False
        grids = {'factor': 10, 'rgb': full, 'semantic': full, 'dsm': holes}
        self.assertEqual(ei.uncovered(grids, 'semantic', [0, 0, 30, 30], (100, 100)), [])
        self.assertEqual(ei.uncovered(grids, 'height', [0, 0, 30, 30], (100, 100)), ['dsm'])
        self.assertEqual(ei.uncovered(grids, 'height', [20, 20, 50, 50], (100, 100)), [])
        self.assertEqual(ei.uncovered(grids, 'semantic', [-1, 0, 30, 30], (100, 100)), ['raster_bounds'])
        self.assertEqual(ei.uncovered(grids, 'semantic', [70, 0, 101, 30], (100, 100)), ['raster_bounds'])

    def test_interior_hole_does_not_invalidate_window(self):
        """地图内部无数据不算地图外：足迹保留，由数据质量规则处理。"""
        valid = np.ones((10, 10), bool)
        valid[4:6, 4:6] = False
        grids = {'factor': 10, 'rgb': ~ei._outside(~valid), 'semantic': ~ei._outside(~valid)}
        self.assertEqual(ei.uncovered(grids, 'semantic', [20, 20, 80, 80], (100, 100)), [])

    def test_frames_are_concentric_and_inside_image(self):
        outer, inner = [100, 100, 200, 200], [118, 118, 182, 182]
        self.assertEqual(ei.frame_box(outer, outer, 100), [2., 2., 97., 97.])
        box = ei.frame_box(inner, outer, 100)
        self.assertAlmostEqual((box[0]+box[2])/2, 50)
        image = Image.new('RGB', (100, 100), (60, 120, 60))
        ei.draw_tier_frames(image, [(64, box), (100, ei.frame_box(outer, outer, 100))])
        array = np.asarray(image)
        self.assertTrue((array[50, 50] == (60, 120, 60)).all())  # 框内部不填色
        self.assertEqual(array[int(box[1]), 50].tolist(), [255, 255, 255])

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            GlobalPreprocessConfig(vlm_expansion_enabled=True)
        base = {'block_features_enabled': True, 'vlm_review_enabled': True, 'vlm_expansion_enabled': True, 'block_size_m': 64}
        GlobalPreprocessConfig(**base)
        for bad in ({'vlm_expansion_sizes_m': (100, 80)}, {'vlm_expansion_sizes_m': (64, 80)}, {'vlm_expansion_sizes_m': ()},
                    {'vlm_expansion_max_tokens': 1024, 'vlm_expansion_thinking_budget': 2048}):
            with self.assertRaises(ValueError):
                GlobalPreprocessConfig(**base, **bad)


class PipelineTests(node5.BlockFeatureTests):
    """合成图背景改成非纯黑：纯黑与栅格边缘连通时按地图外处理。"""

    def setUp(self):
        super().setUp()
        rgb = np.array(Image.open(self.image))
        rgb[(rgb == 0).all(axis=2)] = (30, 30, 30)
        Image.fromarray(rgb).save(self.image)
        self.config = replace(self.config, block_size_m=64., vlm_review_enabled=False, vlm_review_image_px=128,
                              vlm_expansion_sizes_m=(80., 100.), vlm_expansion_image_px=128, vlm_expansion_workers=2)

    def reviewed(self, states: dict, centers: dict | None = None) -> tuple[str, dict]:
        """states: block_id → {review_type: 'degraded'|'normal'|'unknown'|'error'}，跑完 Node 6；centers 把块中心放到指定位置。"""
        state, manifest = self.run_map(self.config)
        path = state['manifest_path']
        for block in manifest['blocks']:
            block['global_xy'] = list((centers or {}).get(block['block_id'], block['global_xy']))
            wanted = states.get(block['block_id'], {})
            d = block['features']['degradation']
            d['semantic_degraded'], d['height_degraded'] = 'semantic' in wanted, 'height' in wanted
        write_json(path, manifest)
        answers = {}
        for block_id, kinds in states.items():
            for kind, result in kinds.items():
                if result == 'error':
                    answers[(block_id, kind)] = RuntimeError('失败')
                    continue
                answers[(block_id, kind)] = output_for(kind, block_id, degraded=result == 'degraded')
                if result == 'unknown':
                    answers[(block_id, kind)]['data_check']['status'] = '不确定'
        run_vlm_review(path, self.config, client=FakeVLM(answers), semantic_segmenter=node5.FakeSemantic(self.events))
        return path, load_manifest(path)

    def expand(self, path, answers, **kwargs):
        client = ExpansionVLM(answers)
        summary = ve.run_vlm_expansion(path, self.config, client=client, semantic_segmenter=node5.FakeSemantic(self.events), **kwargs)
        extents = {} if summary['status'] == 'dry_run' else read_json(resolve_path(summary['degradation_extents']))
        return summary, client, {(e['block_id'], e['review_type']): e for e in extents.get('extents', [])}

    def interior_blocks(self, manifest, count=2):
        """100 m 窗口完整落在图内的块。"""
        size = (manifest['image']['width'], manifest['image']['height'])
        blocks = [b for b in manifest['blocks'] if self.inside(b, 100, manifest, size)]
        self.assertGreaterEqual(len(blocks), count)
        return [b['block_id'] for b in blocks[:count]]

    @staticmethod
    def inside(block, size_m, manifest, size):
        x0, y0, x1, y1 = vi.review_window(block, size_m, manifest['resolution_m_per_px'])
        return x0 >= 0 and y0 >= 0 and x1 <= size[0] and y1 <= size[1]

    def block_ids(self):
        state, manifest = self.run_map(self.config)
        return self.interior_blocks(manifest), manifest

    def test_branches_are_independent_and_inputs_match(self):
        (one, two), _ = self.block_ids()
        path, before = self.reviewed({one: {'semantic': 'degraded', 'height': 'degraded'}, two: {'height': 'degraded'}})
        answers = {(one, 'semantic'): answer_for('semantic', one, {80: '退化', 100: '退化'}),
                   (one, 'height'): answer_for('height', one, {80: '不退化', 100: '退化'}),
                   (two, 'height'): answer_for('height', two, {80: '退化', 100: '未知'})}
        summary, client, extents = self.expand(path, answers)
        semantic, height = extents[(one, 'semantic')], extents[(one, 'height')]
        self.assertEqual((semantic['max_size_m'], semantic['stop_reason'], semantic['stop_label']), (100, 'max_tier_reached', '达到最大档位'))
        self.assertEqual((height['max_size_m'], height['stop_reason']), (64, 'no_longer_degraded'))
        self.assertEqual((extents[(two, 'height')]['max_size_m'], extents[(two, 'height')]['stop_reason']), (80, 'insufficient_evidence'))
        # 基准继承 Node 6，停止档位之后不采纳
        tiers = {t['size_m']: t for t in height['tiers']}
        self.assertEqual((tiers[64]['state'], tiers[64]['source']), ('退化', 'node6'))
        self.assertEqual(tiers[64]['field_confidence']['raised_structure_confidence'], '高')
        self.assertEqual(tiers[64]['confirmed_reasons'], ['vlm_verdict'])
        self.assertEqual(tiers[80]['state'], '不退化')
        self.assertEqual(tiers[100], {'size_m': 100, 'state': NOT_EVALUATED, 'source': None, 'not_evaluated_reason': 'after_stop'})
        self.assertEqual(summary['job_count'], 3)
        for call in client.calls:
            self.assertEqual([p.name for p in call['image_paths']], ['rgb.png', f"{call['type']}.png"])
            sizes = set()
            for image_path in call['image_paths']:
                with Image.open(image_path) as image:
                    sizes.add(image.size)
            self.assertEqual(sizes, {(128, 128)})
            self.assertEqual(call['system'], load_expansion_system(call['type'])['prompt'])
            self.assertIn('核心 Rule', call['prompt'])
            self.assertEqual(list(call['schema']['properties']['tiers']['required']), ['size_80', 'size_100'])
            self.assertEqual(call['max_tokens'], self.config.vlm_expansion_max_tokens)
        # 块、块尺寸和 Node 6 结论不变
        after = load_manifest(path)
        self.assertEqual(after['blocks'], before['blocks'])
        self.assertEqual(after['format_version'], 2)
        self.assertEqual(after['extensions']['vlm_review'], before['extensions']['vlm_review'])
        self.assertIn('vlm_expansion', after['extensions'])
        self.assertTrue(resolve_path(after['files']['vlm_expansion_overview']).is_file())

    def test_model_input_omits_local_geometry_and_statistics(self):
        (one, _), _ = self.block_ids()
        path, _ = self.reviewed({one: {'semantic': 'degraded', 'height': 'degraded'}})
        summary, client, _ = self.expand(path, {(one, k): answer_for(k, one, {80: '退化', 100: '退化'}) for k in ('semantic', 'height')})
        for call in client.calls:
            kind = call['type']
            sent = read_json(resolve_path(summary['inputs_dir'])/one/f'{kind}_input.json')
            self.assertEqual(set(sent), {'block_id', 'review_type', 'image_description'})
            metadata = read_json(resolve_path(summary['inputs_dir'])/one/f'{kind}_metadata.json')
            description = sent['image_description']
            self.assertIn(sent['image_description'], call['prompt'])
            for word in ('实线=64 m', '短虚线=80 m', '点线=100 m', '有效评估档位：80 m、100 m', '继承的地面类型：道路'):
                self.assertIn(word, description)
            for key in ('image_global_bbox', 'center_image_xy', 'ring_nodata_pct', 'windows'):
                self.assertIn(key, metadata)
            self.assertNotIn(str(metadata['image_global_bbox'][0]), description)
            self.assertNotIn('%', description)
            for prior in ('已确认退化', '证据摘要', metadata['node6']['evidence']):
                self.assertNotIn(prior, description)
            self.assertEqual(metadata['image_extent_m'], [100., 100.])
            self.assertIn('从原始卫星 RGB 裁出并缩放', description)
            self.assertIn('中心十字、白芯黑边框和尺寸文字都由程序添加', description)
            self.assertIn('新增环带位于本档框和上一档框之间', description)
            self.assertIn('两侧缺少足够稳定、可区分的对应结构', call['system'])
            if kind == 'height':
                self.assertIsNotNone(metadata['dsm_reference']['h_ref_m'])
                self.assertEqual(metadata['dsm_color_range_m'], list(self.config.vlm_review_dsm_range_m))
                self.assertIn('与 Node 6 相同的周边低处着色基准', description)
                self.assertIn('白色=无数据', description)
                self.assertIn('最大高度', call['system'])
            else:
                self.assertIn('从整幅卫星图的自动语义分割结果裁出', description)
                self.assertIn('只使用道路、植被、建筑三类', call['system'])
            self.assertNotIn('中心线', description)   # 输入图不画道路拓扑线

    def test_map_boundary_removes_frames_and_skips_vlm(self):
        (one, two), _ = self.block_ids()
        # 左边界距离 34 px：64 m 完整、80 m 越界；距离 42 px：80 m 完整、100 m 越界（1 m/px）
        path, _ = self.reviewed({one: {'height': 'degraded'}, two: {'height': 'degraded'}}, {one: (34, 350), two: (42, 350)})
        summary, client, extents = self.expand(path, {(two, 'height'): answer_for('height', two, {80: '退化'})})
        self.assertEqual([c['block_id'] for c in client.calls], [two])
        self.assertEqual((extents[(one, 'height')]['max_size_m'], extents[(one, 'height')]['stop_reason']), (64, 'map_boundary'))
        self.assertEqual(extents[(one, 'height')]['valid_tiers_m'], [])
        self.assertEqual({t['state'] for t in extents[(one, 'height')]['tiers'][1:]}, {NOT_EVALUATED})
        self.assertEqual((extents[(two, 'height')]['max_size_m'], extents[(two, 'height')]['stop_reason']), (80, 'map_boundary'))
        self.assertEqual(extents[(two, 'height')]['valid_tiers_m'], [80])
        self.assertEqual(extents[(two, 'height')]['tiers'][2]['not_evaluated_reason'], 'map_boundary')
        metadata = read_json(resolve_path(summary['inputs_dir'])/two/'height_metadata.json')
        self.assertEqual(sorted(metadata['frames']), ['64', '80'])
        self.assertEqual(metadata['image_extent_m'], [80., 80.])
        self.assertEqual(list(client.calls[0]['schema']['properties']['tiers']['required']), ['size_80'])
        self.assertNotIn('100 m', read_json(resolve_path(summary['inputs_dir'])/two/'height_input.json')['image_description'])

    def test_branch_coverage_differs_by_required_data(self):
        """DSM 只覆盖部分地图：height 被边界剔除，同位置 semantic 仍可扩展。"""
        (one, _), _ = self.block_ids()
        path, _ = self.reviewed({one: {'semantic': 'degraded', 'height': 'degraded'}}, {one: (450, 350)})
        original = ei.coverage_grids

        def narrow_dsm(*args):
            grids = original(*args)
            grids['dsm'] = grids['dsm'].copy()
            grids['dsm'][:, 485//grids['factor']:] = False
            return grids
        ei.coverage_grids = narrow_dsm
        self.addCleanup(setattr, ei, 'coverage_grids', original)
        answers = {(one, 'semantic'): answer_for('semantic', one, {80: '退化', 100: '退化'})}
        _, client, extents = self.expand(path, answers)
        self.assertEqual([c['type'] for c in client.calls], ['semantic'])
        self.assertEqual(extents[(one, 'semantic')]['max_size_m'], 100)
        self.assertEqual(extents[(one, 'height')]['stop_reason'], 'map_boundary')
        self.assertEqual(extents[(one, 'height')]['tier_coverage'], {'80': ['dsm']})

    def test_incomplete_base_is_skipped_and_node6_kept(self):
        (one, two), _ = self.block_ids()
        # 中心距左边界 20 px：Node 6 只覆盖 52 m 宽；另一块 Node 6 完整但基准落在 DSM 足迹外
        path, before = self.reviewed({one: {'height': 'degraded'}, two: {'height': 'degraded'}}, {one: (20, 350), two: (450, 350)})
        self.assertLess(before['blocks'][0]['vlm_review']['reviews']['height']['coverage']['image_extent_m'][0], 64)
        original = ei.coverage_grids

        def narrow_dsm(*args):
            grids = original(*args)
            grids['dsm'] = grids['dsm'].copy()
            grids['dsm'][:, 460//grids['factor']:] = False
            return grids
        ei.coverage_grids = narrow_dsm
        self.addCleanup(setattr, ei, 'coverage_grids', original)
        summary, client, extents = self.expand(path, {})
        self.assertEqual(client.calls, [])
        for block_id in (one, two):   # 不扩展、不调用 VLM，但保留 Node 6 的基准结论进入融合
            result = extents[(block_id, 'height')]
            self.assertEqual((result['status'], result['skip_reason'], result['max_size_m'], result['stop_reason']),
                             ('completed', None, 64, 'base_incomplete'))
            self.assertEqual(result['tiers'][0]['source'], 'node6')
            self.assertEqual({t['state'] for t in result['tiers'][1:]}, {NOT_EVALUATED})
            self.assertIs(result['node6']['degraded'], True)
        self.assertEqual(extents[(two, 'height')]['base_problem'], {'missing_sources': ['dsm']})
        self.assertEqual(summary['skipped'], {})
        self.assertEqual(summary['completed_count'], 2)
        self.assertEqual(load_manifest(path)['blocks'], before['blocks'])

    def test_node6_unconfirmed_branches_do_not_enter(self):
        (one, two), _ = self.block_ids()
        path, _ = self.reviewed({one: {'semantic': 'degraded', 'height': 'unknown'}, two: {'semantic': 'normal', 'height': 'error'}})
        _, client, extents = self.expand(path, {(one, 'semantic'): answer_for('semantic', one, {80: '退化', 100: '退化'})})
        self.assertEqual([(c['block_id'], c['type']) for c in client.calls], [(one, 'semantic')])
        for key in ((one, 'height'), (two, 'semantic'), (two, 'height')):
            self.assertEqual((extents[key]['status'], extents[key]['skip_reason'], extents[key]['max_size_m']),
                             ('skipped', 'node6_not_confirmed', None))

    def test_reason_change_is_recorded_per_tier(self):
        (one, _), _ = self.block_ids()
        path, _ = self.reviewed({one: {'semantic': 'degraded'}})
        answer = answer_for('semantic', one, {80: '纵向退化', 100: '不退化'})
        _, _, extents = self.expand(path, {(one, 'semantic'): answer})
        result = extents[(one, 'semantic')]
        tiers = {t['size_m']: t for t in result['tiers']}
        self.assertEqual(result['max_size_m'], 80)
        self.assertEqual(tiers[64]['degradation_type'], '退化')
        self.assertEqual((tiers[80]['degradation_type'], tiers[80]['confirmed_reasons']), ('纵向退化', ['vlm_verdict']))
        self.assertEqual(tiers[80]['evidence'], '80 证据')
        self.assertEqual(tiers[100]['state'], '不退化')

    def test_request_failure_keeps_base_and_dry_run_has_no_side_effects(self):
        (one, _), _ = self.block_ids()
        path, before = self.reviewed({one: {'semantic': 'degraded'}})
        summary, client, _ = self.expand(path, {}, dry_run=True)
        self.assertEqual((summary['status'], client.calls), ('dry_run', []))
        self.assertEqual(load_manifest(path), before)
        self.assertTrue((resolve_path(summary['results_dir'])/f'{one}_semantic_prompt.txt').is_file())
        summary, _, extents = self.expand(path, {(one, 'semantic'): RuntimeError('超时')})
        result = extents[(one, 'semantic')]
        self.assertEqual((result['max_size_m'], result['stop_reason'], result['stop_label']), (64, 'request_failed', '请求失败'))
        self.assertEqual(summary['errors'][0]['block_id'], one)

    def test_diagnostics_leave_publication_intact_and_full_rerun_invalidates_downstream(self):
        (one, two), _ = self.block_ids()
        path, _ = self.reviewed({one: {'semantic': 'degraded'}, two: {'semantic': 'degraded'}})
        answers = {(identifier, 'semantic'): answer_for('semantic', identifier, {80: '退化', 100: '退化'})
                   for identifier in (one, two)}
        initial, _, _ = self.expand(path, answers)
        manifest_path = resolve_path(path)
        manifest = load_manifest(path)
        manifest['extensions'].update(degradation_fusion={'status': 'completed', 'region_count': 1},
                                      localization_recommendation={'status': 'completed'})
        manifest['files'].update(degradation_regions='old_regions.json', localization_recommendations='old_recommendations.json')
        manifest['final_summary'] = {'conclusion': 'degraded', 'complete': True}
        write_json(path, manifest)
        state_path = manifest_path.with_name('state.json')
        write_json(state_path, {**manifest['extensions'], 'final_summary': manifest['final_summary']})
        watched = [manifest_path, state_path, resolve_path(initial['path']), resolve_path(initial['degradation_extents']),
                   resolve_path(initial['results_dir'])/f'{two}_semantic.json']
        snapshots = {p: p.read_bytes() for p in watched}

        self.expand(path, {}, dry_run=True)
        _, _, partial = self.expand(path, answers, block_ids=[one])
        self.assertEqual(set(partial), {(one, 'semantic')})
        self.assertEqual({p: p.read_bytes() for p in watched}, snapshots)

        _, client, complete = self.expand(path, {}, resume=True)
        self.assertEqual(client.calls, [])
        self.assertEqual(set(complete), set(answers))
        current = load_manifest(path)
        for key in ('degradation_regions', 'localization_recommendations'):
            self.assertNotIn(key, current['files'])
        self.assertEqual(current['final_summary']['conclusion'], 'incomplete')
        self.assertEqual(read_json(state_path)['final_summary'], current['final_summary'])
        self.assertNotIn('localization_recommendation', read_json(state_path))

    def test_resume_reuses_successful_requests_only(self):
        (one, two), _ = self.block_ids()
        path, _ = self.reviewed({one: {'height': 'degraded'}, two: {'height': 'degraded'}})
        first = {(one, 'height'): answer_for('height', one, {80: '退化', 100: '不退化'}), (two, 'height'): RuntimeError('中断')}
        _, _, extents = self.expand(path, first)
        self.assertEqual(extents[(two, 'height')]['stop_reason'], 'request_failed')
        second = {(two, 'height'): answer_for('height', two, {80: '退化', 100: '退化'})}
        summary, client, extents = self.expand(path, second, resume=True)
        self.assertEqual([c['block_id'] for c in client.calls], [two])
        self.assertEqual((extents[(one, 'height')]['max_size_m'], extents[(one, 'height')]['stop_reason']), (80, 'no_longer_degraded'))
        self.assertEqual((extents[(two, 'height')]['max_size_m'], extents[(two, 'height')]['stop_reason']), (100, 'max_tier_reached'))
        self.assertEqual(summary['errors'], [])
        # Prompt 版本更新后，同档旧结果也要重跑。
        cached = resolve_path(summary['results_dir'])/f'{one}_height.json'
        old = read_json(cached)
        old.pop('pipeline_version')
        write_json(cached, old)
        refreshed = {(one, 'height'): answer_for('height', one, {80: '退化', 100: '退化'})}
        _, client, extents = self.expand(path, refreshed, resume=True)
        self.assertEqual([c['block_id'] for c in client.calls], [one])
        self.assertEqual(extents[(one, 'height')]['max_size_m'], 100)

        # 档位、流程版本相同，但色带变化使实际输入 PNG 不同，两条支路都应重新请求。
        self.config = replace(self.config, vlm_review_dsm_range_m=(-4., 10.))
        _, client, _ = self.expand(path, {**refreshed, **second}, resume=True)
        self.assertEqual({c['block_id'] for c in client.calls}, {one, two})

    def test_invalid_output_is_not_adopted(self):
        (one, _), _ = self.block_ids()
        path, _ = self.reviewed({one: {'height': 'degraded'}})
        answer = answer_for('height', one, {80: '退化', 100: '退化'})
        answer['tiers']['size_80']['height_check']['raised_structure'] = '自编枚举'
        _, _, extents = self.expand(path, {(one, 'height'): answer})
        self.assertEqual((extents[(one, 'height')]['max_size_m'], extents[(one, 'height')]['stop_reason']), (64, 'request_failed'))

    def test_all_missing_ring_cannot_be_confirmed(self):
        kind = 'height'
        system = load_expansion_system(kind)
        payload = {'area_type': '道路', 'ring_nodata_pct': {'80': 100.}}
        for status in ('可用', '不可用'):
            tier = tier_for(kind, 80)
            tier['data_check']['status'] = status
            judged = ve.judge_tier(tier, kind, payload, system, 80)
            self.assertIsNone(judged['degraded'])
            self.assertEqual(judged['reason_code'], 'ring_no_data')
        tier = tier_for(kind, 80)
        tier['height_check']['verdict']['evidence_refs'] = ['H']
        judged = ve.judge_tier(tier, kind, {**payload, 'ring_nodata_pct': {}}, system, 80)
        self.assertIsNone(judged['degraded'])
        self.assertEqual(judged['reason_code'], 'invalid_evidence_refs')

    def test_height_without_reference_stops_without_vlm(self):
        (one, _), _ = self.block_ids()
        path, _ = self.reviewed({one: {'height': 'degraded'}})
        original = vi.dsm_reference
        vi.dsm_reference = lambda *args: (None, 0.)   # 100 m 范围内没有有效 DSM
        self.addCleanup(setattr, vi, 'dsm_reference', original)
        _, client, extents = self.expand(path, {})
        result = extents[(one, 'height')]
        self.assertEqual(client.calls, [])
        self.assertEqual((result['status'], result['max_size_m'], result['stop_reason']), ('completed', 64, 'base_not_comparable'))
        self.assertIsNone(result['dsm_reference']['h_ref_m'])
        self.assertEqual(result['tiers'][0]['state'], '退化')

    def test_semantic_map_uses_same_labels_for_every_tier(self):
        """语义图只裁一次整图标签：基准区域的类别与 Node 6 同窗口一致。"""
        (one, _), manifest = self.block_ids()
        block = next(b for b in manifest['blocks'] if b['block_id'] == one)
        labels = np.zeros((manifest['image']['height'], manifest['image']['width']), np.uint8)
        labels[:, ::7] = 2
        size = (manifest['image']['width'], manifest['image']['height'])
        window = vi.review_window(block, 64, 1.)
        outer = vi.review_window(block, 100, 1.)
        small = vi.crop_labels(labels, size, window, 64)
        large = vi.crop_labels(labels, size, outer, 100)
        offset = window[0]-outer[0], window[1]-outer[1]
        self.assertTrue((large[offset[1]:offset[1]+64, offset[0]:offset[0]+64] == small).all())

    def test_pipeline_runs_node7_after_node6(self):
        from agent.map_preprocess.graph import build_graph
        config = replace(self.config, vlm_review_enabled=True, vlm_expansion_enabled=True)
        nodes = list(build_graph(config=config).get_graph().nodes)
        self.assertLess(nodes.index('vlm_review_blocks'), nodes.index('vlm_expand_blocks'))
        self.assertLess(nodes.index('vlm_expand_blocks'), nodes.index('refresh_global_manifest'))


# 继承的 Node 5 用例在本文件不重复运行
for _name in [n for n in dir(node5.BlockFeatureTests) if n.startswith('test_')]:
    setattr(PipelineTests, _name, None)


if __name__ == '__main__':
    unittest.main()
