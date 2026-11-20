"""Node 6 退化块 VLM 复核离线回归：假 VLM 客户端 + 合成图/DSM，不加载 Qwen、不需要 GPU。"""
from __future__ import annotations

import json
import threading
import unittest
import warnings
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from jsonschema import ValidationError

from agent.map_preprocess import load_manifest
from agent.map_preprocess.io import resolve_path, write_json
import yaml

from agent.map_preprocess.nodes.vlm_review import backup_path, decide, derive, review_types, run_vlm_review, scene_record
from agent.map_preprocess.utils import block_features as bf
from agent.map_preprocess.utils import vlm_review_inputs as vi
from agent.map_preprocess.tests import test_block_features as node5
from agent.vlm.block_review import load_review_system


def output_for(review_type: str, block_id: str, *, is_road=True, degraded=True) -> dict:
    def check(**fields):
        return {'evidence': 'x', **fields, 'uncertainty': '无'}
    def judged(**fields):
        return {**fields, **{f'{k}_confidence': '高' for k in fields}}
    detail = (check(dominant_class='植被', **judged(dominance='明显占优' if degraded else '无单类明显占优',
                                                   road_boundary='模糊' if degraded else '清晰',
                                                   road_intrusion='存在' if degraded else '无'))
              if review_type == 'semantic' else
              check(relief='平坦' if degraded else '起伏多样',
                    **judged(raised_structure='无明显高出结构' if degraded else '清晰',
                             height_distinctiveness='低' if degraded else '高')))
    detail.update(judged(road_geometry='明显直线' if review_type == 'semantic' else '近似直线', longitudinal_structure='有变化'))
    if review_type == 'height':
        detail['height_comparisons'] = {
            'road_and_surroundings': '对照路面和路侧结构的颜色',
            'along_road': '比较沿路前后两处',
            'overhead_occlusion': '未见俯视遮挡迹象',
        }
    second = 'semantic' if review_type == 'semantic' else 'height'
    return {'block_id': block_id, 'review_type': review_type, 'schema_version': 9 if review_type == 'height' else 8,
            'observations': {'rgb': 'x', second: 'x', 'context': 'x'},
            'drivable_check': check(**judged(area_type='道路' if is_road else '不可通行')),
            'data_check': check(status='可用', data_confidence='高'),
            'degradation_check' if review_type == 'semantic' else 'height_check': detail}



def current_output(output):
    """把旧规则测试的观察样本转成当前请求格式。"""
    from copy import deepcopy
    output = deepcopy(output)
    result = derive(output)['degradation_type']
    height = output['review_type'] == 'height'
    output['schema_version'] = 12 if height else 11
    output.pop('drivable_check', None)
    check = output['height_check' if height else 'degradation_check']
    check['longitudinal_comparisons'] = [
        {'position': '前段', 'observation': '左侧高出结构'},
        {'position': '后段', 'observation': '左侧高出结构'},
    ] if result == '纵向退化' else []
    if not height:
        check['road_comparisons'] = [{'position': '中段', 'rgb_basis': '直接可见',
                                      'rgb_evidence': '可见铺装', 'semantic_road': '缺失' if result == '退化' else '完整'}]
    refs = ['A1', 'A2'] if result == '纵向退化' else ['H' if height else 'R1']
    check['verdict'] = {'result': '证据不足' if result == '未知' else result,
                        'reason': '根据窗口内的对照', 'evidence_refs': [] if result == '未知' else refs}
    return output


class SchemaTests(unittest.TestCase):
    def test_two_independent_systems(self):
        semantic, height = (load_review_system(k) for k in ('semantic', 'height'))
        self.assertNotEqual(semantic['prompt'], height['prompt'])
        self.assertNotEqual(semantic['rule'], height['rule'])
        self.assertIn('degradation_check', semantic['schema']['properties'])
        self.assertIn('height_check', height['schema']['properties'])
        self.assertNotIn('height_check', semantic['schema']['properties'])
        # 每张输入图各有一个观察字段，顺序与图片顺序一致
        self.assertEqual(list(semantic['schema']['properties']['observations']['properties']), ['rgb', 'semantic', 'context'])
        self.assertEqual(list(height['schema']['properties']['observations']['properties']), ['context', 'rgb', 'height'])
        self.assertIn('DSM 伪彩色高程图', height['prompt'])
        self.assertNotIn('DSM 伪彩色高程图', semantic['prompt'])
        self.assertNotIn('语义分割图', height['prompt'])
        for system in (semantic, height):
            yaml.safe_load(system['rule'])

    def test_schema_has_no_derived_or_conditional_fields(self):
        for kind in ('semantic', 'height'):
            schema = load_review_system(kind)['schema']
            for key in ('keep', 'expand', 'size_check'):
                self.assertNotIn(key, schema['properties'])
            self.assertNotIn('allOf', schema)
            # 证据字段排在结论之前：生成顺序即判断顺序
            for name in ('degradation_check' if kind == 'semantic' else 'height_check',):
                props = list(schema['properties'][name]['properties'])
                self.assertEqual(props[0], 'evidence')
            self.assertEqual(list(schema['properties'])[2], 'observations')

    def test_prompt_system_hides_upstream_conclusion(self):
        for kind in ('semantic', 'height'):
            system = load_review_system(kind)
            text = system['prompt']+system['skill']+system['rule']+json.dumps(system['schema'], ensure_ascii=False)
            for word in ('初筛', '退化块', '复核', 'trigger', 'rule_degraded', 'confirmed',
                         'size_options_m', 'recommended_size_m', 'size_check', '当前块', '块中心', '台阶', '连通'):
                self.assertNotIn(word, text)
            self.assertIn('待评估的道路区域', system['prompt'])

    def test_schema_validation_and_block_binding(self):
        from jsonschema import Draft202012Validator
        from agent.vlm.block_review import bind_block
        for kind in ('semantic', 'height'):
            schema = bind_block(load_review_system(kind)['schema'], 'B')
            validator = Draft202012Validator(schema)
            validator.validate(current_output(output_for(kind, 'B')))
            validator.validate(current_output(output_for(kind, 'B', is_road=False)))
            validator.validate({**current_output(output_for(kind, 'B')), 'observations': {'rgb': 'x'*500, kind: 'x', 'context': 'x'}})  # 文本不设长度上限
            bad = [output_for(kind, 'C'),                                          # block_id 必须原样返回
                   {**output_for(kind, 'B'), 'keep': True},                        # 不允许派生字段
                   {**output_for(kind, 'B'), 'review_type': '其它'},
                   {**output_for(kind, 'B'), 'observations': {'rgb': '', kind: 'x', 'context': 'x'}}]                   # 文本不能为空
            for value in bad:
                with self.assertRaises(ValidationError):
                    validator.validate(value)

    def test_loader_rejects_conditional_schema(self):
        import shutil, tempfile
        from pathlib import Path
        from agent.vlm.block_review import ROOT
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(ROOT/'semantic', Path(tmp)/'semantic')
            path = Path(tmp)/'semantic/schema.json'
            schema = json.loads(path.read_text())
            schema['allOf'] = [{'if': {}, 'then': {}}]
            path.write_text(json.dumps(schema))
            with self.assertRaises(ValueError):
                load_review_system('semantic', root=tmp)

    def test_chinese_enum_contract(self):
        from copy import deepcopy
        from jsonschema import Draft202012Validator
        from agent.vlm.block_review import bind_block
        from agent.map_preprocess.nodes.vlm_review import check_output
        for kind in ('semantic', 'height'):
            schema = bind_block(load_review_system(kind)['schema'], 'B')
            validator = Draft202012Validator(schema)
            output = output_for(kind, 'B')
            output = current_output(output)
            check_output({'block_id': 'B', 'review_type': kind}, output, schema)
            bad = deepcopy(output)
            bad['drivable_check'] = {'area_type': '道路'}
            with self.assertRaises(ValidationError):
                validator.validate(bad)
            bad = deepcopy(output)
            bad['schema_version'] = 2
            with self.assertRaises(ValidationError):
                validator.validate(bad)

    def test_semantic_verdict_is_required_and_controls_degradation(self):
        from jsonschema import Draft202012Validator
        system = load_review_system('semantic')
        output = current_output(output_for('semantic', 'B'))
        check = output['degradation_check']
        check['verdict'] = {'result': '不退化', 'reason': '逐段对照后判定', 'evidence_refs': ['R1']}
        validator = Draft202012Validator(system['schema'])
        validator.validate(output)
        self.assertIs(derive(output, system)['degraded'], False)
        del check['road_comparisons']
        with self.assertRaises(ValidationError):
            validator.validate(output)

    def test_height_questions_are_required_and_occlusion_is_evidence_only(self):
        from jsonschema import Draft202012Validator
        system = load_review_system('height')
        output = current_output(output_for('height', 'B', degraded=False))
        comparisons = output['height_check']['height_comparisons']
        comparisons['overhead_occlusion'] = '北侧树冠覆盖路带，DSM 显示冠层'
        validator = Draft202012Validator(system['schema'])
        validator.validate(output)
        self.assertIs(derive(output, system)['degraded'], False)
        del comparisons['along_road']
        with self.assertRaises(ValidationError):
            validator.validate(output)

    def test_height_verdict_controls_degradation(self):
        output = current_output(output_for('height', 'B', degraded=True))
        output['height_check']['verdict'] = {'result': '不退化', 'reason': '两处沿路结构有明显变化', 'evidence_refs': ['H']}
        derived = derive(output)
        self.assertIs(derived['degraded'], False)
        self.assertEqual(derived['reason_code'], 'vlm_verdict')
        self.assertEqual(derived['decision_trace']['reason'], '两处沿路结构有明显变化')

    def test_invalid_evidence_and_unusable_data_discard_verdict(self):
        for kind, name in (('semantic', 'degradation_check'), ('height', 'height_check')):
            output = current_output(output_for(kind, 'B'))
            output[name]['verdict']['evidence_refs'] = ['A2']
            derived = derive(output)
            self.assertIsNone(derived['degraded'])
            self.assertEqual(derived['reason_code'], 'invalid_evidence_refs')
            output['data_check']['status'] = '不可用'
            self.assertEqual(derive(output)['reason_code'], 'insufficient_data')

    def test_verdict_contradicting_reason_is_discarded(self):
        for kind, name in (('semantic', 'degradation_check'), ('height', 'height_check')):
            output = current_output(output_for(kind, 'B'))
            output[name]['verdict']['result'] = '退化'
            output[name]['verdict']['reason'] = '综合判断该区域结构完整，匹配不退化。'
            derived = derive(output)
            self.assertIsNone(derived['degraded'])
            self.assertEqual(derived['reason_code'], 'verdict_reason_contradiction')

            output[name]['verdict']['result'] = '退化'
            output[name]['verdict']['reason'] = '满足单类占优退化条件，但纵向退化不成立。'
            self.assertIs(derive(output)['degraded'], True)

            output[name]['verdict']['result'] = '不退化'
            output[name]['verdict']['reason'] = '植被占优且路带被截断，判为退化。'
            derived = derive(output)
            self.assertIsNone(derived['degraded'])
            self.assertEqual(derived['reason_code'], 'verdict_reason_contradiction')

    def test_verdict_reason_precedes_result_and_allows_occluded_ref(self):
        for kind, name in (('semantic', 'degradation_check'), ('height', 'height_check')):
            verdict_props = list(load_review_system(kind)['schema']['properties'][name]['properties']['verdict']['properties'])
            self.assertLess(verdict_props.index('reason'), verdict_props.index('result'))
        output = current_output(output_for('semantic', 'B'))
        check = output['degradation_check']
        check['road_comparisons'].append({**check['road_comparisons'][0], 'position': '中部树冠段', 'rgb_basis': '无法确认', 'semantic_road': '缺失'})
        check['verdict']['evidence_refs'] = ['R1', 'R2']
        self.assertIs(derive(output)['degraded'], True)

    def test_longitudinal_evidence_requires_distinct_positions(self):
        old = output_for('height', 'B', degraded=False)
        old['height_check']['longitudinal_structure'] = '重复'
        output = current_output(old)
        self.assertEqual(derive(output)['degradation_type'], '纵向退化')
        comparisons = output['height_check']['longitudinal_comparisons']
        comparisons[1]['position'] = comparisons[0]['position']
        self.assertEqual(derive(output)['reason_code'], 'insufficient_longitudinal_evidence')

    def test_derive(self):
        for kind in ('semantic', 'height'):
            self.assertTrue(derive(output_for(kind, 'B'))['degraded'])
            self.assertTrue(derive(output_for(kind, 'B'))['keep'])
            self.assertFalse(derive(output_for(kind, 'B', degraded=False))['degraded'])

    def test_stats_are_neutral(self):
        plant, building = bf.CLASSES.index('plant'), bf.CLASSES.index('building')
        labels = np.full((100, 100), plant, dtype=np.uint8)
        labels[:, :10] = building
        self.assertEqual(vi.semantic_stats(labels), {'road_pct': 0., 'building_pct': 10., 'plant_pct': 90., 'other_pct': 0.})

    def test_decide_uses_program_result(self):
        output = output_for('semantic', 'B', degraded=False)
        record = {'block_id': 'B', 'review_type': 'semantic', 'status': 'ok', 'output': output, 'derived': derive(output)}
        decision = decide({'block_id': 'B', 'size_m': 64, 'valid_global_bbox': [0, 0, 10, 10]}, [record])
        self.assertFalse(decision['degraded_scene'])
        self.assertEqual(decision['degradation_confirmed'], {'semantic': False})
        self.assertEqual((decision['recommended_size_m'], decision['expand'], decision['review_global_bbox']),
                         (64, False, [0, 0, 10, 10]))

    def test_routing(self):
        block = lambda s, h: {'review_types_available': ['semantic', 'height'],
                              'features': {'degradation': {'semantic_degraded': s, 'height_degraded': h}}}
        self.assertEqual(review_types(block(True, None)), ['semantic'])
        self.assertEqual(review_types(block(None, True)), ['height'])
        self.assertEqual(review_types(block(True, True)), ['semantic', 'height'])
        self.assertEqual(review_types(block(False, False)), [])

    def test_semantic_priority_and_unknown(self):
        output = output_for('semantic', 'B', degraded=False)
        detail = output['degradation_check']
        # 类别分布不过度失衡、道路边界清楚、无侵入 → 不退化；不因直路而改判。
        self.assertFalse(derive(output)['degraded'])
        detail['dominance'] = '明显占优'
        self.assertFalse(derive(output)['degraded'])  # 占比大本身不充分
        detail['road_boundary'] = '模糊'
        self.assertTrue(derive(output)['degraded'])
        detail['dominance'] = '无单类明显占优'
        self.assertIsNone(derive(output)['degraded'])  # 证据不落入确定判据，不猜测
        detail['road_intrusion'] = '存在'
        self.assertTrue(derive(output)['degraded'])   # 侵入不以高占比为前提
        output['data_check']['status'] = '不可用'
        self.assertIsNone(derive(output)['degraded']) # 质量检查优先

    def test_height_quality_and_distinctiveness(self):
        output = output_for('height', 'B', degraded=False)
        self.assertFalse(derive(output)['degraded'])
        detail = output['height_check']
        detail['height_distinctiveness'] = '低'
        self.assertTrue(derive(output)['degraded'])  # 高出物清楚但只占一小点，周围结构不足
        detail['raised_structure'] = '不确定'
        self.assertIsNone(derive(output)['degraded'])
        output = output_for('height', 'B')
        for status in ('不可用', '不确定'):
            output['data_check']['status'] = status
            self.assertIsNone(derive(output)['degraded'])

    def test_longitudinal_degradation_priority_and_exclusion(self):
        for kind in ('semantic', 'height'):
            key = 'degradation_check' if kind == 'semantic' else 'height_check'
            output = output_for(kind, 'B', degraded=False)
            straight = '明显直线' if kind == 'semantic' else '近似直线'
            detail = output[key]
            detail.update(road_geometry=straight, longitudinal_structure='重复')
            result = derive(output)
            self.assertTrue(result['degraded'])
            self.assertTrue(result['keep'])
            self.assertEqual(result['degradation_type'], '纵向退化')
            self.assertEqual(result['reason_code'], 'longitudinal_repetition')
            # 清楚的语义边界或充分黄红结构不能豁免纵向重复。
            for geometry, pattern, expected in (
                    (straight, '有变化', '不退化'),
                    ('非直线', '重复', '不退化'),
                    ('不适用', '不适用', '不退化'),
                    (straight, '不确定', '未知'),
                    ('不确定', '重复', '未知')):
                detail.update(road_geometry=geometry, longitudinal_structure=pattern)
                self.assertEqual(derive(output)['degradation_type'], expected)
            output['drivable_check']['area_type'] = '铺装广场'
            detail.update(road_geometry='不适用', longitudinal_structure='不适用')
            self.assertEqual(derive(output)['degradation_type'], '不退化')
            output['drivable_check']['area_type'] = '道路'
            detail.update(road_geometry=straight, longitudinal_structure='重复')
            if kind == 'semantic':
                detail['road_intrusion'] = '存在'
            else:
                detail['height_distinctiveness'] = '低'
            self.assertEqual(derive(output)['degradation_type'], '退化')
            if kind == 'semantic':
                detail.update(road_intrusion='无', dominance='明显占优', road_boundary='模糊')
                self.assertEqual(derive(output)['reason_code'], 'dominant_class_obscures_road')
                self.assertEqual(derive(output)['degradation_type'], '退化')
            output['data_check']['status'] = '不可用'
            self.assertEqual(derive(output)['degradation_type'], '未知')

    def test_longitudinal_fields_are_required_and_conclusion_is_derived(self):
        from copy import deepcopy
        from jsonschema import Draft202012Validator
        from agent.vlm.block_review import bind_block
        for kind in ('semantic', 'height'):
            validator = Draft202012Validator(bind_block(load_review_system(kind)['schema'], 'B'))
            output = output_for(kind, 'B')
            key = 'degradation_check' if kind == 'semantic' else 'height_check'
            for field in ('road_geometry', 'longitudinal_structure'):
                bad = deepcopy(output)
                del bad[key][field]
                with self.assertRaises(ValidationError):
                    validator.validate(bad)
            for legacy in (2, 3, 4, 5, 6, 7):
                bad = deepcopy(output)
                bad['schema_version'] = legacy
                with self.assertRaises(ValidationError):
                    validator.validate(bad)
            output[key]['degradation_type'] = '纵向退化'
            with self.assertRaises(ValidationError):
                validator.validate(output)

    def test_loader_rejects_inconsistent_degradation_type(self):
        import shutil, tempfile
        from pathlib import Path
        from agent.vlm.block_review import ROOT
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(ROOT/'height', Path(tmp)/'height')
            path = Path(tmp)/'height/rule.yaml'
            rules = yaml.safe_load(path.read_text())
            rules['degradation_decision'][-1]['degradation_type'] = '不退化'
            path.write_text(yaml.safe_dump(rules, allow_unicode=True))
            with self.assertRaisesRegex(ValueError, 'degradation_type'):
                load_review_system('height', root=tmp)

    def test_confidence_is_independent(self):
        output = output_for('semantic', 'B')
        output['data_check']['data_confidence'] = '低'
        result = derive(output)
        self.assertIsNone(result['degraded'])
        self.assertEqual(result['reason_code'], 'uncertain_data_quality')

    def test_low_confidence_only_replaces_its_own_field(self):
        output = output_for('height', 'B')
        output['height_check']['longitudinal_structure_confidence'] = '低'
        self.assertEqual(derive(output)['reason_code'], 'low_height_distinctiveness')  # 其余字段照常判定
        output['height_check']['raised_structure_confidence'] = '低'
        output['height_check']['height_distinctiveness'] = '不确定'
        self.assertEqual(derive(output)['reason_code'], 'unclear_height_structure')
        output = output_for('semantic', 'B', degraded=False)
        output['degradation_check'].update(road_geometry='明显直线', longitudinal_structure='重复',
                                           longitudinal_structure_confidence='低')
        self.assertEqual(derive(output)['degradation_type'], '未知')  # 重复不可信，也不能判不退化
        # 低置信度的 不可通行 不走 not_drivable，结构判断继续
        output = output_for('semantic', 'B', is_road=False)
        output['drivable_check']['area_type_confidence'] = '低'
        result = derive(output)
        self.assertEqual(result['reason_code'], 'road_intrusion')

    def test_confidence_precedes_consistency_and_preserves_independent_evidence(self):
        for kind in ('semantic', 'height'):
            key = 'degradation_check' if kind == 'semantic' else 'height_check'
            output = output_for(kind, 'B')
            output[key].update(road_geometry='不适用', longitudinal_structure='重复')
            result = derive(output)
            self.assertEqual(result['consistency_check'], '不通过')
            self.assertTrue(result['degraded'])  # 无关几何矛盾不阻断侵入/高度不足
            self.assertTrue(result['consistency_failures'])
            output[key]['road_geometry_confidence'] = '低'
            result = derive(output)
            self.assertEqual(result['consistency_check'], '通过')
            self.assertEqual(result['consistency_failures'], [])
            self.assertTrue(result['degraded'])
            self.assertEqual(output[key]['road_geometry'], '不适用')  # 原始观察保留

    def test_consistency_masks_only_related_fields(self):
        for kind in ('semantic', 'height'):
            key = 'degradation_check' if kind == 'semantic' else 'height_check'
            output = output_for(kind, 'B', degraded=False)
            output[key].update(road_geometry='不适用', longitudinal_structure='不适用')
            result = derive(output)
            self.assertEqual(result['consistency_check'], '通过')
            self.assertFalse(result['degraded'])
        output = output_for('height', 'B')
        output['height_check']['height_distinctiveness'] = '高'
        result = derive(output)
        self.assertEqual(result['reason_code'], 'inconsistent_fields')
        self.assertEqual(result['matched_reasons'], [])
        self.assertIn('low_height_distinctiveness', result['candidate_reasons'])
        output['height_check']['longitudinal_structure'] = '重复'
        result = derive(output)
        self.assertEqual(result['degradation_type'], '纵向退化')
        self.assertEqual(result['matched_reasons'], ['longitudinal_repetition'])
        self.assertEqual(result['consistency_check'], '不通过')

    def test_missing_road_boundary_does_not_imply_intrusion(self):
        output = output_for('semantic', 'B', degraded=False)
        output['degradation_check'].update(dominant_class='道路', dominance='明显占优',
                                           road_boundary='消失', road_intrusion='无')
        result = derive(output)
        self.assertEqual(result['consistency_check'], '通过')
        self.assertTrue(result['degraded'])
        self.assertEqual(result['reason_code'], 'dominant_class_obscures_road')
        self.assertNotIn('road_intrusion', result['candidate_reasons'])
        self.assertNotIn('road_intrusion', result['matched_reasons'])

    def test_consistency_loader_rejects_unknown_or_unmaskable_fields(self):
        import shutil, tempfile
        from pathlib import Path
        from agent.vlm.block_review import ROOT
        for change in ('condition', 'invalidate'):
            with tempfile.TemporaryDirectory() as tmp:
                shutil.copytree(ROOT/'semantic', Path(tmp)/'semantic')
                path = Path(tmp)/'semantic/rule.yaml'
                rules = yaml.safe_load(path.read_text())
                constraint = rules['consistency_checks']['rules'][0]
                if change == 'condition':
                    constraint['when']['road_geometery'] = ['明显直线']
                else:
                    constraint['invalidate'] = ['data_confidence']
                path.write_text(yaml.safe_dump(rules, allow_unicode=True))
                with self.assertRaises(ValueError):
                    load_review_system('semantic', root=tmp)

    def test_old_drivability_fields_do_not_gate_degradation(self):
        for kind in ('semantic', 'height'):
            old_output = output_for(kind, 'B', is_road=False)
            result = derive(old_output)
            self.assertTrue(result['keep'])
            record = {'review_type': kind, 'status': 'ok', 'output': old_output, 'derived': result}
            decision = decide({'size_m': 75, 'valid_global_bbox': [0, 0, 75, 75]}, [record])
            self.assertEqual(decision['scene_types'], [kind])
            self.assertNotIn('is_drivable', decision)

    def test_semantic_prompts_demand_judgement_under_tree_shade(self):
        system = load_review_system('semantic')
        for text in (system['prompt'], system['skill'], system['rule']):
            self.assertIn('树荫', text)
        self.assertNotIn('not_drivable', system['rule'])
        self.assertNotIn('drivable_check', system['schema']['properties'])

    def test_matched_reasons_exclude_unconfirmed_candidates(self):
        output = output_for('semantic', 'B')
        output['degradation_check'].update(road_geometry='明显直线', longitudinal_structure='重复')
        result = derive(output)
        self.assertEqual((result['reason_code'], result['degradation_type']), ('road_intrusion', '退化'))
        self.assertEqual(result['matched_reasons'],
                         ['road_intrusion', 'dominant_class_obscures_road', 'longitudinal_repetition'])
        output['data_check']['status'] = '不可用'
        result = derive(output)
        self.assertIsNone(result['degraded'])
        self.assertEqual(result['matched_reasons'], [])
        self.assertIn('road_intrusion', result['candidate_reasons'])
        output['data_check'].update(status='可用', data_confidence='低')
        self.assertEqual(derive(output)['matched_reasons'], [])
        output['data_check']['data_confidence'] = '高'
        output['drivable_check']['area_type'] = '不可通行'
        result = derive(output)  # 退化原因不看地面类型，不可通行只决定块不保留
        self.assertEqual(result['matched_reasons'],
                         ['road_intrusion', 'dominant_class_obscures_road', 'longitudinal_repetition'])
        self.assertTrue(result['keep'])
        self.assertEqual(derive(output_for('height', 'B', degraded=False))['matched_reasons'], [])

    def test_height_weak_structure_cannot_be_normal(self):
        output = output_for('height', 'B', degraded=False)
        output['height_check']['raised_structure'] = '较弱'
        self.assertEqual(derive(output)['reason_code'], 'unclear_height_structure')
        output['height_check']['height_distinctiveness'] = '低'
        self.assertEqual(derive(output)['reason_code'], 'low_height_distinctiveness')

    def test_loader_requires_field_confidence(self):
        import shutil, tempfile
        from pathlib import Path
        from agent.vlm.block_review import ROOT
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(ROOT/'height', Path(tmp)/'height')
            path = Path(tmp)/'height/schema.json'
            schema = json.loads(path.read_text(encoding='utf-8'))
            check = schema['properties']['height_check']
            del check['properties']['road_geometry_confidence']
            check['required'].remove('road_geometry_confidence')
            path.write_text(json.dumps(schema, ensure_ascii=False), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'road_geometry_confidence'):
                load_review_system('height', root=tmp)

    def test_unknown_and_conflict_are_not_normal_or_deleted(self):
        block = {'size_m': 64, 'valid_global_bbox': [0, 0, 64, 64]}
        def record(output):
            return {'review_type': output['review_type'], 'status': 'ok', 'output': output, 'derived': derive(output)}
        unknown = output_for('height', 'B')
        unknown['data_check']['status'] = '不可用'
        result = decide(block, [record(unknown)])
        self.assertIsNone(result['degraded_scene'])
        self.assertEqual(result['status'], 'uncertain')
        # 已知不退化 + 另一种未知/失败，不能作为整体不退化。
        normal = record(output_for('semantic', 'B', degraded=False))
        failure = {'review_type': 'height', 'status': 'error', 'error': 'offline'}
        self.assertIsNone(decide(block, [normal, failure])['degraded_scene'])
        self.assertIsNone(decide(block, [normal, record(unknown)])['degraded_scene'])
        # 一种确认退化，另一种未知：已确认的场景仍可保留，同时记录未知。
        result = decide(block, [record(output_for('semantic', 'B')), record(unknown)])
        self.assertEqual(result['scene_types'], ['semantic'])
        self.assertEqual(result['unresolved_review_types'], ['height'])

    def test_output_cannot_duplicate_conclusions(self):
        from agent.vlm.block_review import bind_block
        from agent.map_preprocess.nodes.vlm_review import check_output
        for kind in ('semantic', 'height'):
            schema = bind_block(load_review_system(kind)['schema'], 'B')
            output = output_for(kind, 'B')
            key = 'degradation_check' if kind == 'semantic' else 'height_check'
            output[key]['degraded'] = False
            with self.assertRaises(ValidationError):
                check_output({'block_id': 'B', 'review_type': kind}, output, schema)

    def test_prompt_boundaries_and_rule_mapping(self):
        for kind in ('semantic', 'height'):
            system = load_review_system(kind)
            for name in ('data_check', 'degradation_check' if kind == 'semantic' else 'height_check'):
                self.assertNotIn('confidence', system['schema']['properties'][name]['required'])
                self.assertIn('uncertainty', system['schema']['properties'][name]['required'])
            self.assertEqual(system['rules']['degradation_decision'][-1]['when'], {})
        height = load_review_system('height')
        text = height['prompt'] + height['skill'] + json.dumps(height['schema'], ensure_ascii=False)
        for old in ('去掉坡度', '二次曲面', 'height_grid_m', 'sensor_limit', 'base_on_road', '雷达能感知到'):
            self.assertNotIn(old, text)

    def test_model_context_omits_metadata_statistics_and_schema_dump(self):
        from agent.map_preprocess.nodes.vlm_review import model_input, user_prompt
        for kind in ('semantic', 'height'):
            payload = {'block_id': 'B', 'review_type': kind, 'image_extent_m': [20., 21.],
                       'image_size_px': [80, 84], 'image_global_bbox': [100, 200, 120, 221],
                       'm_per_image_px': [.25, .25], 'center_image_xy': [16., 20.],
                       'road_direction_deg': 123.45, 'road_source': 'LOCAL_SOURCE_SENTINEL',
                       'class_pct': {'road_pct': 77.77}, 'semantic_nodata_pct': 22.22,
                       'height_stats': {'nodata_pct': 22.22, 'above_range_pct': 33.33},
                       'dsm_reference': {'h_ref_m': 12.345, 'percentile': 20, 'context_valid_pct': 44.44},
                       'dsm_color_range_m': [-5., 15.], 'dsm_color_legend': vi.dsm_legend((-5., 15.)),
                       'context': {'context_extent_m': [40., 42.], 'context_global_bbox': [90, 190, 130, 232]}}
            system = load_review_system(kind)
            compact = model_input(payload)
            self.assertEqual(set(compact), {'block_id', 'review_type', 'image_description'})
            prompt = user_prompt({'block_id': 'B', 'review_type': kind, 'payload': payload}, system)
            self.assertIn('宽 20 米、高 21 米', prompt)
            self.assertIn('第 1 张', prompt)
            self.assertIn('第 2 张', prompt)
            self.assertIn('第 3 张', prompt)   # 两类复核都是三图，局部图无标记，上下文图有目标框
            self.assertIn('宽 40 米、高 42 米', prompt)
            self.assertIn('横向约 20%、纵向约 24%', prompt)
            self.assertIn('程序添加的黄框是范围标记', prompt)
            self.assertNotIn('十字', compact['image_description'])
            self.assertNotIn('红线', compact['image_description'])
            text = system['prompt']+prompt
            for forbidden in ('class_pct', 'color_pct', 'semantic_nodata_pct', 'height_stats', 'dsm_reference', 'dsm_color_legend',
                              'nodata_pct', '33.33', '44.44', 'height_color_legend', 'image_size_px', 'image_global_bbox',
                              'm_per_image_px', 'center_image_xy', 'road_source', 'road_direction_deg',
                              'LOCAL_SOURCE_SENTINEL', '77.77', '22.22', '块信息 JSON', '"properties"', '"$schema"'):
                self.assertNotIn(forbidden, text)
            for value in ('12.35', '12.345', '百分位', '参考高程'):   # 只有图 + 介绍，不发送参考高程等数据
                self.assertNotIn(value, compact['image_description'])
            if kind == 'height':   # 高程着色只说明颜色代表的高低
                for text in ('Viridis 连续伪彩色色带', '深紫色=较低高程', '深蓝色=低至中等高程', '蓝绿色/青绿色=中等高程',
                             '绿色=较高高程', '亮黄色=最高高程区域', '深紫色 → 蓝色 → 蓝绿色 → 绿色 → 亮黄色', '白色=无数据'):
                    self.assertIn(text, compact['image_description'])
                self.assertNotIn('-5', compact['image_description'])
                self.assertIn('由 DSM 表面高程减去周边低处的着色基准', prompt)
            else:   # 透明度写明：语义颜色占 30%、原始影像占 70%
                self.assertIn('绿色=植被', compact['image_description'])
                self.assertIn('从整幅卫星图的自动语义分割结果裁出', prompt)
                self.assertIn('不表示分割置信度', prompt)
                self.assertIn('语义颜色占 30%、原始影像占 70%', compact['image_description'])

    def test_boundary_inputs_crop_outside_map_and_preserve_real_nodata(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from PIL import Image
        from agent.map_preprocess.global_config import GlobalPreprocessConfig
        from agent.map_preprocess.nodes.vlm_review import prepare_block
        config = GlobalPreprocessConfig(block_size_m=32., vlm_review_image_px=128)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'rgb.png'
            Image.new('RGB', (64, 64), (17, 29, 43)).save(source)
            manifest = {'resolution_m_per_px': 1., 'config': {'pillow_max_image_pixels': 100000},
                        'image': {'width': 64, 'height': 64, 'path': str(source), 'dsm': {'path': 'mock'}},
                        'edges': [{'edge_id': 'E', 'polyline_xy': [[0., 5.], [20., 5.]]}]}
            block = {'block_id': 'B', 'edge_id': 'E', 'global_xy': [4., 5.], 'tangent_xy': [1., 0.],
                     'features': {'height_stats': {'base_dsm_m': 0.}}}
            for missing in (False, True):
                with self.subTest(missing=missing):
                    labels = np.zeros((64, 64), np.uint8)
                    dsm = np.zeros((64, 64), np.float32)
                    if missing:
                        labels[2, 2], dsm[2, 2] = bf.IGNORE, np.nan
                    def read_dsm(_info, bbox):
                        x0, y0, x1, y1 = bbox
                        return dsm[y0:y1, x0:x1]
                    with patch('agent.map_preprocess.utils.dsm.read_block_dsm', side_effect=read_dsm):
                        jobs = prepare_block(manifest, block, ['semantic', 'height'], labels, config, Path(tmp)/'inputs')
                    for job in jobs:
                        payload = job['payload']
                        directory = resolve_path(job['images'][0]).parent
                        compact = json.loads((directory/f"{job['review_type']}_input.json").read_text())
                        self.assertEqual(set(compact), {'block_id', 'review_type', 'image_description'})
                        self.assertIn('宽 20 米、高 21 米', compact['image_description'])
                        self.assertEqual(json.loads((directory/f"{job['review_type']}_metadata.json").read_text()), payload)
                        self.assertEqual(payload['image_global_bbox'], [0, 0, 20, 21])
                        self.assertEqual(payload['image_size_px'], [80, 84])
                        self.assertEqual(payload['image_extent_m'], [20., 21.])
                        self.assertEqual(payload['m_per_image_px'], [.25, .25])
                        self.assertEqual(payload['center_image_xy'], [16., 20.])
                        local = job['images'][:2] if job['review_type'] == 'semantic' else job['images'][1:]
                        for path in local:   # 局部图与地图内范围一致
                            with Image.open(resolve_path(path)) as image:
                                self.assertEqual(image.size, (80, 84))
                        if job['review_type'] == 'height':
                            self.assertEqual([resolve_path(x).name for x in job['images']],
                                             ['context_rgb.png', 'local_rgb.png', 'local_dsm.png'])
                            with Image.open(resolve_path(job['images'][0])) as image:   # 100 m 上下文裁去地图外
                                self.assertEqual(list(image.size), payload['context']['context_size_px'])
                                self.assertEqual(payload['context']['context_global_bbox'], [0, 0, 54, 55])
                            self.assertEqual(payload['dsm_reference']['h_ref_m'], 0.)
                            with Image.open(resolve_path(job['images'][2])) as image:   # DSM − H_ref = 0 m → 色标 0 m 颜色，无十字、无中心线
                                zero = tuple(vi.dsm_color(np.zeros((1, 1), np.float32), tuple(config.vlm_review_dsm_range_m))[0, 0].tolist())
                                self.assertEqual(image.getpixel((16, 20)), zero)
                                self.assertEqual(image.getpixel((32, 20)), zero)
                                self.assertEqual(image.getpixel((9, 9)), vi.DSM_NODATA_RGB if missing else zero)
                        if job['review_type'] == 'semantic':
                            self.assertEqual([resolve_path(x).name for x in job['images']],
                                             ['local_rgb.png', 'local_semantic.png', 'context_semantic.png'])
                            with Image.open(resolve_path(job['images'][0])) as image:
                                self.assertNotEqual(image.getpixel((16, 20)), (255, 255, 255))   # 局部 RGB 不叠加十字
                            with Image.open(resolve_path(job['images'][2])) as image:  # 100 m 上下文裁去地图外
                                self.assertEqual(list(image.size), payload['context']['context_size_px'])
                                self.assertEqual(payload['context']['context_global_bbox'], [0, 0, 54, 55])
                            self.assertIn('宽 54 米、高 55 米', compact['image_description'])
                    with Image.open(resolve_path(jobs[0]['images'][0])) as rgb:
                        for xy in ((0, 0), (79, 0), (0, 83), (79, 83)):
                            self.assertEqual(rgb.getpixel(xy), (17, 29, 43))  # 无地图外黑边
                    with Image.open(resolve_path(jobs[0]['images'][1])) as image:
                        self.assertEqual(image.getpixel((32, 20)), tuple(vi.semantic_color(np.zeros((1, 1), np.uint8))[0, 0].tolist()))  # 道路线位置仍是类别色，不画线
                    expected_missing = .2 if missing else 0.
                    self.assertEqual(jobs[0]['payload']['semantic_nodata_pct'], expected_missing)
                    self.assertEqual(jobs[1]['payload']['height_stats']['nodata_pct'], expected_missing)

    def test_region_outside_map_does_not_generate_black_input(self):
        import tempfile
        from pathlib import Path
        from agent.map_preprocess.global_config import GlobalPreprocessConfig
        from agent.map_preprocess.nodes.vlm_review import prepare_block
        block = {'global_xy': [200., 200.]}
        manifest = {'resolution_m_per_px': 1., 'image': {'width': 64, 'height': 64}}
        with tempfile.TemporaryDirectory() as tmp:
            jobs = prepare_block(manifest, block, ['semantic'], np.zeros((64, 64), np.uint8),
                                 GlobalPreprocessConfig(block_size_m=32., vlm_review_image_px=128), Path(tmp))
            self.assertEqual(jobs, [])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_dsm_color_uses_fixed_range_and_records_saturation(self):
        relative = np.array([[-10., -5., 0., 15., 30., np.nan]], np.float32)
        image = vi.dsm_color(relative, (-5., 15.))
        legend = vi.dsm_legend((-5., 15.))
        self.assertEqual([x['value_m'] for x in legend], [-5., 0., 5., 10., 15.])
        self.assertEqual(image[0, 0].tolist(), legend[0]['rgb'])    # 低于范围按下端点
        self.assertEqual(image[0, 1].tolist(), legend[0]['rgb'])
        self.assertEqual(image[0, 2].tolist(), legend[1]['rgb'])    # 0 m 在任何块都是同一颜色
        self.assertEqual(image[0, 4].tolist(), legend[-1]['rgb'])   # 高于范围按上端点
        self.assertEqual(tuple(image[0, 5]), vi.DSM_NODATA_RGB)
        stats = vi.dsm_stats(relative, (-5., 15.))
        self.assertEqual((stats['below_range_pct'], stats['above_range_pct'], stats['nodata_pct']), (16.7, 16.7, 16.7))
        self.assertEqual(vi.dsm_reference(np.array([[1., 2., 3., 4., 5., np.nan]]), 20)[0], 1.8)
        self.assertEqual(vi.dsm_reference(np.full((2, 2), np.nan), 20), (None, 0.))

    def test_subpixel_map_overlap_keeps_at_least_one_pixel(self):
        self.assertEqual(vi.valid_crop([-500, -500, 500, 500], (1, 1), 128),
                         ([0, 0, 1, 1], [64, 64, 65, 65]))

    def test_missing_dsm_prepares_unknown_input_instead_of_aborting(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from agent.map_preprocess.global_config import GlobalPreprocessConfig
        from agent.map_preprocess.nodes.vlm_review import prepare_block, check_output
        from agent.vlm.block_review import bind_block
        config = GlobalPreprocessConfig(vlm_review_image_px=128)
        manifest = {'resolution_m_per_px': 1., 'image': {'width': 64, 'height': 64, 'dsm': {}}, 'edges': []}
        manifest['image']['dsm'] = {'path': 'mock-dsm'}
        block = {'block_id': 'B', 'edge_id': 'E', 'global_xy': [32., 32.], 'tangent_xy': [1., 0.], 'features': {}}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(vi, 'read_rgb', return_value=np.zeros((128, 128, 3), np.uint8)), \
                patch.object(vi, 'crop_labels', side_effect=AssertionError('height must not crop semantic labels')), \
                patch.object(vi, 'read_dsm', return_value=np.full((128, 128), np.nan, np.float32)):
            job, = prepare_block(manifest, block, ['height'], None, config, Path(tmp))
            self.assertEqual(job['payload']['height_stats']['nodata_pct'], 100.)
            self.assertIsNone(job['payload']['dsm_reference']['h_ref_m'])
            output = output_for('height', 'B')
            schema = bind_block(load_review_system('height')['schema'], 'B')
            from agent.map_preprocess.nodes.vlm_review import has_no_topic_data
            self.assertTrue(has_no_topic_data(job))
            almost_missing = {**job, 'payload': {'height_stats': {'nodata_pct': 100., 'p50_m': 0.}}}
            self.assertFalse(has_no_topic_data(almost_missing))
            for status in ('可用', '不可用'):
                current = current_output(output)
                current['data_check']['status'] = status
                check_output(job, current, schema)
                derived = derive(current, evidence_problem='topic_no_data')
                self.assertIsNone(derived['degraded'])
                self.assertEqual(derived['reason_code'], 'topic_no_data')


class FakeVLM:
    """按 (block_id, review_type) 返回预设输出；记录每次请求的 system 与图片数。"""

    def __init__(self, answers: dict):
        self.answers, self.calls, self.lock = answers, [], threading.Lock()

    def infer(self, prompt, images=(), *, system, max_tokens=None, temperature=0., json_output=False, output_schema=None,
              image_labels=None, enable_thinking=False, thinking_token_budget=None):
        block_id = prompt.split('待评估道路区域 ')[1].split(' ')[0]
        kind = 'height' if f'{block_id} 的 height 匹配' in prompt else 'semantic'
        with self.lock:
            self.calls.append({'block_id': block_id, 'type': kind, 'system': system, 'images': len(images), 'prompt': prompt,
                               'schema': output_schema, 'image_labels': image_labels, 'image_paths': list(images),
                               'thinking': (enable_thinking, thinking_token_budget), 'max_tokens': max_tokens})
        answer = self.answers[(block_id, kind)]
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(text=json.dumps(current_output(answer), ensure_ascii=False), usage={}, latency_seconds=.1, reasoning='想过了')


class VLMReviewPipelineTests(node5.BlockFeatureTests):
    def mark(self, manifest_path, semantic: list[str], height: list[str]):
        manifest = load_manifest(manifest_path)
        for block in manifest['blocks']:
            d = block['features']['degradation']
            d['semantic_degraded'] = block['block_id'] in semantic
            d['height_degraded'] = block['block_id'] in height
        write_json(manifest_path, manifest)

    def setUp(self):
        super().setUp()
        self.review_config = replace(self.config, vlm_review_image_px=192, vlm_review_workers=2)

    def test_failed_rerun_restores_published_outputs(self):
        state, manifest = self.run_map()
        path = resolve_path(state['manifest_path'])
        block_id = manifest['blocks'][0]['block_id']
        self.mark(path, semantic=[block_id], height=[])
        client = FakeVLM({(block_id, 'semantic'): output_for('semantic', block_id)})
        kwargs = {'client': client, 'semantic_segmenter': node5.FakeSemantic(self.events)}
        summary = run_vlm_review(path, self.review_config, **kwargs)
        self.assertFalse(summary['errors'])
        paths = [path, path.with_name('state.json'), backup_path(path),
                 resolve_path(summary['path']), resolve_path(summary['degraded_scenes']), resolve_path(summary['overview']),
                 resolve_path(summary['results_dir'])/f'{block_id}_semantic.json',
                 resolve_path(summary['inputs_dir'])/block_id/'local_rgb.png']
        before = {file: file.read_bytes() for file in paths}
        with patch('agent.map_preprocess.nodes.vlm_review.prepare_block', side_effect=OSError('图像读取失败')):
            with self.assertRaisesRegex(OSError, '图像读取失败'):
                run_vlm_review(path, self.review_config, **kwargs)
        self.assertEqual({file: file.read_bytes() for file in paths}, before)

    def test_review_removes_expands_and_routes(self):
        state, manifest = self.run_map()
        path = state['manifest_path']
        ids = [b['block_id'] for b in manifest['blocks']]
        self.assertGreaterEqual(len(ids), 4)
        removed, expanded, both, rejected = ids[0], ids[1], ids[2], ids[3]
        self.mark(path, semantic=[removed, both], height=[expanded, both, rejected])
        base = self.review_config.block_size_m
        output = output_for
        answers = {(removed, 'semantic'): output('semantic', removed, is_road=False),
                   (expanded, 'height'): output('height', expanded, degraded=False),
                   (both, 'semantic'): output('semantic', both),
                   (both, 'height'): output('height', both),
                   (rejected, 'height'): output('height', rejected, degraded=False)}
        answers[(expanded, 'height')]['height_check']['longitudinal_structure'] = '重复'
        answers[(both, 'semantic')]['drivable_check'].update(area_type='不可通行', area_type_confidence='低')
        client = FakeVLM(answers)
        summary = run_vlm_review(path, self.review_config, client=client,
                                 semantic_segmenter=node5.FakeSemantic(self.events))
        # 分流：语义复核三图（局部两图同尺寸 + 上下文），高度复核三图（上下文 + 局部两图同尺寸）；各用自己的 system prompt；默认关闭 thinking
        from PIL import Image
        systems = {k: load_review_system(k)['prompt'] for k in ('semantic', 'height')}
        expected = {'semantic': ['local_rgb.png', 'local_semantic.png', 'context_semantic.png'], 'height': ['context_rgb.png', 'local_rgb.png', 'local_dsm.png']}
        for call in client.calls:
            self.assertEqual([p.name for p in call['image_paths']], expected[call['type']])
            self.assertEqual(call['images'], len(expected[call['type']]))
            sizes = set()
            for p in (call['image_paths'][:2] if call['type'] == 'semantic' else call['image_paths'][1:]):
                with Image.open(p) as image:
                    sizes.add(image.size)
            input_payload = json.loads((resolve_path(summary['inputs_dir'])/call['block_id']/f"{call['type']}_metadata.json").read_text())
            self.assertEqual(sizes, {tuple(input_payload['image_size_px'])})
            self.assertLessEqual(max(input_payload['image_size_px']), self.review_config.vlm_review_image_px)
            self.assertGreater(min(input_payload['image_size_px']), 0)
            self.assertEqual(call['image_labels'][0], '第 1 张：局部 RGB（local_rgb）' if call['type'] == 'semantic'
                             else '第 1 张：整体 RGB 上下文（context_rgb）')
            self.assertIn('局部语义分割' if call['type'] == 'semantic' else '局部 RGB', call['image_labels'][1])
            self.assertEqual(len(call['image_labels']), call['images'])
            self.assertEqual(call['system'], systems[call['type']])
            compact = json.loads((resolve_path(summary['inputs_dir'])/call['block_id']/f"{call['type']}_input.json").read_text())
            self.assertIn(compact['image_description'], call['prompt'])
            self.assertNotIn('image_global_bbox', call['prompt'])
            self.assertNotIn('class_pct', call['prompt'])
            self.assertNotIn('"properties"', call['prompt'])
            for word in ('初筛', 'trigger', 'rule_degraded', 'height_reasons', 'per_size_stats'):
                self.assertNotIn(word, call['prompt'])   # VLM 不知道 Node 5 的结论
            self.assertEqual(call['schema']['properties']['block_id'], {'type': 'string', 'const': call['block_id']})
            self.assertEqual(call['thinking'], (False, None))
            self.assertEqual(call['max_tokens'], self.review_config.vlm_review_max_tokens)
        self.assertEqual(summary['jobs_by_type'], {'semantic': 2, 'height': 3})
        # 退化场景清单：只有 VLM 确认退化成立的道路块
        self.assertEqual(sorted(summary['degraded_scene_ids']), sorted([removed, expanded, both]))
        self.assertEqual(summary['rejected_block_ids'], [rejected])
        scenes = json.loads(resolve_path(summary['degraded_scenes']).read_text())
        self.assertEqual(scenes['count'], 3)
        scene = {s['block_id']: s for s in scenes['scenes']}
        self.assertEqual(scene[both]['scene_types'], ['semantic', 'height'])
        self.assertEqual(scene[expanded]['scene_types'], ['height'])
        self.assertEqual(scene[expanded]['vlm']['height']['degradation_type'], '纵向退化')
        self.assertIn('height_comparisons', scene[expanded]['vlm']['height'])
        self.assertEqual(scene[expanded]['vlm']['height']['matched_reasons'], ['vlm_verdict'])
        self.assertEqual(scene[expanded]['vlm']['height']['candidate_reasons'], [])
        self.assertEqual(scene[expanded]['vlm']['height']['field_confidence']['road_geometry_confidence'], '高')
        self.assertEqual(summary['schema_version'], 12)
        result = load_manifest(path)
        kept = {b['block_id']: b for b in result['blocks']}
        self.assertIn(removed, kept)
        self.assertEqual(summary['removed_block_ids'], [])
        # 退化不成立：块留在路网中，但不是退化场景
        self.assertFalse(kept[rejected]['vlm_review']['degraded_scene'])
        self.assertFalse(kept[rejected]['vlm_review']['expand'])
        # 不判断范围扩展：退化场景范围就是当前块
        review = kept[expanded]['vlm_review']
        self.assertEqual((review['recommended_size_m'], review['expand']), (base, False))
        self.assertEqual(review['review_global_bbox'], [int(v) for v in kept[expanded]['valid_global_bbox']])
        self.assertNotIn('rule_max_size_m', review)
        self.assertNotIn('expanded_scenes', summary)
        # 两种退化都有：两套复核都跑
        self.assertEqual(set(kept[both]['vlm_review']['reviews']), {'semantic', 'height'})
        # 输入图与结果文件
        task = resolve_path(state['output_dir'])
        self.assertTrue((task/'visualization/10_vlm_review_inputs'/expanded/'local_dsm.png').is_file())
        self.assertFalse((task/'visualization/10_vlm_review_inputs'/expanded/'local_semantic.png').exists())  # 高度复核不用语义图
        record = json.loads((resolve_path(summary['results_dir'])/f'{expanded}_height.json').read_text())
        self.assertEqual(record['reasoning'], '想过了')
        self.assertEqual(record['derived']['keep'], True)
        self.assertFalse((task/'visualization/10_vlm_review_inputs'/removed/'local_dsm.png').exists())
        self.assertTrue((task/'visualization/09_vlm_review.png').is_file())
        self.assertTrue(backup_path(path).is_file())
        payload = json.loads((task/'visualization/10_vlm_review_inputs'/expanded/'height_metadata.json').read_text())
        bounds = payload['image_global_bbox']
        expected_extent = [(bounds[2]-bounds[0])*manifest['resolution_m_per_px'],
                           (bounds[3]-bounds[1])*manifest['resolution_m_per_px']]
        self.assertEqual(payload['image_extent_m'], expected_extent)
        for actual, m, n in zip(payload['m_per_image_px'], expected_extent, payload['image_size_px']):
            self.assertAlmostEqual(actual, m/n)
        self.assertEqual(set(payload['height_stats']), {'nodata_pct', 'below_range_pct', 'above_range_pct', 'p5_m', 'p50_m', 'p95_m'})
        self.assertEqual(payload['dsm_color_range_m'], [-2., 8.])
        self.assertEqual(payload['dsm_reference']['percentile'], 20.)
        self.assertNotIn('height_color_legend', payload)
        self.assertNotIn('height_grid_m', payload)
        self.assertFalse({'trigger', 'per_size_stats', 'semantic_per_size_stats', 'size_options_m', 'height_reference'} & set(payload))

        # 重跑从备份开始：被删的块重新参与复核；VLM 失败的块保留
        answers[(removed, 'semantic')] = RuntimeError('boom')
        summary = run_vlm_review(path, self.review_config, client=FakeVLM(answers),
                                 semantic_segmenter=node5.FakeSemantic(self.events))
        self.assertEqual(summary['removed_block_ids'], [])
        self.assertEqual(summary['errors'][0]['block_id'], removed)
        again = {b['block_id']: b for b in load_manifest(path)['blocks']}
        self.assertEqual(again[removed]['vlm_review']['status'], 'error')
        self.assertIsNone(again[removed]['vlm_review']['degraded_scene'])
        self.assertEqual(summary['unverified_block_ids'], [removed])
        # Node 5 重算会恢复复核前的 manifest
        from agent.map_preprocess.nodes.block_features import run_block_features
        run_block_features(path, self.review_config, semantic_segmenter=node5.FakeSemantic(self.events))
        self.assertFalse(backup_path(path).exists())
        self.assertTrue(all('vlm_review' not in b for b in load_manifest(path)['blocks']))

    def test_building_share_no_longer_overrides_structure(self):
        state, manifest = self.run_map()
        path = state['manifest_path']
        target = manifest['blocks'][2]['block_id']
        self.mark(path, semantic=[target], height=[])
        config = replace(self.review_config, vlm_review_semantic_min_building_pct=1.)
        summary = run_vlm_review(path, config, client=FakeVLM({(target, 'semantic'): output_for('semantic', target)}),
                                 semantic_segmenter=node5.FakeSemantic(self.events))
        self.assertEqual(summary['rejected_block_ids'], [])
        self.assertEqual(summary['degraded_scene_ids'], [target])
        record = json.loads((resolve_path(summary['results_dir'])/f'{target}_semantic.json').read_text())
        self.assertNotIn('building_enough', record)
        self.assertTrue(record['derived']['degraded'])
        payload = json.loads((resolve_path(summary['inputs_dir'])/target/'semantic_metadata.json').read_text())
        self.assertGreaterEqual(payload['class_pct']['building_pct'], 1.)
        self.assertNotIn('building_min_pct', payload)

    def test_unknown_results_persist_without_deleting_or_rejecting(self):
        state, manifest = self.run_map()
        ids = [b['block_id'] for b in manifest['blocks']]
        missing_height, uncertain_road = ids[:2]
        self.mark(state['manifest_path'], semantic=[uncertain_road], height=[missing_height])
        height = output_for('height', missing_height)
        height['data_check'].update(status='不可用', data_confidence='高', uncertainty='关键高度缺失')
        semantic = output_for('semantic', uncertain_road, is_road=False)
        semantic['drivable_check'].update(area_type_confidence='低', uncertainty='遮挡无法确认')
        semantic['data_check'].update(status='不确定', uncertainty='遮挡无法确认')  # 地面未知本身不阻止退化结论
        summary = run_vlm_review(state['manifest_path'], self.review_config,
                                 client=FakeVLM({(missing_height, 'height'): height,
                                                 (uncertain_road, 'semantic'): semantic}),
                                 semantic_segmenter=node5.FakeSemantic(self.events))
        self.assertEqual(summary['removed_block_ids'], [])
        self.assertEqual(summary['rejected_block_ids'], [])
        self.assertEqual(summary['degraded_scene_ids'], [])
        self.assertEqual(summary['unverified_block_ids'], sorted([missing_height, uncertain_road]))
        self.assertEqual(summary['discarded_insufficient_evidence_ids'], sorted([missing_height, uncertain_road]))
        blocks = {b['block_id']: b for b in load_manifest(state['manifest_path'])['blocks']}
        for key in (missing_height, uncertain_road):
            self.assertIsNone(blocks[key]['vlm_review']['degraded_scene'])

    def test_invalid_output_keeps_block_and_dry_run(self):
        state, manifest = self.run_map()
        path = state['manifest_path']
        target = manifest['blocks'][0]['block_id']
        self.mark(path, semantic=[target], height=[])
        dry = run_vlm_review(path, self.review_config, dry_run=True, semantic_segmenter=node5.FakeSemantic(self.events))
        self.assertEqual(dry['status'], 'dry_run')
        self.assertFalse(backup_path(path).exists())
        prompt = (resolve_path(dry['results_dir'])/f'{target}_semantic_prompt.txt').read_text()
        self.assertIn('[system]', prompt)
        self.assertIn('Rule：', prompt)
        self.assertIn('第 2 张：局部语义分割（local_semantic）\n<image', prompt)
        self.assertIn('第 3 张：整体语义上下文（context_semantic）\n<image', prompt)
        # 服务端未按约束输出（block_id 不一致）→ 视为失败，块保留
        bad = output_for('semantic', 'WRONG', is_road=False)
        summary = run_vlm_review(path, self.review_config, client=FakeVLM({(target, 'semantic'): bad}),
                                 semantic_segmenter=node5.FakeSemantic(self.events))
        self.assertEqual(summary['removed_block_ids'], [])
        self.assertEqual(len(summary['errors']), 1)

    def test_pipeline_skips_vlm_when_thresholds_find_no_candidates(self):
        # DSM 与语义有效，但未命中初筛规则。
        config = replace(self.review_config, vlm_review_enabled=True, feature_height_flat_min_pct=99.99,
                         feature_height_below_max_pct=99.99)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            from agent.map_preprocess import run_pipeline
            state = run_pipeline(str(self.image), config, dsm_path=str(self.dsm),
                                 semantic_segmenter=node5.FakeSemantic(self.events), vlm_client=FakeVLM({}))
        self.assertEqual(state['block_features']['degraded_blocks'], [])
        self.assertNotIn('vlm_review', state)
        self.assertEqual(state['final_summary']['conclusion'], 'no_degradation')

    def test_pipeline_ends_incomplete_without_review_data(self):
        import numpy as np
        from PIL import Image

        class NoDataSemantic(node5.FakeSemantic):
            cache_key = None

            def predict(self, image, valid_mask=None):
                return {'mask': np.full(image.shape[:2], 255, np.uint8),
                        'confidence': np.zeros(image.shape[:2], np.float32), 'metadata': {}}

        with Image.open(self.image) as image:
            dsm = node5.write_raster(self.root/'empty.tif', np.full(image.size[::-1], -9999., np.float32), nodata=-9999.)
        config = replace(self.review_config, vlm_review_enabled=True)
        client = FakeVLM({})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            from agent.map_preprocess import run_pipeline
            state = run_pipeline(str(self.image), config, dsm_path=str(dsm),
                                 semantic_segmenter=NoDataSemantic(self.events), vlm_client=client)
        self.assertNotIn('vlm_review', state)
        self.assertEqual(client.calls, [])
        self.assertEqual(state['final_summary']['conclusion'], 'incomplete')
        self.assertFalse(state['final_summary']['complete'])
        issue = state['final_summary']['issues'][0]
        self.assertEqual(issue['kind'], 'no_data')
        self.assertTrue(all(reasons == ['dsm_no_data'] for reasons in issue['reasons_by_block'].values()))
        self.assertEqual(state['final_summary']['early_exit']['stage'], 'node5')
        nodes = list(load_manifest(state['manifest_path'])['files']['node_states'])
        self.assertEqual(nodes[-2:], ['05_compute_block_features', '06_refresh_global_manifest'])


# 父类的 Node 5 用例不在本文件重复运行
for name in [n for n in dir(node5.BlockFeatureTests) if n.startswith('test_')]:
    if getattr(VLMReviewPipelineTests, name) is getattr(node5.BlockFeatureTests, name):
        setattr(VLMReviewPipelineTests, name, None)


if __name__ == '__main__':
    unittest.main()
