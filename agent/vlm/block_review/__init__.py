"""退化块 VLM 复核的两套独立 prompt 系统：semantic/（语义退化）与 height/（高度退化）。

每套目录各自包含 prompt.md（system）、skill.md（步骤）、rule.yaml（判定规则）、schema.json（输出结构），
互不引用。load_review_system 读取一套；bind_block 把 block_id 绑定为常量，生成时由结构化输出（llguidance）强制。

语义 v9 采用 VLM 的逐段对照及 verdict，高度 v8 和历史语义记录按有序规则计算。
置信度与一致性检查仍用于通行性及观察诊断。
不向生成 schema 添加条件规则；无法判断时派生结论为 None，不当作正常。
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

REVIEW_TYPES = ('semantic', 'height')
ROOT = Path(__file__).parent
CONDITIONAL_KEYS = ('allOf', 'anyOf', 'oneOf', 'if', 'then', 'else', 'not')
CONSISTENCY_VALUES = ('通过', '不通过')
# 这些条件不是 VLM 的判定字段，不受 preprocess 的低置信度替换
NON_JUDGED_KEYS = ('data_status', 'data_confidence', 'consistency_check')


def bind_block(schema: dict, block_id: str) -> dict:
    """每次请求：block_id 绑定为常量（生成时强制原样返回）。"""
    schema = copy.deepcopy(schema)
    schema['properties']['block_id'] = {'type': 'string', 'const': block_id}
    return schema


def _check_unconditional(value, where: str) -> None:
    if isinstance(value, dict):
        bad = [k for k in CONDITIONAL_KEYS if k in value]
        if bad:
            raise ValueError(f'{where} 含条件规则 {bad}：生成阶段不支持，可推导的结论请由程序计算')
        for key, item in value.items():
            _check_unconditional(item, f'{where}/{key}')
    elif isinstance(value, list):
        for k, item in enumerate(value):
            _check_unconditional(item, f'{where}/{k}')


def load_review_system(review_type: str, root: str | Path | None = None) -> dict:
    """返回 {'prompt', 'skill', 'rule', 'schema'}。"""
    from jsonschema import Draft202012Validator
    import yaml
    from agent.vlm.common import read_config
    if review_type not in REVIEW_TYPES:
        raise ValueError(f'未知复核类型：{review_type}')
    directory = Path(root or ROOT)/review_type
    schema = json.loads(read_config(directory/'schema.json'))
    Draft202012Validator.check_schema(schema)
    if schema['properties']['review_type'].get('const') != review_type:
        raise ValueError(f'{directory}/schema.json 的 review_type 与目录不一致')
    _check_unconditional(schema, f'{directory}/schema.json')
    rule = read_config(directory/'rule.yaml')
    rules = yaml.safe_load(rule)
    if rules['review_type'] != review_type or rules['schema_version'] != schema['properties']['schema_version']['const']:
        raise ValueError(f'{directory} 的 rule/schema 版本或类型不一致')
    check_name = 'degradation_check' if review_type == 'semantic' else 'height_check'
    data_fields = schema['properties']['data_check']['properties']
    fields = schema['properties'][check_name]['properties']
    contexts = {k: v.get('enum', []) for k, v in fields.items()}
    contexts.update(data_status=data_fields['status']['enum'], data_confidence=data_fields['data_confidence']['enum'],
                    consistency_check=CONSISTENCY_VALUES)
    decisions = rules['degradation_decision']
    if not decisions or decisions[-1]['when'] != {} or decisions[-1]['result'] is not None:
        raise ValueError('degradation_decision 必须以未知结论兜底')
    for index, decision in enumerate(decisions):
        if decision['result'] is not None and type(decision['result']) is not bool:
            raise ValueError('decision.result 必须为 boolean/null')
        labels = ('未知',) if decision['result'] is None else ('退化', '纵向退化') if decision['result'] else ('不退化',)
        if decision.get('degradation_type') not in labels:
            raise ValueError('decision.degradation_type 与 result 不一致')
        if not decision['when'] and index != len(decisions)-1:
            raise ValueError('兜底规则必须在最后')
        for key, values in decision['when'].items():
            if key not in contexts or not isinstance(values, list) or not values or any(v not in contexts[key] for v in values):
                raise ValueError(f'规则条件与 schema 不一致：{key}={values}')
            if key not in NON_JUDGED_KEYS and f'{key}_confidence' not in contexts:
                raise ValueError(f'判定字段 {key} 缺少 {key}_confidence')
    for constraint in rules['consistency_checks']['rules']:
        if not constraint.get('when') or not constraint.get('invalidate') or not constraint.get('reason'):
            raise ValueError('一致性约束必须包含 when、invalidate 和 reason')
        for key, values in constraint['when'].items():
            if (key not in contexts or not isinstance(values, list) or not values
                    or any(v not in contexts[key] or v in ('未知', '不确定') for v in values)):
                raise ValueError(f'一致性约束条件与 schema 不一致：{key}={values}')
        for key in constraint['invalidate']:
            unknown = '不确定'
            if key not in fields or f'{key}_confidence' not in fields or unknown not in contexts[key]:
                raise ValueError(f'一致性约束不能屏蔽字段 {key}')
    # 规则映射用于解释没有 verdict 的旧记录，新请求发送观察和判定准则。
    rule = yaml.safe_dump({key: value for key, value in rules.items()
                           if key != 'degradation_decision'}, allow_unicode=True, sort_keys=False)
    return {'prompt': read_config(directory/'prompt.md'), 'skill': read_config(directory/'skill.md'),
            'rule': rule, 'rules': rules, 'schema': schema,
            'directory': str(directory)}


def _apply_consistency(context: dict, rules: dict) -> tuple[dict, list[dict]]:
    """按 Rule 检查置信度处理后的值；同时收集冲突，只屏蔽相关字段。"""
    effective = dict(context)
    failures = []
    for constraint in rules['consistency_checks']['rules']:
        if all(context[key] in values for key, values in constraint['when'].items()):
            failures.append({'reason': constraint['reason'], 'fields': list(constraint['invalidate'])})
            for key in constraint['invalidate']:
                effective[key] = '不确定'
    return effective, failures


def _apply_low_confidence(fields: dict) -> dict:
    """rule.preprocess：<字段名>_confidence=低 的字段视为 不确定，其余字段不变。"""
    result = dict(fields)
    for key, value in fields.items():
        if key.endswith('_confidence') and value == '低':
            name = key.removesuffix('_confidence')
            result[name] = '不确定'
    return result


_NON_DEGRADED_REASON = re.compile(
    r'(?:判(?:定)?(?:为)?不退化|匹配不退化|整体不退化|该?区域[^，。；]{0,6}不退化|不构成退化|无(?:明确)?退化证据)'
)
_DEGRADED_REASON = re.compile(
    r'(?<![不非未无没])(?<!并非)(?<!不是)(?:判(?:定)?(?:为)?(?:纵向)?退化|构成(?:纵向)?退化|满足(?:纵向)?退化条件)'
)


def _verdict_contradicts_reason(verdict: dict) -> bool:
    result, reason = verdict['result'], verdict['reason']
    if result in ('退化', '纵向退化'):
        return bool(_NON_DEGRADED_REASON.search(reason))
    if result == '不退化':
        return bool(_DEGRADED_REASON.search(reason))
    return False


def verdict_evidence_problem(check: dict, data: dict, ring_evidence: str | None) -> str | None:
    """校验新版 verdict 引用当前窗口已填写的证据，保留旧版记录的解释方式。"""
    verdict = check['verdict']
    if 'evidence_refs' not in verdict or verdict['result'] == '证据不足':
        return None
    if _verdict_contradicts_reason(verdict):
        return 'verdict_reason_contradiction'
    if data['status'] in ('不可用', '不确定') or data['data_confidence'] == '低':
        return 'insufficient_data'

    refs = set(verdict['evidence_refs'])
    positions = check['longitudinal_comparisons']
    along = {f'A{i}' for i, item in enumerate(positions, 1)
             if item['position'].strip() and item['observation'].strip()}
    if 'road_comparisons' in check:
        primary_filled = {f'R{i}' for i, item in enumerate(check['road_comparisons'], 1)
                          if item['position'].strip() and item['rgb_evidence'].strip()}
        primary = {f'R{i}' for i, item in enumerate(check['road_comparisons'], 1)
                   if item['position'].strip() and item['rgb_evidence'].strip()
                   and item['rgb_basis'] != '无法确认' and item['semantic_road'] != '无法判读'}
        if check['evidence'].strip() and check['dominance'] == '明显占优' and check['road_boundary'] in ('模糊', '消失') and (
                check['dominance_confidence'] != '低' and check['road_boundary_confidence'] != '低'):
            primary.add('E')
            primary_filled.add('E')
    else:
        primary = {'H'} if check['height_comparisons']['road_and_surroundings'].strip() else set()
        primary_filled = set(primary)
    ring = {'N'} if ring_evidence and ring_evidence.strip() else set()
    if not refs or refs - (primary_filled | along | ring) or (ring_evidence is not None and 'N' not in refs):
        return 'invalid_evidence_refs'
    if verdict['result'] == '纵向退化':
        if (not {'A1', 'A2'} <= refs
                or len({item['position'].strip() for item in positions}) < 2
                or check['road_geometry'] not in ('明显直线', '近似直线')
                or check['longitudinal_structure'] != '重复'
                or check['road_geometry_confidence'] == '低'
                or check['longitudinal_structure_confidence'] == '低'):
            return 'insufficient_longitudinal_evidence'
    elif not refs & primary:
        return 'insufficient_structure_evidence'
    return None


def derive_review(output: dict, rules: dict, *, evidence_problem: str | None = None,
                  ring_evidence: str | None = None) -> dict:
    """先处理置信度和局部矛盾，再匹配规则；候选原因与最终确认原因分开记录。"""
    data = output['data_check']
    check = output['degradation_check' if output['review_type'] == 'semantic' else 'height_check']
    observed = {**check, 'data_status': data['status'],
                'data_confidence': data['data_confidence']}
    normalized = _apply_low_confidence(observed)
    context, failures = _apply_consistency(normalized, rules)
    consistency = CONSISTENCY_VALUES[1 if failures else 0]
    context['consistency_check'] = consistency
    candidates = {**normalized, 'consistency_check': consistency}
    matched = [d for d in rules['degradation_decision']
               if all(context[key] in values for key, values in d['when'].items())]
    if not matched:
        raise ValueError('缺少未知结论兜底规则')
    decision = matched[0]
    verdict = check.get('verdict')
    if verdict:
        evidence_problem = evidence_problem or verdict_evidence_problem(check, data, ring_evidence)
        label = '证据不足' if evidence_problem else verdict['result']
        decision = {
            'reason': evidence_problem or 'vlm_verdict',
            'result': {'退化': True, '纵向退化': True, '不退化': False, '证据不足': None}[label],
            'degradation_type': '未知' if label == '证据不足' else label,
            'when': {},
        }
        matched = [decision]
    candidate_reasons = list(dict.fromkeys(
        d['reason'] for d in rules['degradation_decision'] if d['result'] is True
        and all(candidates[key] in values for key, values in d['when'].items())))
    confirmed_reasons = list(dict.fromkeys(d['reason'] for d in matched if d['result'] is True))
    trace = {
        'selected_rule': decision['reason'],
        'conditions': {key: {'effective': context[key], 'accepted': values}
                       for key, values in decision['when'].items()},
        'adjustments': {key: {'observed': value, 'after_confidence': normalized[key], 'effective': context[key]}
                        for key, value in observed.items()
                        if value != normalized[key] or normalized[key] != context[key]},
    }
    if verdict:
        trace.update(reason=verdict['reason'], **{field: check[field] for field in
                                                ('road_comparisons', 'height_comparisons') if field in check})
        if 'evidence_refs' in verdict:
            trace.update(evidence_refs=verdict['evidence_refs'],
                         longitudinal_comparisons=check['longitudinal_comparisons'])
        if evidence_problem:
            trace['override'] = {'reason': evidence_problem, 'original_verdict': verdict['result']}
        candidate_reasons = []
    return {'decision_trace': trace, 'degraded': decision['result'],
            'keep': decision['result'] is True, 'reason_code': decision['reason'],
            'degradation_type': decision['degradation_type'], 'consistency_check': consistency,
            'consistency_failures': failures,
            'matched_reasons': confirmed_reasons if decision['result'] is True else [],
            'candidate_reasons': candidate_reasons}
