"""Node 7 退化区域尺寸扩展的 prompt 系统。

prompt.md / skill.md 按支路各一份，rule.yaml / schema.json 两支路共用，规则按支路补充扩展观察与停止逻辑。
退化判据不复制：请求附带 Node 6 同支路 Rule 的核心部分，每档字段沿用其 data_check 与结构字段 schema，
结论沿用同一个 derive_review 适配，两条支路都取模型 verdict。最大尺寸由 walk_tiers 按档计算，模型不选择。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from agent.vlm.block_review import REVIEW_TYPES, derive_review, load_review_system

ROOT = Path(__file__).parent
# 发送给模型的 Node 6 Rule 部分。
CORE_RULE_KEYS = ('confidence', 'uncertainty', 'data', 'degradation', 'consistency_checks', 'preprocess')
STATE_LABELS = {True: '退化', False: '不退化', None: '未知'}
NOT_EVALUATED = '未评估'
STOP_REASONS = ('no_longer_degraded', 'insufficient_evidence', 'map_boundary', 'max_tier_reached',
                'base_not_comparable', 'request_failed', 'base_incomplete')
SKIP_REASONS = ('node6_not_confirmed',)


def check_name(review_type: str) -> str:
    return 'degradation_check' if review_type == 'semantic' else 'height_check'


def tier_key(size: float) -> str:
    return f'size_{size:g}'


def load_expansion_system(review_type: str, root: str | Path | None = None) -> dict:
    """返回 {'prompt', 'skill', 'rule', 'rules', 'core_rule', 'core_rules', 'schema', 'review_schema', 'directory'}。"""
    from jsonschema import Draft202012Validator
    import yaml
    from agent.vlm.common import read_config
    if review_type not in REVIEW_TYPES:
        raise ValueError(f'未知支路：{review_type}')
    base = Path(root or ROOT)
    review = load_review_system(review_type)
    schema = json.loads(read_config(base/'schema.json'))
    Draft202012Validator.check_schema(schema)
    rules = yaml.safe_load(read_config(base/'rule.yaml'))
    if rules['schema_version'] != schema['properties']['schema_version']['const']:
        raise ValueError(f'{base} 的 rule/schema 版本不一致')
    if set(rules['stop_reasons']) != set(STOP_REASONS) or set(rules['skip_reasons']) != set(SKIP_REASONS):
        raise ValueError('rule.yaml 的 stop_reasons/skip_reasons 与程序不一致')
    confirmed = {d['reason'] for d in review['rules']['degradation_decision'] if d['result'] is True}
    if not confirmed <= set(rules['reason_labels']):
        raise ValueError(f'reason_labels 缺少 {sorted(confirmed-set(rules["reason_labels"]))}')
    branch_checks = rules['branch_checks']
    if set(branch_checks) != set(REVIEW_TYPES):
        raise ValueError('rule.yaml 的 branch_checks 必须覆盖语义和高程支路')
    model_rules = {**rules, 'branch_checks': {review_type: branch_checks[review_type]}}
    rule = yaml.safe_dump(model_rules, allow_unicode=True, sort_keys=False, width=1000).strip()
    core = {key: review['rules'][key] for key in CORE_RULE_KEYS}
    return {'prompt': read_config(base/review_type/'prompt.md'), 'skill': read_config(base/review_type/'skill.md'),
            'rule': rule, 'rules': rules, 'core_rule': yaml.safe_dump(core, allow_unicode=True, sort_keys=False, width=1000).strip(),
            'core_rules': review['rules'], 'schema': schema, 'review_schema': review['schema'], 'directory': str(base)}


def bind_expansion(system: dict, block_id: str, review_type: str, sizes: list[float]) -> dict:
    """每次请求：绑定区域、支路及有效档位；档位字段按尺寸从小到大排列，生成顺序即判断顺序。"""
    schema = copy.deepcopy(system['schema'])
    tier = schema.pop('$defs')['tier']
    name = check_name(review_type)
    check = system['review_schema']['properties']
    schema['properties']['block_id'] = {'type': 'string', 'const': block_id}
    schema['properties']['review_type'] = {'type': 'string', 'const': review_type}
    tiers = schema['properties']['tiers']
    for size in sizes:
        item = copy.deepcopy(tier)
        fields = item['properties']
        item['properties'] = {'size_m': {'type': 'integer', 'const': round(size)}, 'new_ring': fields['new_ring'],
                              'data_check': copy.deepcopy(check['data_check']), name: copy.deepcopy(check[name]),
                              'evidence': fields['evidence']}
        item['required'] = list(item['properties'])
        tiers['properties'][tier_key(size)] = item
        tiers['required'].append(tier_key(size))
    return schema


def derive_tier(tier: dict, review_type: str, core_rules: dict, *,
                evidence_problem: str | None = None) -> dict:
    """单档字段 → Node 6 同一套推导；"""
    name = check_name(review_type)
    output = {'review_type': review_type, 'data_check': tier['data_check'], name: tier[name]}
    return derive_review(output, core_rules, evidence_problem=evidence_problem, ring_evidence=tier['new_ring'])


def walk_tiers(base_size: float, sizes: list[float], valid_sizes: list[float], degraded: dict,
               blocked: str | None = None) -> tuple[float, str, dict]:
    """按档推进，返回 (最大确认尺寸, 停止原因, 各档 {state, not_evaluated_reason})。

    degraded：size → True/False/None（Rule 推导的三态）；blocked：无法评估扩展档的原因（基准不可比、请求失败）。
    无效档及其后的档位不评估；停止档位之后即使 degraded 有值也不采纳。
    """
    best, stop, states = base_size, None, {}
    for size in sizes:
        reason = None
        if size not in valid_sizes:
            reason = 'map_boundary'
            stop = stop or reason
        elif stop:
            reason = 'after_stop'
        elif blocked:
            reason = stop = blocked
        if reason:
            states[size] = {'state': NOT_EVALUATED, 'not_evaluated_reason': reason}
            continue
        result = degraded.get(size)
        states[size] = {'state': STATE_LABELS[result], 'not_evaluated_reason': None}
        if result is True:
            best = size
        else:
            stop = 'no_longer_degraded' if result is False else 'insufficient_evidence'
    return best, stop or 'max_tier_reached', states
