"""Node 9 退化区域辅助定位方式推荐的 prompt 系统。

prompt 定义角色与任务边界，skill 定义分析步骤，rule.yaml 定义三种方式的适用范围与选择规则，schema.json 定义输出结构。
模型只接收带边界的 RGB、区域 ID 和尺度，不接收退化类型和坐标；bbox、degradation_type 由程序按 region_id 从 Node 8 结果补充。
rule.yaml 的 program 一节只由程序读取：强制的验证状态和通用限制由 apply_program_rules 附加，不依赖模型是否写出。
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from agent.vlm.block_review import _check_unconditional

ODOMETRY_METHODS = ('lidar_inertial_slam', 'wheel_imu_odometry')
METHODS = (*ODOMETRY_METHODS, 'gps')
STATUSES = ('provisional', 'requires_validation')
ROOT = Path(__file__).parent
PROMPT_NAME = 'node9_localization_recommendation_prompt.md'


class InvalidRecommendation(ValueError):
    """模型输出通过了结构校验，但违反推荐规则。"""


def load_recommendation_system(root: str | Path | None = None) -> dict:
    """返回 {'prompt', 'skill', 'rule', 'rules', 'schema', 'directory'}；rule 是发送给模型的文本，不含 program。"""
    from jsonschema import Draft202012Validator
    import yaml
    from agent.vlm.common import read_config
    base = Path(root or ROOT)
    schema = json.loads(read_config(base/'schema.json'))
    Draft202012Validator.check_schema(schema)
    _check_unconditional(schema, f'{base}/schema.json')
    rules = yaml.safe_load(read_config(base/'rule.yaml'))
    fields = schema['properties']
    if tuple(fields['recommended_method']['enum']) != METHODS or tuple(fields['recommendation_status']['enum']) != STATUSES:
        raise ValueError(f'{base}/schema.json 的枚举与程序不一致')
    program = rules['program']
    if set(rules['methods']) != set(METHODS) or set(program['mandatory_limitations']) != set(METHODS):
        raise ValueError(f'{base}/rule.yaml 的定位方式与程序不一致')
    if {key: value.get('priority') for key, value in rules['methods'].items()} != {method: f'P{rank}' for rank, method in enumerate(METHODS)}:
        raise ValueError(f'{base}/rule.yaml 的定位优先级与程序不一致')
    if set(rules['status']) != set(STATUSES):
        raise ValueError(f'{base}/rule.yaml 的 status 与程序不一致')
    if not set(program['always_requires_validation']) <= set(METHODS) or not all(program['mandatory_limitations'].values()):
        raise ValueError(f'{base}/rule.yaml 的 program 配置不完整')
    visible = {key: value for key, value in rules.items() if key != 'program'}
    return {'prompt': read_config(base/PROMPT_NAME), 'skill': read_config(base/'skill.md'),
            'rule': yaml.safe_dump(visible, allow_unicode=True, sort_keys=False, width=1000).strip(),
            'rules': rules, 'schema': schema, 'directory': str(base)}


def bind_region(schema: dict, region_id: str) -> dict:
    """每次请求：region_id 绑定为常量，生成时强制原样返回。"""
    schema = copy.deepcopy(schema)
    schema['properties']['region_id'] = {'type': 'string', 'const': region_id}
    return schema


def overclaims(output: dict, rules: dict) -> list[str]:
    """reason、visual_evidence 中未被否定的过度承诺用语；“不能保证定位成功”这样的否定说法不算。"""
    program = rules['program']
    found = set()
    for text in (output['reason'], *output['visual_evidence']):
        for phrase in program['overclaim_phrases']:
            start = text.find(phrase)
            while start != -1:
                if not set(text[max(0, start-4):start]) & set(program['negations']):
                    found.add(phrase)
                start = text.find(phrase, start+1)
    return sorted(found)


def reliability_numbers(output: dict, rules: dict) -> list[str]:
    """模型文本中的可靠性数值（百分比、评分等）；这些数值没有实验标定，只允许定性比较。"""
    patterns = rules['program']['reliability_number_patterns']
    texts = (output['reason'], *output['visual_evidence'], *output['limitations'])
    return sorted({match.group() for pattern in patterns for text in texts for match in re.finditer(pattern, text)})


def check_recommendation(output: dict, schema: dict, rules: dict) -> None:
    """检查结构、证据文字及推荐与各项适用条件的一致性。"""
    from jsonschema import Draft202012Validator
    Draft202012Validator(schema).validate(output)
    if any(not text.strip() for text in (output['reason'], *output['visual_evidence'], *output['limitations'])):
        raise InvalidRecommendation('reason、visual_evidence、limitations 不能是空白文本')
    claims = overclaims(output, rules)
    if claims:
        raise InvalidRecommendation(f'不得声称定位已获保证，请删除或改写这些用语：{"、".join(claims)}')
    numbers = reliability_numbers(output, rules)
    if numbers:
        raise InvalidRecommendation(f'只做定性比较，不得给出百分比、评分等数值：{"、".join(numbers)}')
    method, assessment = output['recommended_method'], output['assessment']
    if assessment['structure_density'] == 'dense':
        if assessment['lidar_inertial_slam'] == 'unsuitable':
            raise InvalidRecommendation('结构密集区域（structure_density=dense）具备三维结构线索，lidar_inertial_slam 不应判为 unsuitable')
        if method != 'lidar_inertial_slam':
            raise InvalidRecommendation('结构密集区域（structure_density=dense）可使用 lidar_inertial_slam 过渡，不得改选 wheel_imu_odometry 或 gps')
    suitable = next((key for key in ODOMETRY_METHODS if assessment[key] == 'suitable'), None)
    if method == 'gps':
        if (any(assessment[key] != 'unsuitable' for key in ODOMETRY_METHODS) or assessment['sky_view'] != 'open'
                or assessment['obstruction_or_reflection'] != 'absent'):
            raise InvalidRecommendation('GPS 要求 LIO 和轮速均为 unsuitable、sky_view=open 且 obstruction_or_reflection=absent')
    elif suitable and method != suitable:
        raise InvalidRecommendation(f'优先推荐适用的方法 {suitable}')
    elif not suitable and output['recommendation_status'] != 'requires_validation':
        raise InvalidRecommendation('暂定的里程方式需要 recommendation_status=requires_validation')


def apply_program_rules(output: dict, rules: dict) -> dict:
    """返回程序确定的 recommendation_status 与 limitations：只收紧状态、只增加限制，不改模型选择的方式。"""
    program = rules['program']
    method = output['recommended_method']
    forced = method in program['always_requires_validation']
    status = 'requires_validation' if forced else output['recommendation_status']
    return {'recommendation_status': status,
            'limitations': list(dict.fromkeys([*output['limitations'], *program['mandatory_limitations'][method]]))}
