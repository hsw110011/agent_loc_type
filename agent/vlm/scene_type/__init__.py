"""整图道路结构类型（结构化 / 非结构化）的 prompt 系统：prompt.md（system）、skill.md（步骤）、schema.json（输出结构）。

结论映射由程序计算（decide_scene），不让 VLM 输出权重或阈值：
非结构化 → wild，结构化 → urban（配置键沿用 wild/urban）；“不确定”或置信度“低”时使用配置的回退场景。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
SCENE_KEYS = {'非结构化': 'wild', '结构化': 'urban'}


def load_scene_system(root: str | Path | None = None) -> dict:
    from jsonschema import Draft202012Validator
    from agent.vlm.common import read_config
    directory = Path(root or ROOT)
    schema = json.loads(read_config(directory/'schema.json'))
    Draft202012Validator.check_schema(schema)
    if set(schema['properties']['scene_type']['enum']) != set(SCENE_KEYS) | {'不确定'}:
        raise ValueError(f'{directory}/schema.json 的 scene_type 枚举与程序映射不一致')
    return {'prompt': read_config(directory/'prompt.md'), 'skill': read_config(directory/'skill.md'),
            'schema': schema, 'schema_version': schema['properties']['schema_version']['const'],
            'directory': str(directory)}


def decide_scene(output: dict | None, fallback: str) -> dict:
    """VLM 输出 → {'scene': wild/urban, 'source': vlm/fallback, 'reason': ...}。output=None 表示请求失败。"""
    if output is None:
        return {'scene': fallback, 'source': 'fallback', 'reason': 'vlm_error'}
    if output['scene_type'] not in SCENE_KEYS:
        return {'scene': fallback, 'source': 'fallback', 'reason': 'vlm_uncertain'}
    if output['scene_type_confidence'] == '低':
        return {'scene': fallback, 'source': 'fallback', 'reason': 'low_confidence'}
    return {'scene': SCENE_KEYS[output['scene_type']], 'source': 'vlm', 'reason': 'vlm'}
