"""Node 1b：整图场景类型（classify_scene_type，segmenter=magroad_scene 时启用）。

本地 VLM 只看整图缩略图，判断结构化 / 非结构化（prompt 系统见 agent/vlm/scene_type/）；
程序把结论映射到 MaGRoad 权重：非结构化 → magroad_wild_checkpoint，结构化 → magroad_urban_checkpoint。
VLM 失败、判“不确定”或置信度低时用 scene_type_fallback；scene_type_override 或 scene_type_known（已确认的图）可跳过 VLM。
请求结束后关闭本节点启动的 VLM 服务，把显存留给 Node 2 的 MaGRoad。
"""
from __future__ import annotations

import logging
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image

from ..global_config import GlobalPreprocessConfig
from ..io import open_image, relative_path, resolve_path, write_json
from .. import output_layout as layout
from ..state import PreprocessState


def _overview(state: PreprocessState, config: GlobalPreprocessConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """整图缩略 RGB 与有效掩码（dataset_mask，无掩码时退回大片纯黑判断），返回 (rgb, valid, 缩放比例)。"""
    from percection_tool.magroad.inference import black_border_invalid
    width, height = state['image']['width'], state['image']['height']
    scale = min(1., config.scene_vlm_image_px/max(width, height))
    size = (max(1, round(width*scale)), max(1, round(height*scale)))
    with open_image(state['image_path'], config.pillow_max_image_pixels) as source:
        if source.raster is not None:
            from rasterio.enums import Resampling
            rgb = source.raster.read([1, 2, 3], out_shape=(3, size[1], size[0]),
                                     resampling=Resampling.average).transpose(1, 2, 0)
            valid = source.raster.dataset_mask(out_shape=(size[1], size[0]), resampling=Resampling.nearest) > 0
        else:
            rgb = np.asarray(source.pillow.convert('RGB').resize(size, Image.Resampling.LANCZOS))
            valid = np.ones(rgb.shape[:2], bool)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if valid.all():
        black = black_border_invalid(rgb, 3)
        if black is not None:
            valid = ~black
    return rgb, valid, scale


def render_inputs(state: PreprocessState, config: GlobalPreprocessConfig, directory) -> dict:
    """保存 overview.png，返回送模型的说明和本地元数据。"""
    directory.mkdir(parents=True, exist_ok=True)
    rgb, valid, scale = _overview(state, config)
    overview = rgb.copy()
    overview[~valid] = 0
    overview_path = directory/'overview.png'
    Image.fromarray(overview).save(overview_path)
    resolution = state['resolution_m_per_px']
    width_m, height_m = state['image']['width']*resolution, state['image']['height']*resolution
    description = (f'整图缩略图，覆盖约 {width_m:.0f} 米 × {height_m:.0f} 米，每像素约 {resolution/scale:.1f} 米；'
                   '纯黑为地图外无数据区域。')
    return {'images': [relative_path(overview_path)], 'image_description': description,
            'valid_fraction_overview': round(float(valid.mean()), 4), 'overview_scale': scale}


IMAGE_LABELS = ['整图缩略图']


def user_prompt(description: str, system: dict) -> str:
    return (f'请判断这幅遥感地图的道路结构类型。\n\n图像说明：\n{description}\n\n'
            f"Skill：\n{system['skill']}\n\n按 API 指定的结构化输出格式返回 JSON，先观察后判断；不要添加字段。")


def classify_scene_type(state: PreprocessState, *, config: GlobalPreprocessConfig, client: Any = None) -> dict:
    from agent.vlm.scene_type import decide_scene, load_scene_system
    logger = logging.getLogger(f"map_preprocess.{state['task_id']}")
    checkpoints = {'wild': config.magroad_wild_checkpoint, 'urban': config.magroad_urban_checkpoint}
    record: dict[str, Any] = {'fallback': config.scene_type_fallback, 'checkpoints': checkpoints}
    stem = Path(state['image_path']).stem
    known = config.scene_type_known.get(stem, config.scene_type_known.get(stem.removesuffix('_dom')))
    if config.scene_type_override:
        decision = {'scene': config.scene_type_override, 'source': 'override', 'reason': 'scene_type_override'}
    elif known is not None:
        decision = {'scene': known, 'source': 'known_image', 'reason': 'scene_type_known'}
    else:
        from agent.vlm.common import _generation_schema, _parse
        from jsonschema import Draft202012Validator
        system = load_scene_system()
        directory = layout.vis_dir(state['output_dir'])/layout.FIGURES['scene_type_inputs']
        if directory.exists():
            shutil.rmtree(directory)
        inputs = render_inputs(state, config, directory)
        record.update({'schema_version': system['schema_version'], 'prompt_system': relative_path(system['directory']),
                       'inputs': inputs})
        owned = client is None
        if owned:
            from agent.vlm.client import VLMClient
            client = VLMClient()
        output = None
        budget = config.scene_vlm_thinking_budget
        effort = config.scene_vlm_reasoning_effort
        thinking = {'reasoning_effort': effort} if budget and effort is not None else {}
        try:
            result = client.infer(user_prompt(inputs['image_description'], system),
                                  images=[resolve_path(p) for p in inputs['images']], system=system['prompt'],
                                  max_tokens=config.scene_vlm_max_tokens, temperature=config.scene_vlm_temperature, json_output=True,
                                  output_schema=_generation_schema(system['schema']), image_labels=IMAGE_LABELS,
                                  enable_thinking=budget > 0, thinking_token_budget=budget or None, **thinking)
            record.update({'raw_text': result.text, 'reasoning': getattr(result, 'reasoning', ''),
                           'usage': result.usage, 'latency_seconds': round(result.latency_seconds, 2)})
            seconds, completion = getattr(result, 'inference_seconds', 0.), (result.usage or {}).get('completion_tokens') or 0
            logger.info('Node 1b VLM：输出 %d tokens，%.1f tok/s', completion, completion/seconds if seconds > 0 else 0.)
            output = _parse(result.text)
            Draft202012Validator(system['schema']).validate(output)
            record.update({'status': 'ok', 'output': output})
        except Exception as exc:  # 场景判断失败不阻断流程，按回退场景继续
            output = None
            record.update({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'})
            logger.warning('Node 1b VLM 场景判断失败，使用回退场景 %s：%s', config.scene_type_fallback, exc)
        finally:
            # 停止本客户端启动的 VLM 服务（注入的客户端同样会在后续节点按需重启），给 MaGRoad 让出显存。
            client.close()
        decision = decide_scene(output, config.scene_type_fallback)
    checkpoint = checkpoints[decision['scene']]
    summary = {**decision, 'magroad_checkpoint': checkpoint,
               'vlm_scene_type': (record.get('output') or {}).get('scene_type'),
               'vlm_confidence': (record.get('output') or {}).get('scene_type_confidence'),
               'vlm_status': record.get('status', 'skipped')}
    path = layout.state_dir(state['output_dir'])/'scene_type.json'
    write_json(path, {**record, 'decision': summary})
    logger.info('Node 1b 场景=%s（来源 %s），MaGRoad 权重=%s', decision['scene'], decision['source'], checkpoint)
    return {'scene_type': {**summary, 'record_path': relative_path(path)}}
