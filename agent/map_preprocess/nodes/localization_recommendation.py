"""Node 9：退化区域辅助定位方式推荐（recommend_localization）。

对 Node 8 的每个融合区域独立请求一次 VLM，推荐 gps、lidar_inertial_slam、wheel_imu_odometry 之一（prompt 系统见
agent/vlm/localization_recommendation/）。模型不接收也不返回坐标：程序按 Node 8 的 bbox 裁图，并按 region_id 把 bbox、
degradation_type、union_polygon 原样补回。不重新判断退化，不改 Node 8 的结果，不发布 ROS。

每个 Node 8 区域只出现在 regions 或 failed_regions 之一；无效输出经有限重试后记入 failed_regions，不生成推荐。
结果写入 state/localization_recommendations.json 与 extensions.localization_recommendation。
"""
from __future__ import annotations

import logging
import math
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..global_config import GlobalPreprocessConfig, PIPELINE_VERSION
from ..io import load_manifest, open_image, read_json, relative_path, resolve_path, write_json
from .. import output_layout as layout
from ..state import PreprocessState
from ..utils.result_publication import protect_stage_outputs, publish_stage
from ..utils import vlm_review_inputs as vi
from .degradation_fusion import union_rings
from .vlm_review import failure_fields, run_jobs

IMAGE_LABELS = ['第 1 张：原始 RGB 影像']
BOUNDARY_RGB = (255, 230, 0)
METHOD_COLORS = {'gps': (40, 220, 60), 'lidar_inertial_slam': (40, 140, 255), 'wheel_imu_odometry': (255, 140, 0)}
FAILED_COLOR = (150, 150, 150)


def crop_box(bbox: list, width: int, height: int) -> list[int] | None:
    """Node 8 的 bbox 取整并限制在原图内的读取窗口；完全在图外时返回 None。"""
    x0, y0 = max(0, math.floor(bbox[0])), max(0, math.floor(bbox[1]))
    x1, y1 = min(width, math.ceil(bbox[2])), min(height, math.ceil(bbox[3]))
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def image_size(crop: list[int], max_side: int) -> tuple[int, int]:
    """保持长宽比，长边不超过 max_side，不放大。"""
    width, height = crop[2]-crop[0], crop[3]-crop[1]
    scale = min(1., max_side/max(width, height))
    return max(1, round(width*scale)), max(1, round(height*scale))


def read_crop(manifest: dict, crop: list[int], size: tuple[int, int]) -> np.ndarray:
    with open_image(manifest['image']['path'], manifest['config']['pillow_max_image_pixels']) as source:
        rgb = source.read(crop)
        valid = source.valid_mask(crop)
    return cv2.resize(np.where(valid[..., None], rgb, 0).astype(np.uint8), size, interpolation=cv2.INTER_AREA)


def image_rings(region: dict, crop: list[int], size: tuple[int, int]) -> list[list[list[tuple[float, float]]]]:
    """union_polygon 转到图像坐标：global = crop 原点 + image / 缩放比。"""
    sx, sy = size[0]/(crop[2]-crop[0]), size[1]/(crop[3]-crop[1])
    return [[[((x-crop[0])*sx, (y-crop[1])*sy) for x, y in ring] for ring in rings] for rings in union_rings(region)]


def draw_outline(array: np.ndarray, rings: list):
    """黄芯黑边线标出真实融合区域，不覆盖区域内部的影像细节。"""
    from PIL import Image, ImageDraw
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    for rings_of_polygon in rings:
        for ring in rings_of_polygon:
            draw.line(ring, fill=(0, 0, 0), width=7, joint='curve')
            draw.line(ring, fill=BOUNDARY_RGB, width=3, joint='curve')
    return image


def prepare_job(manifest: dict, region: dict, config: GlobalPreprocessConfig, input_dir: Path) -> dict | None:
    """渲染一个区域的输入图并写本地说明；bbox 完全在图外时返回 None。"""
    full = manifest['image']['width'], manifest['image']['height']
    crop = crop_box(region['bbox'], *full)
    if crop is None:
        return None
    size = image_size(crop, config.vlm_localization_image_px)
    rings = image_rings(region, crop, size)
    directory = input_dir/region['merged_id']
    path = vi.save_png(draw_outline(read_crop(manifest, crop, size), rings), directory/'rgb.png')
    resolution = manifest['resolution_m_per_px']
    payload = {'region_id': region['merged_id'], 'bbox': region['bbox'], 'crop_global_bbox': crop, 'image_size_px': list(size),
               'global_px_per_image_px': [(crop[2]-crop[0])/size[0], (crop[3]-crop[1])/size[1]],
               'image_extent_m': [(crop[2]-crop[0])*resolution, (crop[3]-crop[1])*resolution],
               'm_per_image_px': (crop[2]-crop[0])*resolution/size[0],
               'full_frame': region['union_area_px2'] == (crop[2]-crop[0])*(crop[3]-crop[1])}
    job = {'region_id': region['merged_id'], 'images': [relative_path(path)], 'payload': payload}
    write_json(directory/'input.json', {'region_id': job['region_id'], 'image_description': image_description(payload)})
    write_json(directory/'metadata.json', payload)
    return job


def image_description(payload: dict) -> str:
    """模型只收到区域 ID、实际尺度和边界线含义；退化类型不发送，坐标、裁剪窗口留在 metadata。"""
    width, height = payload['image_extent_m']
    return '\n'.join([f'区域：{payload["region_id"]}。这是上游识别出的潜在退化区域，不需要分析退化原因。',
                      f'图像实际覆盖宽 {width:.0f} 米、高 {height:.0f} 米，比例尺约 {payload["m_per_image_px"]:.3g} 米/图像像素。',
                      '黄芯黑边线贴着画面四边，整幅图都在真实融合区域内，没有线外部分，画面中的所有地物都是评估对象。' if payload['full_frame']
                      else '黄芯黑边线是真实融合区域的边界：线内是评估对象，线外只是周边环境。'])


def user_prompt(job: dict, system: dict, feedback: str | None = None) -> str:
    prompt = (f"请为区域 {job['region_id']} 推荐一种辅助定位方式。\n\n图像说明：\n{image_description(job['payload'])}\n\n"
              f"Skill：\n{system['skill']}\n\nRule：\n{system['rule']}\n\n按 API 指定的结构化输出格式返回 JSON，不要添加字段。")
    return f'{prompt}\n\n上一次输出未通过校验：{feedback}\n请修正后重新输出。' if feedback else prompt


def recommend_one(job: dict, client: Any, system: dict, config: GlobalPreprocessConfig, result_dir: Path) -> dict:
    """有限重试：输出无效时把校验信息反馈给模型；服务瞬时错误原样重试；其他异常不重试。"""
    from jsonschema import ValidationError
    from agent.vlm.common import PositioningError, VLMError, _generation_schema, _parse
    from agent.vlm.localization_recommendation import InvalidRecommendation, bind_region, check_recommendation
    region_id, budget = job['region_id'], config.vlm_localization_thinking_budget
    effort = config.vlm_localization_reasoning_effort
    thinking = {'reasoning_effort': effort} if budget != 0 and effort is not None else {}
    schema = bind_region(system['schema'], region_id)
    record: dict = {'region_id': region_id, 'images': job['images'], 'schema_version': system['rules']['schema_version'], 'attempts': []}
    feedback = None
    for attempt in range(config.vlm_localization_max_retries+1):
        try:
            result = client.infer(user_prompt(job, system, feedback), images=[resolve_path(path) for path in job['images']], system=system['prompt'],
                                  max_tokens=config.vlm_localization_max_tokens, temperature=0.0, json_output=True,
                                  output_schema=_generation_schema(schema), image_labels=IMAGE_LABELS,
                                  enable_thinking=budget != 0, thinking_token_budget=budget or None, **thinking)
            record.update({'raw_text': result.text, 'usage': result.usage, 'latency_seconds': round(result.latency_seconds, 2),
                           'inference_seconds': round(getattr(result, 'inference_seconds', 0.), 2)})
            output = _parse(result.text)
            check_recommendation(output, schema, system['rules'])
            record.update({'status': 'ok', 'output': output})
            record.pop('error', None)
            break
        except (PositioningError, ValidationError, InvalidRecommendation) as exc:
            feedback = getattr(exc, 'message', None) or str(exc)
            record.update(failure_fields(exc))
        except VLMError as exc:
            feedback = None
            record.update(failure_fields(exc))
        except Exception as exc:
            record.update(failure_fields(exc))
            record['attempts'].append({'attempt': attempt+1, 'error': record['error']})
            break
        record['attempts'].append({'attempt': attempt+1, 'error': record['error']})
    write_json(result_dir/f'{region_id}.json', record)
    return record


def unchanged_fields(region: dict) -> dict:
    """来自 Node 8 的不可变字段，模型输出不参与。"""
    return {'region_id': region['merged_id'], 'degradation_type': region['degradation_type'], 'bbox': list(region['bbox'])}


def final_region(region: dict, output: dict, system: dict) -> dict:
    from agent.vlm.localization_recommendation import apply_program_rules
    program = apply_program_rules(output, system['rules'])
    return {**unchanged_fields(region), 'recommended_method': output['recommended_method'], 'assessment': output['assessment'], 'reason': output['reason'],
            'visual_evidence': output['visual_evidence'], 'limitations': program['limitations'],
            'recommendation_status': program['recommendation_status'],
            'union_polygon': region['union_polygon'], 'source_patch_ids': region['source_patch_ids']}


def failed_region(region: dict, error: str) -> dict:
    return {**unchanged_fields(region), 'error': error, 'union_polygon': region['union_polygon'], 'source_patch_ids': region['source_patch_ids']}


def render_overview(manifest: dict, config: GlobalPreprocessConfig, regions: list[dict], failed: list[dict], output: Path) -> Path:
    """全图推荐结果：按定位方式着色，失败区域为灰色。"""
    from PIL import Image, ImageDraw
    from ..utils.global_render import read_preview
    preview, scale = read_preview(manifest['image'], config)
    image = preview.convert('RGB')
    side = max(image.size)
    width = max(2, side//700)
    font = vi._font(max(12, side//160))
    items = [(region, METHOD_COLORS[region['recommended_method']]) for region in regions]+[(region, FAILED_COLOR) for region in failed]

    def scaled(ring: list) -> list[tuple[float, float]]:
        return [(x*scale, y*scale) for x, y in ring]

    for region, color in items:
        mask = Image.new('L', image.size, 0)
        painter = ImageDraw.Draw(mask)
        for exterior, *holes in union_rings(region):
            painter.polygon(scaled(exterior), fill=110)
            for hole in holes:
                painter.polygon(scaled(hole), fill=0)
        image.paste(color, (0, 0), mask)
    draw = ImageDraw.Draw(image)
    for region, color in items:
        for rings in union_rings(region):
            for ring in rings:
                draw.line(scaled(ring), fill=color, width=2*width, joint='curve')
        suffix = f" {region['recommended_method']}" if 'recommended_method' in region else ' FAILED'
        suffix += ' (validate)' if region.get('recommendation_status') == 'requires_validation' else ''
        draw.text((region['bbox'][0]*scale+3, region['bbox'][1]*scale+1), region['region_id']+suffix, fill='white', font=font, stroke_width=2, stroke_fill='black')
    legend = [*((color, name) for name, color in METHOD_COLORS.items()), (FAILED_COLOR, 'failed')]
    vi.add_legend(image, legend, 'Recommended auxiliary localization (satellite-based, needs online validation)').save(output)
    return output


def run_localization_recommendation(manifest_path: str | Path, config: GlobalPreprocessConfig, *, client: Any = None,
                                    dry_run: bool | None = None, logger: logging.Logger | None = None) -> dict:
    """为 Node 8 的每个区域推荐辅助定位方式；返回摘要，完整结果写入 state/localization_recommendations.json。"""
    from agent.vlm.localization_recommendation import load_recommendation_system
    logger = logger or logging.getLogger('map_preprocess.localization_recommendation')
    dry_run = config.vlm_localization_dry_run if dry_run is None else dry_run
    manifest_path = resolve_path(manifest_path)
    manifest = load_manifest(manifest_path)
    regions_path = manifest['files'].get('degradation_regions')
    if not regions_path:
        raise ValueError('需要先运行 Node 8（degradation_regions.json）')
    document = read_json(resolve_path(regions_path))
    regions = document['regions']
    if len({region['merged_id'] for region in regions}) != len(regions):
        raise ValueError('Node 8 的 merged_id 不唯一，无法一一对应')
    system = load_recommendation_system()
    task_dir = layout.task_dir_of_manifest(manifest_path)
    if dry_run:
        input_dir, result_dir = layout.diagnostic_dirs(task_dir, 'node9', dry_run=True)
    else:
        input_dir = layout.figure(task_dir, 'localization_inputs')
        result_dir = layout.state_dir(task_dir)/'localization_recommendation'
    output = layout.state_dir(task_dir)/'localization_recommendations.json'
    overview_path = layout.figure(task_dir, 'localization_recommendation')
    outputs = (input_dir, result_dir, output, overview_path)
    publication = nullcontext() if dry_run else protect_stage_outputs(manifest_path, outputs)
    with publication:
        for directory in (input_dir, result_dir):
            if directory.exists():
                shutil.rmtree(directory)
        result_dir.mkdir(parents=True)

        jobs, errors = [], {}
        for region in regions:
            job = prepare_job(manifest, region, config, input_dir)
            if job is None:
                errors[region['merged_id']] = 'bbox_outside_image'
            else:
                jobs.append(job)
        logger.info('Node 9：%d 个融合区域，VLM 任务 %d', len(regions), len(jobs))
        summary: dict = {'pipeline_version': PIPELINE_VERSION, 'status': 'dry_run' if dry_run else 'completed', 'schema_version': system['rules']['schema_version'], 'region_count': len(regions),
                         'job_count': len(jobs), 'inputs_dir': relative_path(input_dir), 'results_dir': relative_path(result_dir)}
        if dry_run:
            for job in jobs:
                labelled = ''.join(f'{label}\n<image {path}>\n' for label, path in zip(IMAGE_LABELS, job['images']))
                (result_dir/f"{job['region_id']}_prompt.txt").write_text(f"[system]\n{system['prompt']}\n\n[user]\n{labelled}{user_prompt(job, system)}\n", encoding='utf-8')
            return summary

        records: list[dict] = []
        if jobs:
            owned = client is None
            if owned:
                from agent.vlm.client import VLMClient
                client = VLMClient()
            try:
                records = run_jobs(lambda job: recommend_one(job, client, system, config, result_dir), jobs, config.vlm_localization_workers, 'Node 9 VLM 定位推荐', logger)
            finally:
                if owned:
                    client.close()
        by_id = {record['region_id']: record for record in records}
        results, failed = [], []
        for region in regions:  # 按 Node 8 的顺序组装，每个区域只查自己的记录
            record = by_id.get(region['merged_id'], {})
            if record.get('status') == 'ok':
                results.append(final_region(region, record['output'], system))
            else:
                error = errors.get(region['merged_id']) or record.get('error', 'no_result')
                logger.warning('Node 9：区域 %s 没有有效推荐（%s）', region['merged_id'], error)
                failed.append(failed_region(region, error))
        overview = render_overview(manifest, config, results, failed, overview_path)
        write_json(output, {'pipeline_version': PIPELINE_VERSION, 'task_id': manifest.get('task_id'), 'image': manifest['image']['path'], 'resolution_m_per_px': manifest['resolution_m_per_px'],
                            'coordinate_system': 'global_pixel', 'schema_version': system['rules']['schema_version'],
                            'meaning': '推荐仅基于卫星影像的环境线索，不代表定位效果已验证',
                            'regions': results, 'failed_regions': failed})
        methods: dict[str, int] = {}
        for region in results:
            methods[region['recommended_method']] = methods.get(region['recommended_method'], 0)+1
        summary.update({'status': 'partial' if failed else 'completed', 'recommended_count': len(results), 'failed_count': len(failed),
                        'method_counts': methods,
                        'errors': [{'region_id': region['region_id'], 'error': region['error']} for region in failed],
                        'regions': [{key: region[key] for key in ('region_id', 'degradation_type', 'bbox', 'recommended_method', 'recommendation_status')}
                                    for region in results],
                        'failed_region_ids': [region['region_id'] for region in failed],
                        'localization_recommendations': relative_path(output), 'overview': relative_path(overview)})
        manifest.setdefault('extensions', {})['localization_recommendation'] = summary
        manifest['files'].update({'localization_recommendations': relative_path(output), 'localization_recommendation_overview': relative_path(overview)})
        publish_stage(manifest_path, manifest, 'localization_recommendation')
        logger.info('Node 9 完成：推荐 %d（%s），失败 %d', len(results), methods, len(failed))
        return summary


def recommend_localization(state: PreprocessState, *, config: GlobalPreprocessConfig, client: Any = None) -> dict:
    if (state.get('degradation_fusion') or {}).get('status') == 'skipped':  # Node 8 没有产出融合区域
        return {'localization_recommendation': {'status': 'skipped', 'reason': 'node8_skipped'}}
    summary = run_localization_recommendation(state['manifest_path'], config, client=client, logger=logging.getLogger(f"map_preprocess.{state['task_id']}"))
    return {'localization_recommendation': summary}
