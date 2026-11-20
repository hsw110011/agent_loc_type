"""Node 6：退化块 VLM 复核（vlm_review_blocks）。

按 Node 5 的候选支路分流；有有效 DSM 时，路面基准无效或道路中心向上阶跃也进入高程复核。两套 prompt 系统独立：
  semantic：局部 RGB（无标记）+ 局部语义分割（同范围）+ 整体语义上下文（通常为 100 m，RGB 叠半透明语义和目标框）
  height：整体 RGB 上下文（默认 100 m，标目标框）+ 局部 RGB（无标记）+ 局部 DSM 伪彩色
                       （DSM − 上下文范围有效 DSM 的 P20，固定 [-2, 8] m 线性 Viridis 色标，所有块同一色标）
局部图统一裁去地图外部分、保留长宽比例，同尺寸、像素一一对应（图上不画图例，颜色含义、色标和参考高程由本次图像说明提供）；每张图前插入文字标签。

VLM 独立判断：请求不发送 Node 5 的初筛结论、阈值或统计值；道路中心向上阶跃时附条件核查提示。
模型只接收图像说明、实际覆盖宽高、中心位置及颜色含义；完整几何与统计留作本地元数据。Schema 保留英文字段名，判断枚举为中文，每个判定字段输出独立置信度。
退化结论来自模型 verdict；程序校验证据并记录结构字段的置信度和一致性。
本阶段不判断定位方式切换或传感器可见性。Node 6 不判断范围扩展，退化场景的范围就是当前块；扩展由 Node 7 单独记录。
保留的块写 blocks[].vlm_review。首次运行备份 manifest，重跑从备份开始。
"""
from __future__ import annotations

import logging
import math
import shutil
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from ..run_logging import VLMMeter

from ..global_config import GlobalPreprocessConfig
from ..io import load_manifest, read_json, relative_path, resolve_path, write_json
from .. import output_layout as layout
from ..state import PreprocessState
from ..utils.result_publication import protect_stage_outputs, publish_stage
from ..utils import block_features as bf
from ..utils import vlm_review_inputs as vi

BACKUP_NAME = 'manifest_before_vlm_review.json'
SEMANTIC_CONTEXT_M = 100.  # 语义上下文的目标边长，至少覆盖整个局部 Patch
CONTEXT_TRANSPARENCY = .7  # 上下文图中语义颜色的透明度（0 = 纯语义色，1 = 纯 RGB）
PATCH_BOX_RGB = (255, 230, 0)


def review_types(block: dict) -> list[str]:
    """规则命中进入对应支路；有有效 DSM 时，路面基准无效或中心向上阶跃也进入高程复核。"""
    features = block.get('features') or {}
    d = features.get('degradation') or {}
    reference = features.get('road_reference') or {}
    kinds = ['semantic'] if d.get('semantic_degraded') else []
    height_fallback = reference.get('valid') is False and 'height' in block.get('review_types_available', ())
    center_step_above = (features.get('height_stats') or {}).get('center_step_above')
    if d.get('height_degraded') or height_fallback or center_step_above:
        kinds.append('height')
    return kinds


def backup_path(manifest_path: str | Path) -> Path:
    return resolve_path(manifest_path).parent/BACKUP_NAME


def _semantic_labels(manifest: dict, config: GlobalPreprocessConfig, segmenter: Any, logger: logging.Logger) -> np.ndarray:
    """整图语义（语义路线已缓存，工作网格见 semantic_roads_resolution_m）；低置信度像元记为 other，与 Node 5 一致。"""
    from .semantic_roads import whole_image_semantic
    state = {'image': manifest['image'], 'resolution_m_per_px': manifest['resolution_m_per_px'],
             'input_sha256': manifest['input_sha256'], 'image_path': manifest['image']['path']}
    labels, confidence, info = whole_image_semantic(state, config, segmenter, logger)
    logger.info('Node 6 整图语义：%s（cache_hit=%s）', info['working_size'], info['cache_hit'])
    return bf.semantic_labels(labels, confidence, config.feature_min_class_confidence)


def _payload(block: dict, review_type: str, edge: dict | None, geometry: dict, stats: dict | None,
             height: dict | None) -> dict:
    """本地元数据：几何信息和中性统计，供记录及程序检查，不直接发送给 VLM。"""
    tx, ty = block['tangent_xy']
    mx, my = geometry['m_per_image_px']
    payload = {'block_id': block['block_id'], 'review_type': review_type, **geometry,
               'road_direction_deg': round(math.degrees(math.atan2(ty/my, tx/mx)) % 360, 1),
               'road_source': (edge or {}).get('source', 'samroad')}
    if review_type == 'semantic':
        return {**payload, **stats}
    stats = (block.get('features') or {}).get('height_stats') or {}
    return {**payload, **height, 'center_step_above': bool(stats.get('center_step_above'))}


def model_input(payload: dict) -> dict:
    """模型只接收图像 + 图像说明：每张图是什么、覆盖多大、颜色和透明度代表什么；不发送统计、参考高程等数据，均留在 metadata。"""
    kind = payload['review_type']
    width, height = payload['image_extent_m']
    cx, cy = payload['center_image_xy']
    iw, ih = payload['image_size_px']
    cw, ch = payload['context']['context_extent_m']
    where = f'道路区域中心位于图像横向约 {100*cx/iw:.0f}%、纵向约 {100*cy/ih:.0f}% 处（未贴地图边缘时即图像正中）。'
    crop_note = '各图由程序按窗口裁剪和缩放；贴地图边缘时只显示地图内范围。'
    if kind == 'semantic':
        semantic_pct = round(100*(1-CONTEXT_TRANSPARENCY))
        description = ('第 1 张：局部 RGB（local_rgb）。从原始卫星 RGB 裁出的目标区域，不叠加任何标记，用于看清道路和周边的真实地物。\n'
                       '第 2 张：局部语义分割（local_semantic）。与第 1 张覆盖完全相同、像素一一对应，从整幅卫星图的自动语义分割结果裁出，像素颜色表示预测类别，'
                       '不叠加原始影像，也不画中心线；分割可能有错分，需要与第 1 张对照。\n'
                       f'第 1、2 张实际覆盖宽 {width:.4g} 米、高 {height:.4g} 米；{where}\n'
                       f'第 3 张：整体语义上下文（context_semantic）。以目标区域为中心，实际覆盖宽 {cw:.4g} 米、高 {ch:.4g} 米。'
                       f'在原始卫星影像上半透明叠加与第 2 张相同的语义颜色：语义颜色占 {semantic_pct}%、原始影像占 {100-semantic_pct}%，'
                       '混合颜色的深浅随底图变化，不表示分割置信度；语义无数据处只显示 RGB。程序添加的黄框是范围标记，圈出第 1、2 张的覆盖范围。'
                       '第 3 张只用于看道路在框外的走向、衔接和周边结构，判断对象是黄框内区域。\n')
        return {'block_id': payload['block_id'], 'review_type': kind,
                'image_description': description+crop_note+'\n'+color_description(kind)}
    description = (f'第 1 张：整体 RGB 上下文（context_rgb）。以目标区域为中心的原始卫星影像，实际覆盖宽 {cw:.4g} 米、高 {ch:.4g} 米，不叠加颜色；'
                   '程序添加的黄框是范围标记，圈出第 2、3 张的覆盖范围。用于理解周围地形、道路延伸、桥梁及疑似隧道结构，判断对象是黄框内区域。\n'
                   '第 2 张：局部 RGB（local_rgb）。从原始卫星 RGB 裁出的目标区域，不叠加任何标记，用于辨认目标道路及高程结构对应的真实地物。\n'
                   '第 3 张：局部 DSM 伪彩色高程图（local_dsm）。与第 2 张覆盖完全相同、像素一一对应，由 DSM 表面高程减去周边低处的着色基准后生成，显示的是相对高度，可能对应路面、树冠、屋顶或桥面。图中没有叠加 RGB、影像阴影或标记。\n'
                   f'第 2、3 张实际覆盖宽 {width:.4g} 米、高 {height:.4g} 米；{where}\n')
    if payload.get('center_step_above'):
        description += ('补充核查：请对照 RGB 和 DSM，检查道路中央是否有树冠、屋顶或其他俯视遮挡，'
                        '道路是否可能延伸到可供车辆进入的隧道或下穿入口，或建筑是否实际占据道路中央。'
                        '沿道路走向核对入口、遮挡前后的道路接续和结构位置，证据不足时保留不确定。\n')
    return {'block_id': payload['block_id'], 'review_type': kind, 'image_description': description+crop_note+'\n'+color_description(kind)}


# 高程图的颜色说明（Node 6、Node 7 共用）：只讲颜色代表的高低，不给参考高程或米数（这些留在 metadata）。
DSM_DESCRIPTION = ('高程着色：高程图使用 Viridis 连续伪彩色色带表示高程变化。颜色与高程的对应关系：'
                   '深紫色=较低高程；深蓝色=低至中等高程；蓝绿色/青绿色=中等高程；绿色=较高高程；亮黄色=最高高程区域。'
                   '颜色从深紫色 → 蓝色 → 蓝绿色 → 绿色 → 亮黄色变化时，代表 DSM 高程逐渐增加。'
                   '所有区域使用同一条固定色带，以周围区域较低的地面为着色基准，同一颜色在不同区域代表相近的相对高低；颜色不是绝对海拔。'
                   '色带两端以外的值不再变色：比色带更低的仍为深紫色，更高的（如高楼）仍为亮黄色。白色=无数据，不等于低洼或平地。')


def color_description(kind: str) -> str:
    """专题图颜色含义（Node 6、Node 7 共用）；输入图都不画道路拓扑线。"""
    if kind == 'semantic':
        return '语义颜色：红色=道路或硬化地面；绿色=植被；蓝色=建筑；品红=其它或分类不确定；深灰=无数据。'
    return DSM_DESCRIPTION


def height_reference(manifest: dict, block: dict, config: GlobalPreprocessConfig) -> dict:
    """参考高程 H_ref：以块中心为中心、vlm_review_height_context_m 范围内有效 DSM 的百分位（Node 6、Node 7 共用，结果相同）。
    只是着色基准，不是路面高程；没有有效 DSM 时 h_ref_m 为 None。"""
    size = (manifest['image']['width'], manifest['image']['height'])
    window = vi.review_window(block, config.vlm_review_height_context_m, manifest['resolution_m_per_px'])
    h_ref, valid_pct = vi.dsm_reference(vi.read_dsm(manifest['image']['dsm'], size, window, config.vlm_review_image_px),
                                        config.vlm_review_dsm_ref_percentile)
    return {'h_ref_m': None if h_ref is None else round(h_ref, 3), 'percentile': config.vlm_review_dsm_ref_percentile,
            'source': 'context_window_valid_dsm', 'context_valid_pct': valid_pct}


def prepare_block(manifest: dict, block: dict, kinds: list[str], labels_whole: np.ndarray | None, config: GlobalPreprocessConfig,
                  input_dir: Path) -> list[dict]:
    """渲染一个块的输入图并返回该块的复核任务。两类复核共用无标记局部 RGB；semantic 另配局部语义图和 100 m 语义上下文（至少覆盖局部 Patch），
    height 另配整体 RGB 上下文（vlm_review_height_context_m）和局部 DSM 伪彩色图。
    """
    extent, out_px = config.block_size_m, config.vlm_review_image_px
    resolution = manifest['resolution_m_per_px']
    size = (manifest['image']['width'], manifest['image']['height'])
    window = vi.review_window(block, extent, resolution)
    region = vi.valid_crop(window, size, out_px)
    if region is None:
        return []  # 无地图内像元：不生成补黑输入，合并阶段按未复核保留。
    clipped, crop_box = region
    x0, y0, x1, y1 = crop_box
    view = np.s_[y0:y1, x0:x1]
    image_size = (x1-x0, y1-y0)
    span_x, span_y = clipped[2]-clipped[0], clipped[3]-clipped[1]
    sx, sy = image_size[0]/span_x, image_size[1]/span_y
    center = ((block['global_xy'][0]-clipped[0])*sx, (block['global_xy'][1]-clipped[1])*sy)
    geometry = {'image_global_bbox': clipped, 'image_size_px': list(image_size),
                'image_extent_m': [span_x*resolution, span_y*resolution],
                'm_per_image_px': [resolution/sx, resolution/sy],
                'center_image_xy': list(center)}
    edge = next((e for e in manifest['edges'] if str(e['edge_id']) == str(block['edge_id'])), None)
    directory = input_dir/block['block_id']
    from PIL import Image
    # 两类复核共用无标记局部 RGB
    local_rgb_path = vi.save_png(Image.fromarray(vi.read_rgb(manifest, window, out_px)[view]), directory/'local_rgb.png')
    images_by_kind, context, height, semantic_info = {}, None, None, None
    if 'semantic' in kinds:  # 语义复核三图：局部 RGB 与局部语义都不加标记，上下文图叠半透明语义并标出目标框
        if labels_whole is None:
            raise ValueError('语义复核需要整图语义标签')
        visible_labels = vi.crop_labels(labels_whole, size, window, out_px)[view]
        semantic_info = {'class_pct': vi.semantic_stats(visible_labels),
                         'semantic_nodata_pct': round(float(np.mean(visible_labels == bf.IGNORE)*100), 1)}
        context_path, context = render_context(manifest, block, clipped, config, directory/'context_semantic.png',
                                               max(extent, SEMANTIC_CONTEXT_M), labels_whole)
        images_by_kind['semantic'] = [local_rgb_path, vi.save_png(Image.fromarray(vi.semantic_color(visible_labels)),
                                                                  directory/'local_semantic.png'), context_path]
    if 'height' in kinds:  # 高度复核三图：整体 RGB（目标框）+ 局部 RGB + 局部 DSM 伪彩色
        dsm_info = manifest['image'].get('dsm')
        if not dsm_info:
            raise ValueError(f"{block['block_id']} 为高度退化但 manifest 没有 DSM")
        context_path, height_context = render_context(manifest, block, clipped, config, directory/'context_rgb.png',
                                                      config.vlm_review_height_context_m)
        reference = height_reference(manifest, block, config)
        h_ref = reference['h_ref_m']
        dsm = vi.read_dsm(dsm_info, size, window, out_px)[view]
        relative = dsm-h_ref if h_ref is not None else np.full_like(dsm, np.nan)   # 无有效 DSM：出全白图，由 VLM 判数据不可用
        value_range = tuple(float(v) for v in config.vlm_review_dsm_range_m)
        dsm_path = vi.save_png(Image.fromarray(vi.dsm_color(relative, value_range)), directory/'local_dsm.png')
        images_by_kind['height'] = [context_path, local_rgb_path, dsm_path]
        height = {'context': height_context,
                  'dsm_reference': reference,
                  'dsm_color_range_m': list(value_range), 'dsm_color_legend': vi.dsm_legend(value_range),
                  'height_stats': vi.dsm_stats(relative, value_range)}
    jobs = []
    for kind in kinds:
        images = images_by_kind[kind]
        payload = _payload(block, kind, edge, geometry, semantic_info, height)
        if kind == 'semantic':
            payload['context'] = context
        write_json(directory/f'{kind}_input.json', model_input(payload))
        write_json(directory/f'{kind}_metadata.json', payload)
        jobs.append({'block_id': block['block_id'], 'review_type': kind,
                     'images': [relative_path(p) for p in images], 'payload': payload})
    return jobs


def render_context(manifest: dict, block: dict, patch_bbox: list[int], config: GlobalPreprocessConfig, path: Path,
                   extent_m: float, labels_whole: np.ndarray | None = None) -> tuple[Path, dict]:
    """整体上下文：以块中心为中心、边长 extent_m 的窗口（裁去地图外），黄框为局部图实际覆盖范围 patch_bbox（全局像素）。
    给出 labels_whole 时在原始 RGB 上半透明叠加语义颜色（语义复核），否则为原始 RGB（高度复核）。返回 (图像路径, 几何信息)。"""
    from PIL import Image, ImageDraw
    resolution, out_px = manifest['resolution_m_per_px'], config.vlm_review_image_px
    size = (manifest['image']['width'], manifest['image']['height'])
    window = vi.review_window(block, extent_m, resolution)
    clipped, (x0, y0, x1, y1) = vi.valid_crop(window, size, out_px)   # 局部窗口在其中，必有地图内像元
    rgb = vi.read_rgb(manifest, window, out_px)[y0:y1, x0:x1]
    if labels_whole is not None:
        labels = vi.crop_labels(labels_whole, size, window, out_px)[y0:y1, x0:x1]
        color = vi.semantic_color(labels).astype(np.float32)
        valid = (labels != bf.IGNORE)[..., None]
        rgb = np.clip(np.rint(np.where(valid, rgb*CONTEXT_TRANSPARENCY + color*(1-CONTEXT_TRANSPARENCY), rgb)), 0, 255)
    image = Image.fromarray(rgb.astype(np.uint8))
    sx, sy = (x1-x0)/(clipped[2]-clipped[0]), (y1-y0)/(clipped[3]-clipped[1])
    box = [(patch_bbox[0]-clipped[0])*sx, (patch_bbox[1]-clipped[1])*sy, (patch_bbox[2]-clipped[0])*sx, (patch_bbox[3]-clipped[1])*sy]
    ImageDraw.Draw(image).rectangle(box, outline=PATCH_BOX_RGB, width=max(2, out_px//192))
    return vi.save_png(image, path), {'context_global_bbox': clipped, 'context_size_px': [x1-x0, y1-y0],
                                      'context_extent_m': [(clipped[2]-clipped[0])*resolution, (clipped[3]-clipped[1])*resolution],
                                      'context_patch_box_px': [round(v, 1) for v in box]}


def image_names(review_type: str) -> list[str]:
    """Node 7 的双图名称（RGB + 一张专题图）。"""
    return ['原始 RGB 影像', '语义分割图' if review_type == 'semantic' else 'DSM 伪彩色高程图']


def image_labels(review_type: str) -> list[str]:
    """Node 7 每张图前插入的标签（与 image_names 顺序一致）。"""
    return [f'第 {k+1} 张：{name}' for k, name in enumerate(image_names(review_type))]


def review_image_names(review_type: str) -> list[str]:
    """Node 6 的图像名称（两类复核各三图）。"""
    if review_type == 'semantic':
        return ['局部 RGB（local_rgb）', '局部语义分割（local_semantic）', '整体语义上下文（context_semantic）']
    return ['整体 RGB 上下文（context_rgb）', '局部 RGB（local_rgb）', '局部 DSM 伪彩色高程图（local_dsm）']


def review_image_labels(review_type: str) -> list[str]:
    """Node 6 每张图前插入的标签（与 review_image_names 顺序一致）。"""
    return [f'第 {k+1} 张：{name}' for k, name in enumerate(review_image_names(review_type))]


def user_prompt(job: dict, system: dict) -> str:
    description = model_input(job['payload'])['image_description']
    return (f"请独立评估待评估道路区域 {job['block_id']} 的 {job['review_type']} 匹配退化风险。\n\n"
            f"图像说明：\n{description}\n\n"
            f"Skill：\n{system['skill']}\n\nRule：\n{system['rule']}\n\n"
            '按 API 指定的结构化输出格式返回 JSON，先观察后判断；不要添加字段。')


def job_schema(job: dict, system: dict) -> dict:
    from agent.vlm.block_review import bind_block
    return bind_block(system['schema'], job['block_id'])


def derive(output: dict, system: dict | None = None, *, evidence_problem: str | None = None) -> dict:
    """读取模型结论；旧记录沿用 Rule 推导三态结论。"""
    from agent.vlm.block_review import derive_review, load_review_system
    return derive_review(output, (system or load_review_system(output['review_type']))['rules'],
                         evidence_problem=evidence_problem)


def check_output(job: dict, output: dict, schema: dict) -> None:
    """schema 为已绑定 block_id 的请求 schema；生成已受同一 schema 约束，这里防御服务端未按约束输出。"""
    from jsonschema import Draft202012Validator
    Draft202012Validator(schema).validate(output)
    if output['block_id'] != job['block_id']:
        raise ValueError(f"输出 block_id={output['block_id']} 与输入 {job['block_id']} 不一致")

def has_no_topic_data(job: dict) -> bool:
    payload = job['payload']
    if job['review_type'] == 'semantic':
        return payload.get('class_pct') is None if 'class_pct' in payload else payload.get('semantic_nodata_pct') == 100
    stats = payload.get('height_stats') or {}
    return stats['p50_m'] is None if 'p50_m' in stats else stats.get('nodata_pct') == 100


def coverage(payload: dict) -> dict:
    """Node 6 实际送入模型的范围，Node 7 据此核对基准是否完整，不凭配置尺寸推断。"""
    return {key: payload.get(key) for key in ('image_global_bbox', 'image_extent_m')}


def run_jobs(function: Callable[[dict], dict], jobs: list[dict], workers: int, description: str, logger: logging.Logger | None = None) -> list[dict]:
    """并发执行 VLM 任务，进度条按完成数推进并显示 token 速度（Node 6、Node 7 共用）；结果按 jobs 顺序返回。"""
    meter = VLMMeter()
    results: list[dict] = [{}]*len(jobs)
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = {pool.submit(function, job): index for index, job in enumerate(jobs)}
        with tqdm(total=len(jobs), desc=description, unit='job', dynamic_ncols=True) as bar:
            for future in as_completed(futures):
                record = results[futures[future]] = future.result()
                meter.add(record)
                bar.set_postfix(meter.postfix(), refresh=False)
                bar.update()
                if logger is not None:
                    usage = record.get('usage') or {}
                    tokens = usage.get('completion_tokens')
                    seconds = record.get('inference_seconds') or record.get('latency_seconds') or 0.
                    speed = f' 输出 {tokens/seconds:.1f} tok/s' if tokens is not None and seconds > 0 else ''
                    logger.info('VLM 请求 %d/%d %s tokens=%s/%s 推理 %.1fs%s', bar.n, len(jobs), record.get('status', ''),
                                usage.get('prompt_tokens'), tokens, seconds, speed)
    if logger is not None:
        logger.info('%s 汇总：%s', description, meter.summary())
    return results


def failure_fields(exc: Exception) -> dict:
    """失败记录（Node 6、Node 7 共用）；模型已生成的文本（如超出 token 上限时）一并保存，便于排查。"""
    partial = {key: getattr(exc, key) for key in ('raw_text', 'reasoning') if getattr(exc, key, '')}
    return {'status': 'error', 'error': f'{type(exc).__name__}: {exc}', **partial}


def review_one(job: dict, client: Any, systems: dict, config: GlobalPreprocessConfig, result_dir: Path) -> dict:
    from jsonschema import ValidationError
    from agent.vlm.common import VLMError, _generation_schema, _parse
    system = systems[job['review_type']]
    record = {'block_id': job['block_id'], 'review_type': job['review_type'], 'images': job['images'],
              'schema_version': system['rules']['schema_version'], 'coverage': coverage(job['payload']), 'attempts': []}
    schema = job_schema(job, system)
    budget = config.vlm_review_thinking_budget
    effort = config.vlm_review_reasoning_effort
    thinking = {'reasoning_effort': effort} if budget != 0 and effort is not None else {}
    for attempt in range(config.vlm_review_max_retries+1):
        try:
            result = client.infer(user_prompt(job, system), images=[resolve_path(p) for p in job['images']],
                                  system=system['prompt'], max_tokens=config.vlm_review_max_tokens, temperature=0.0,
                                  json_output=True, output_schema=_generation_schema(schema),
                                  image_labels=review_image_labels(job['review_type']),
                                  enable_thinking=budget != 0, thinking_token_budget=budget or None, **thinking)
            record.update({'raw_text': result.text, 'reasoning': getattr(result, 'reasoning', ''), 'usage': result.usage,
                           'latency_seconds': round(result.latency_seconds, 2), 'inference_seconds': round(getattr(result, 'inference_seconds', 0.), 2)})
            output = _parse(result.text)
            check_output(job, output, schema)
            derived = derive(output, system, evidence_problem='topic_no_data' if has_no_topic_data(job) else None)
            record.update({'status': 'ok', 'output': output, 'derived': derived})
            record.pop('error', None)
            break
        except (VLMError, ValidationError, ValueError) as exc:
            record.update(failure_fields(exc))
            record['attempts'].append({'attempt': attempt+1, 'error': record['error']})
        except Exception as exc:  # 单块失败不影响其它块；该块按未复核保留
            record.update(failure_fields(exc))
            record['attempts'].append({'attempt': attempt+1, 'error': record['error']})
            break
    write_json(result_dir/f"{job['block_id']}_{job['review_type']}.json", record)
    return record


def _check(output: dict) -> dict:
    return output.get('degradation_check') or output['height_check']


def decide(block: dict, records: list[dict]) -> dict:
    """按各支路独立的退化证据合并，不删除道路块。"""
    ok = [r for r in records if r['status'] == 'ok']
    fixed = {'recommended_size_m': block['size_m'], 'expand': False, 'review_global_bbox': block['valid_global_bbox']}
    decision = {'reviews': {r['review_type']: {'status': r['status'], 'coverage': r.get('coverage'),
                                               **({'output': r['output'], 'derived': r['derived']} if r['status'] == 'ok'
                                                  else {'error': r['error']})} for r in records}}
    confirmed = {r['review_type']: r['derived']['degraded'] if r['status'] == 'ok' else None for r in records}
    scenes = [r for r in ok if r['derived']['keep']]
    unknown = [kind for kind, value in confirmed.items() if value is None]
    scene = True if scenes else False if confirmed and not unknown else None
    status = 'error' if not ok else 'partial' if len(ok) < len(records) else 'uncertain' if unknown else 'ok'
    return {**decision, **fixed, 'status': status, 'unresolved_review_types': unknown,
            'degradation_confirmed': confirmed, 'degraded_scene': scene,
            'scene_types': [r['review_type'] for r in scenes]}


def scene_record(block: dict) -> dict:
    """退化场景清单中的一条：下游直接使用的位置、范围、类型与 VLM 证据。"""
    review, d = block['vlm_review'], block['features']['degradation']
    reasons = {'semantic': {'road_pct': d.get('semantic_road_pct'), 'plant_pct': d.get('semantic_plant_pct')},
               'height': {'reasons': d.get('height_reasons'), 'flat_pct': d.get('height_flat_pct'),
                          'below_road_pct': d.get('height_below_pct')}}
    outputs = {k: v['output'] for k, v in review['reviews'].items() if v['status'] == 'ok' and k in review['scene_types']}
    return {'block_id': block['block_id'], 'edge_id': block['edge_id'], 'global_xy': block['global_xy'],
            'scene_types': review['scene_types'], 'size_m': review['recommended_size_m'], 'expand': review['expand'],
            'global_bbox': review['review_global_bbox'], 'original_bbox': block['valid_global_bbox'],
            'rule_trigger': {k: reasons[k] for k in review['scene_types']},
            'vlm': {k: {'pattern': review['reviews'][k]['derived']['reason_code'],
                        'degradation_type': review['reviews'][k]['derived']['degradation_type'],
                        'matched_reasons': review['reviews'][k]['derived']['matched_reasons'],
                        'candidate_reasons': review['reviews'][k]['derived']['candidate_reasons'],
                        'consistency_failures': review['reviews'][k]['derived']['consistency_failures'],
                        'evidence': _check(o)['evidence'],
                        **{field: _check(o)[field] for field in ('road_comparisons', 'height_comparisons', 'longitudinal_comparisons', 'verdict') if field in _check(o)},
                        'field_confidence': {f: v for f, v in _check(o).items() if f.endswith('_confidence')},
                        'uncertainty': _check(o)['uncertainty'],
                        'data_confidence': o['data_check']['data_confidence']} for k, o in outputs.items()}}


def render_overview(manifest: dict, config: GlobalPreprocessConfig, output: Path) -> Path:
    """全图复核结果：退化场景（绿框）、不退化（白虚框）、不是道路已删除（红叉）、复核失败（黄框）。"""
    from PIL import ImageDraw
    from ..utils.global_render import read_preview
    preview, scale = read_preview(manifest['image'], config)
    image = preview.convert('RGB')
    draw = ImageDraw.Draw(image)
    side = max(image.size)
    width = max(2, side//700)
    font = vi._font(max(12, side//160))
    box = lambda b: [v*scale for v in b]
    for edge in manifest['edges']:
        draw.line([(x*scale, y*scale) for x, y in edge['polyline_xy']], fill=(255, 255, 255), width=width)
    counts = {'scene': 0, 'rejected': 0, 'error': 0}
    label = lambda b, color: draw.text(((b[0]+b[2])/2, (b[1]+b[3])/2), block['block_id'], fill=color, font=font,
                                       anchor='mm', stroke_width=2, stroke_fill='black')
    for block in manifest['blocks']:
        review = block.get('vlm_review')
        rect = box(block['valid_global_bbox'])
        if not review:
            draw.rectangle(rect, outline=(160, 160, 160), width=1)
            continue
        if review['degraded_scene'] is None:
            counts['error'] += 1
            draw.rectangle(rect, outline=(255, 220, 0), width=2*width)
        elif review['degraded_scene']:
            counts['scene'] += 1
            draw.rectangle(rect, outline=(40, 220, 60), width=2*width)
        else:
            counts['rejected'] += 1
            vi._dashed_rect(draw, rect, (255, 255, 255), 2*width, max(6, side//200))
        label(rect, 'white')
    items = [((40, 220, 60), f"degraded scene: {counts['scene']}"),
             ((255, 255, 255), f"not degraded (dashed): {counts['rejected']}"),
             ((255, 220, 0), f"unverified: {counts['error']}")]
    output.parent.mkdir(parents=True, exist_ok=True)
    vi.add_legend(image, items, 'VLM verification of degraded blocks').save(output)
    return output


def run_vlm_review(manifest_path: str | Path, config: GlobalPreprocessConfig, *, client: Any = None,
                   semantic_segmenter: Any = None, block_ids: list[str] | None = None, dry_run: bool | None = None,
                   logger: logging.Logger | None = None) -> dict:
    """独立复核 Node 5 命中的候选支路，全量运行写回 manifest，预览和选块写入诊断目录。"""
    from agent.vlm.block_review import load_review_system
    logger = logger or logging.getLogger('map_preprocess.vlm_review')
    dry_run = config.vlm_review_dry_run if dry_run is None else dry_run
    manifest_path = resolve_path(manifest_path)
    backup = backup_path(manifest_path)
    # 重跑从复核前的 manifest 开始，被删除的块重新参与复核
    manifest = load_manifest(backup if backup.is_file() else manifest_path)
    if not manifest.get('extensions', {}).get('block_features'):
        raise ValueError('需要先运行 Node 5（block_features）')
    task_dir = layout.task_dir_of_manifest(manifest_path)
    systems = {kind: load_review_system(kind) for kind in ('semantic', 'height')}
    selected = [(block, kinds) for block in manifest['blocks'] if (kinds := review_types(block))]
    if block_ids:
        wanted = set(block_ids)
        selected = [(b, k) for b, k in selected if b['block_id'] in wanted]
    diagnostic = dry_run or block_ids is not None
    if diagnostic:
        input_dir, result_dir = layout.diagnostic_dirs(task_dir, 'node6', dry_run=dry_run)
    else:
        input_dir = layout.figure(task_dir, 'vlm_review_inputs')
        result_dir = layout.state_dir(task_dir)/'vlm_review'
    summary_path = (result_dir if diagnostic else layout.state_dir(task_dir))/'vlm_review.json'
    overview_path = input_dir/'overview.png' if diagnostic else layout.figure(task_dir, 'vlm_review')
    scenes_path = summary_path.with_name('degraded_scenes.json')
    outputs = (input_dir, result_dir, summary_path, scenes_path, overview_path, backup)
    publication = nullcontext() if diagnostic else protect_stage_outputs(manifest_path, outputs, keep=(backup,))
    with publication:
        for directory in (input_dir, result_dir):
            if directory.exists():
                shutil.rmtree(directory)
        jobs = []
        labels = (_semantic_labels(manifest, config, semantic_segmenter, logger)
                  if any('semantic' in kinds for _, kinds in selected) else None)
        for block, kinds in selected:
            jobs += prepare_block(manifest, block, kinds, labels, config, input_dir)
        logger.info('Node 6：%d 个待评估道路块，%d 个复核任务（semantic %d / height %d）', len(selected), len(jobs),
                    sum(j['review_type'] == 'semantic' for j in jobs), sum(j['review_type'] == 'height' for j in jobs))
        versions = {kind: system['rules']['schema_version'] for kind, system in systems.items()}
        summary = {'status': 'dry_run' if dry_run else 'completed',
                   'schema_version': max(versions.values()), 'schema_versions': versions,
                   'prompt_systems': {k: relative_path(v['directory']) for k, v in systems.items()},
                   'job_count': len(jobs), 'jobs_by_type': {k: sum(j['review_type'] == k for j in jobs) for k in systems},
                   'inputs_dir': relative_path(input_dir), 'results_dir': relative_path(result_dir), 'path': relative_path(summary_path)}
        if dry_run:
            for job in jobs:
                path = result_dir/f"{job['block_id']}_{job['review_type']}_prompt.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                images = ''.join(f'{label}\n<image {p}>\n' for label, p in zip(review_image_labels(job['review_type']), job['images']))
                path.write_text(f"[system]\n{systems[job['review_type']]['prompt']}\n\n[user]\n{images}"
                                f"{user_prompt(job, systems[job['review_type']])}\n", encoding='utf-8')
            write_json(summary_path, summary)
            return summary

        records: list[dict] = []
        if jobs:
            owned = client is None
            if owned:
                from agent.vlm.client import VLMClient
                client = VLMClient()
            try:
                records = run_jobs(lambda job: review_one(job, client, systems, config, result_dir), jobs,
                                   config.vlm_review_workers, 'Node 6 VLM 复核', logger)
            finally:
                if owned:
                    client.close()
        if not diagnostic and not backup.is_file():
            shutil.copyfile(manifest_path, backup)
        by_block: dict[str, list[dict]] = {}
        for record in records:
            by_block.setdefault(record['block_id'], []).append(record)
        for block, _ in selected:
            decision = decide(block, by_block.get(block['block_id'], []))
            decision['review_global_bbox'] = [int(v) for v in decision['review_global_bbox']]
            block['vlm_review'] = decision
        overview = render_overview(manifest, config, overview_path)
        reviewed = [b for b in manifest['blocks'] if b.get('vlm_review')]
        scenes = [scene_record(b) for b in reviewed if b['vlm_review']['degraded_scene']]
        summary.update({
            'reviewed_blocks': len(selected),
            'degraded_scene_count': len(scenes),
            'degraded_scene_ids': [s['block_id'] for s in scenes],
            'rejected_block_ids': sorted(b['block_id'] for b in reviewed if b['vlm_review']['degraded_scene'] is False),
            'removed_block_ids': [],
            'discarded_insufficient_evidence_ids': sorted(
                b['block_id'] for b in reviewed
                if b['vlm_review']['degraded_scene'] is None
                and any(r['status'] == 'ok' and r['derived']['degraded'] is None
                        for r in b['vlm_review']['reviews'].values())),
            'unverified_block_ids': sorted(b['block_id'] for b in reviewed if b['vlm_review']['degraded_scene'] is None),
            'unresolved_branch_block_ids': sorted(b['block_id'] for b in reviewed if b['vlm_review'].get('unresolved_review_types')),
            'errors': [{'block_id': r['block_id'], 'review_type': r['review_type'], 'error': r['error']}
                       for r in records if r['status'] == 'error'],
            'block_count_before': len(manifest['blocks']), 'block_count_after': len(manifest['blocks']),
            'degraded_scenes': relative_path(scenes_path), 'overview': relative_path(overview),
            'note': '复核前 manifest 备份为 state/' + BACKUP_NAME})
        write_json(scenes_path, {'task_id': manifest.get('task_id'), 'image': manifest['image']['path'],
                                 'resolution_m_per_px': manifest['resolution_m_per_px'],
                                 'count': len(scenes),
                                 'by_type': {k: sum(k in s['scene_types'] for s in scenes) for k in ('semantic', 'height')},
                                 'scenes': scenes})
        write_json(summary_path, summary)
        if diagnostic:
            return summary
        manifest['files']['degraded_scenes'] = relative_path(scenes_path)
        manifest.setdefault('extensions', {})['vlm_review'] = summary
        manifest['files']['vlm_review'] = relative_path(summary_path)
        manifest['files']['vlm_review_overview'] = relative_path(overview)
        publish_stage(manifest_path, manifest, 'vlm_review')
        logger.info('Node 6 完成：退化场景 %d，不退化 %d，失败 %d', len(scenes),
                    len(summary['rejected_block_ids']), len(summary['errors']))
        return summary


def vlm_review_blocks(state: PreprocessState, *, config: GlobalPreprocessConfig, client: Any = None,
                      semantic_segmenter: Any = None) -> dict:
    logger = logging.getLogger(f"map_preprocess.{state['task_id']}")
    summary = run_vlm_review(state['manifest_path'], config, client=client, semantic_segmenter=semantic_segmenter,
                             logger=logger)
    return {'vlm_review': summary}
