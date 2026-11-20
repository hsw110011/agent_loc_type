"""Node 5：对 manifest 每块的图内部分计算语义和可选 DSM 指标，记录可送审的数据支路。

块超出原图的部分直接裁掉（读取 valid_global_bbox），不补边，因此贴边块是长方形。
道路拓扑和块只来自 RGB；DSM 按块的地理范围裁剪（见 utils/dsm.py），不参与拓扑。

- 语义：生成或复用整图语义后裁块（semantic_roads.whole_image_semantic，同一缓存），不逐块推理；语义退化 = 道路 < 15% 且植被 > 65%。
- 高程：manifest image.dsm 存在时，按 DSM 相对块中心路面的高度（米）分段；最低段（默认 0-0.5m）> 90% 或低于路面像元 > 70%
  即高度退化（>2m 占比大不算）。
  没有 DSM 时不判高度（height_* 为 null），只判语义。
阈值初筛为 Node 6 选择对应支路的候选块。context 不参与本节点的指标。
"""
from __future__ import annotations

import gc
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..global_config import GlobalPreprocessConfig, PIPELINE_VERSION
from ..io import load_manifest, open_image, relative_path, resolve_path, write_json
from .. import output_layout as layout
from ..state import PreprocessState
from ..utils import block_features as bf
from ..utils.block_feature_vis import save_block_visuals
from ..utils.dsm import downsample_labels, inspect_dsm, read_block_dsm

def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    return value


def _read_block(manifest: dict, block: dict) -> tuple[np.ndarray, np.ndarray]:
    """读取块在原图内的部分（valid_global_bbox），超出原图的部分直接裁掉；
    有效掩码只排除源图自身的无数据像元（GeoTIFF mask）。"""
    x0, y0, x1, y1 = bbox = block['valid_global_bbox']
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"块 {block['block_id']} 与原图无交集")
    with open_image(manifest['image']['path'], manifest['config']['pillow_max_image_pixels']) as source:
        return source.read(bbox), source.valid_mask(bbox)


def _model_fingerprint(tool: Any, settings: Any) -> dict | None:
    """语义模型缓存身份；注入实现须提供字符串 cache_key，否则不缓存。"""
    if tool is not None:
        key = getattr(tool, 'cache_key', None)
        return {'injected': key} if isinstance(key, str) else None
    from percection_tool.segearth_ov3.inference import cache_identity
    return cache_identity(settings)


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def semantic_crop(labels: np.ndarray, confidence: np.ndarray, image_size: tuple[int, int], bbox: list[int]) -> dict[str, np.ndarray]:
    """从整图语义（工作网格）裁出块在原图内的范围 valid_global_bbox，最近邻缩放到块的原图像素尺寸。
    工作网格与原图同分辨率（默认 0.05 m）时就是精确裁剪。"""
    import cv2
    width, height = image_size
    x0, y0, x1, y1 = bbox
    sx, sy = labels.shape[1]/width, labels.shape[0]/height
    a, b = int(math.floor(x0*sx)), int(math.floor(y0*sy))
    c, d = max(a+1, int(math.ceil(x1*sx))), max(b+1, int(math.ceil(y1*sy)))
    mask, score = labels[b:d, a:c], confidence[b:d, a:c]
    if mask.shape != (y1-y0, x1-x0):
        mask = cv2.resize(np.ascontiguousarray(mask), (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST)
        score = cv2.resize(np.ascontiguousarray(score, dtype=np.float32), (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST)
    return {'mask': np.ascontiguousarray(mask), 'confidence': np.asarray(score, np.float32)}


def _degradation(features: dict, flat_min: float = 90., below_max: float = 70.,
                 max_road: float = GlobalPreprocessConfig.feature_semantic_max_road_pct, min_plant: float = 65., has_dsm: bool = False, *, judge_semantic: bool | None = None,
                 min_clutter: float = GlobalPreprocessConfig.feature_semantic_min_clutter_pct) -> dict:
    """语义退化（启用语义分支时判定）：道路 < max_road 且（植被 > min_plant 或 植被+建筑 > min_clutter）。
    默认有 DSM 时只判高度；显式双分支时同时判定。
    高度退化（有 DSM 时），任一满足即退化：最低段（默认 0-0.5m，含缓坡低于）> flat_min（几乎全和路面一样平）；
    阶跃低于路面的像元 step_below_pct > below_max（路堤/高架等，四周隔着台阶比路低）。高段（>2m）占比大不算退化。"""
    pct = features['class_pct']
    if judge_semantic is None:
        judge_semantic = not has_dsm
    semantic = None if not pct or not judge_semantic else bool(
        pct['road'] < max_road and (
            pct['plant'] > min_plant
            or (pct['building'] > 25. and pct['plant'] > 30. and pct['plant']+pct['building'] > min_clutter)))
    bands, stats = features.get('height_band_pct'), features.get('height_stats')
    flat_pct = next(iter(bands.values())) if bands else None
    # 低于路面只看阶跃低于（隔着台阶的低处）；缓坡低于已计入最低段
    below_pct = (stats.get('step_below_pct', stats.get('below_base_pct')) if stats else None)
    reasons = []
    if flat_pct is not None and flat_pct > flat_min:
        reasons.append('flat')
    if below_pct is not None and below_pct > below_max:
        reasons.append('below_road')
    height = None if flat_pct is None and below_pct is None else bool(reasons)
    return {'semantic_degraded': semantic, 'semantic_road_pct': pct['road'] if pct else None,
            'semantic_plant_pct': pct['plant'] if pct else None,
            'height_degraded': height, 'height_reasons': reasons, 'height_flat_pct': flat_pct,
            'height_below_pct': below_pct, 'degraded': bool(semantic or height)}


def _block_features(block: dict, semantic: dict, dsm: np.ndarray | None, config: GlobalPreprocessConfig,
                    rgb: np.ndarray | None = None, vis_dir: Path | None = None,
                    dsm_m_per_px: float = 1., manifest_resolution: float = 1.,
                    cleaned_road: np.ndarray | None = None) -> tuple[dict, dict | None]:
    """单块全部指标（只看整块，不与邻块或全图比较）及可选可视化路径；dsm 为 None 时只算语义。"""
    confidence = semantic['confidence'].astype(np.float32)
    labels = bf.semantic_labels(semantic['mask'], confidence, config.feature_min_class_confidence)
    height = None
    if dsm is not None:
        # 块中心（道路上的布块点）在 DSM 块网格中的行列：网格原点 valid_global_bbox 左上，像元 = factor 个 RGB 像元
        x0, y0 = block['valid_global_bbox'][:2]
        factor = dsm_m_per_px/manifest_resolution
        gx, gy = block['global_xy']
        center = ((gy-y0)/factor-.5, (gx-x0)/factor-.5)
        if cleaned_road is not None:
            cleaned_road = downsample_labels(cleaned_road.astype(np.uint8), dsm.shape).astype(bool)
        height = bf.dsm_height_features(dsm, downsample_labels(labels, dsm.shape), config, dsm_m_per_px, center, cleaned_road)
    features = {'class_pct': bf.class_pct(labels),
                'height_band_pct': height['height_band_pct'] if height else None,
                'height_stats': height['height_stats'] if height else None,
                'road_reference': height['road_reference'] if height else None}
    features['degradation'] = _degradation(features, config.feature_height_flat_min_pct, config.feature_height_below_max_pct,
                                           config.feature_semantic_max_road_pct, config.feature_semantic_min_plant_pct,
                                           has_dsm=dsm is not None,
                                           judge_semantic=True if config.analysis == "both" else None,
                                           min_clutter=config.feature_semantic_min_clutter_pct)
    paths = None
    if vis_dir is not None and rgb is not None:
        paths = save_block_visuals(vis_dir/block['block_id'], rgb, labels, height['_relative'] if height else None,
                                   tuple(config.feature_dsm_height_bands_m), config.feature_height_vis_max_m,
                                   config.feature_vis_max_side, raw_dsm=dsm)
    return _round(features), paths


def run_block_features(manifest_path: str | Path, config: GlobalPreprocessConfig, *,
                       semantic_segmenter: Any = None, dsm_path: str | Path | None = None,
                       logger: logging.Logger | None = None) -> dict:
    """计算并写回 manifest 的 features；返回轻量摘要（同时保存为 block_features.json）。

    dsm_path 给定时重新检查对齐并覆盖 manifest image.dsm；否则沿用 Node 1 记录的 DSM（没有则只判语义）。
    """
    logger = logger or logging.getLogger('map_preprocess.block_features')
    # 重算特征前恢复 Node 6 删除块之前的 manifest（若有），避免在已删块的 manifest 上计算
    from .vlm_review import backup_path
    backup = backup_path(manifest_path)
    manifest = load_manifest(backup if backup.is_file() else manifest_path)
    if manifest.get('workflow') != 'global_blocks':
        raise ValueError('Node 5 只支持 global_blocks manifest')
    from agent.perception_config import load_perception_config
    perception = load_perception_config(config.perception_config)
    blocks, edges = manifest['blocks'], manifest['edges']
    # 下游 Node 6/7 单独运行时读 manifest 记录的 feature_*，必须与本次计算一致
    manifest['config'].update({k: v for k, v in config.to_dict().items() if k.startswith('feature_')})
    task_dir = layout.task_dir_of_manifest(manifest_path)
    manifest['config']['analysis'] = config.analysis
    if dsm_path is not None and config.analysis != 'semantic':
        manifest['image']['dsm'] = inspect_dsm(dsm_path, manifest['image']['path'], manifest['resolution_m_per_px'])
    dsm = manifest['image'].get('dsm') if config.analysis != 'semantic' else None
    if config.analysis in ('height', 'both') and dsm is None:
        raise ValueError(f'--analysis {config.analysis} 需要 DSM，请提供 --dsm 或使用已有 DSM 的任务')
    analyze_semantic = dsm is None or config.analysis == 'both'
    mask_path = manifest['files'].get('cleaned_road_reference_mask')
    cleaned_road = np.load(resolve_path(mask_path), mmap_mode='r', allow_pickle=False) if dsm and mask_path else None
    if cleaned_road is not None and cleaned_road.shape != (manifest['image']['height'], manifest['image']['width']):
        raise ValueError('清洗道路掩码尺寸与原图不一致')
    logger.info('Node 5 高程来源：%s', f"DSM {dsm['path']}" if dsm else '无 DSM，只判语义')
    # 语义直接生成或复用整图语义后裁块（同一模型、同一缓存），不再逐块重新推理。
    from .semantic_roads import whole_image_semantic
    semantic_info = {'model': None, 'cache_hit': False, 'working_size': None}
    if blocks:
        whole = {'image': manifest['image'], 'resolution_m_per_px': manifest['resolution_m_per_px'],
                 'input_sha256': manifest['input_sha256'], 'image_path': manifest['image']['path']}
        whole_labels, whole_confidence, semantic_info = whole_image_semantic(whole, config, semantic_segmenter, logger)
    image_size = (manifest['image']['width'], manifest['image']['height'])

    vis_dir = layout.figure(task_dir, 'block_images') if config.feature_vis_enabled else None
    from ..utils.result_publication import protect_stage_outputs
    paths = (layout.state_dir(task_dir)/'block_features.json', backup,
             *(layout.figure(task_dir, key) for key in ('block_images', 'blocks_degraded', 'degraded_sheets')))
    with protect_stage_outputs(resolve_path(manifest_path), paths, keep=(backup,)):
        unavailable = {}
        no_road_reference = []
        invalid_road_reference = []
        height_extra_candidates = []
        for block in blocks:
            identifier = block['block_id']
            semantic = semantic_crop(whole_labels, whole_confidence, image_size, block['valid_global_bbox'])
            rgb = _read_block(manifest, block)[0] if vis_dir is not None else None
            block_dsm = read_block_dsm(dsm, block['valid_global_bbox']) if dsm else None
            grid_m = manifest['resolution_m_per_px']*dsm['block_grid_factor'] if dsm else 1.
            x0, y0, x1, y1 = block['valid_global_bbox']
            cleaned_crop = cleaned_road[y0:y1, x0:x1] if cleaned_road is not None else None
            block['features'], paths = _block_features(block, semantic, block_dsm, config, rgb, vis_dir, grid_m,
                                                       manifest['resolution_m_per_px'], cleaned_crop)
            block['feature_vis'] = {name: relative_path(path) for name, path in paths.items()} if paths else None
            available, missing = [], []
            if block['features']['class_pct'] is not None:
                available.append('semantic')
            elif analyze_semantic:
                missing.append('semantic_no_data')
            if block_dsm is not None:
                if np.isfinite(block_dsm).any():
                    available.append('height')
                else:
                    missing.append('dsm_no_data')
                quality = block['features']['road_reference']
                if not quality['valid']:
                    invalid_road_reference.append(identifier)
                    if 'height' in available and all(attempt['sample_count'] == 0 for attempt in quality['attempts']):
                        no_road_reference.append(identifier)
                stats = block['features']['height_stats'] or {}
                if 'height' in available and (not quality['valid'] or stats.get('center_step_above')):
                    height_extra_candidates.append(identifier)
            block['review_types_available'] = available
            if missing:
                unavailable[identifier] = missing

        overview = None
        if vis_dir is not None:
            from ..utils.block_feature_vis import render_degradation_overview
            from ..utils.global_render import read_preview
            preview, scale = read_preview(manifest['image'], config)
            overview = render_degradation_overview(preview, scale, blocks, edges, layout.figure(task_dir, 'blocks_degraded'),
                title=("degraded blocks (semantic + height)" if config.analysis == "both" else
                       f"degraded blocks (height: {bf.band_labels(tuple(config.feature_dsm_height_bands_m))[0]} > {config.feature_height_flat_min_pct:g}% "
                       f"or below road > {config.feature_height_below_max_pct:g}%)" if dsm else
                       f"degraded blocks (no DSM, semantic only: road < {config.feature_semantic_max_road_pct:g}% & plant > "
                       f"{config.feature_semantic_min_plant_pct:g}%)"))
        degraded = lambda key: sorted(b['block_id'] for b in blocks if b['features']['degradation'][key])
        summary = {'status': 'completed', 'pipeline_version': PIPELINE_VERSION, 'implemented': True, 'block_count': len(blocks),
                   'input_region': 'block valid_global_bbox (block cropped to image; no padding); context not used',
                   'models': {'semantic': semantic_info['model']}, 'semantic_source': 'whole_image',
                   'semantic_cache_hit': semantic_info['cache_hit'], 'semantic_working_size': semantic_info['working_size'],
                   'vis_dir': relative_path(vis_dir) if vis_dir is not None else None,
                   'degradation_overview': relative_path(overview) if overview is not None else None,
                   'degraded_sheets': [],
                   'semantic_degraded_blocks': degraded('semantic_degraded'),
                   'height_degraded_blocks': degraded('height_degraded'),
                   # 阈值命中及无有效路面高程基准的块进入对应 VLM 支路。
                   'degraded_blocks': degraded('degraded'),
                   'review_candidate_block_ids': sorted(set(degraded('degraded')) | set(height_extra_candidates)),
                   'invalid_road_reference_block_ids': sorted(invalid_road_reference),
                   'unavailable_review_data': unavailable,
                   'missing_road_reference_block_ids': sorted(no_road_reference),
                   'height_flat_min_pct': config.feature_height_flat_min_pct,
                   'height_below_max_pct': config.feature_height_below_max_pct,
                   'semantic_max_road_pct': config.feature_semantic_max_road_pct,
                   'semantic_min_plant_pct': config.feature_semantic_min_plant_pct,
                   'analysis': config.analysis,
                   'analysis_branches': [name for name, enabled in (('semantic', analyze_semantic), ('height', dsm is not None)) if enabled],
                   'semantic_judged': analyze_semantic,
                   # 高程来源：'dsm'（绝对高度，米）或 None（没有 DSM，height_* 全为 null，只判语义）。
                   'height_source': 'dsm' if dsm else None, 'dsm': dsm,
                   'height_bands_m': list(config.feature_dsm_height_bands_m),
                   'height_vis_max_m': config.feature_height_vis_max_m,
                   'perception_config': asdict(perception),
                   'settings': {k: v for k, v in config.to_dict().items() if k.startswith('feature_')}}
        if vis_dir is not None:
            from ..feature_contact_sheet import degraded_blocks, write_sheets
            manifest.setdefault('extensions', {})['block_features'] = summary
            sheet_dir = layout.figure(task_dir, 'degraded_sheets')
            sheets = write_sheets(manifest, degraded_blocks(manifest), sheet_dir, 'degraded',
                                  title=f"退化块汇总 — {Path(manifest['image']['path']).stem}")
            summary['degraded_sheets'] = [relative_path(p) for p in sheets]

        summary_path = layout.state_dir(task_dir)/'block_features.json'
        summary['path'] = relative_path(summary_path)
        write_json(summary_path, summary)
        manifest.setdefault('extensions', {})['block_features'] = {k: v for k, v in summary.items()
                                                                   if k not in ('perception_config', 'settings')}
        manifest['files']['block_features'] = summary['path']
        if overview is not None:
            manifest['files']['blocks_degraded'] = relative_path(overview)
            manifest['files']['block_images'] = relative_path(vis_dir)
            manifest['files']['degraded_sheets'] = relative_path(sheet_dir)
        else:
            for key in ('degraded_sheets', 'blocks_degraded', 'block_images'):
                manifest['files'].pop(key, None)
        from ..utils.result_publication import publish_stage
        publish_stage(manifest_path, manifest, 'block_features')
        backup.unlink(missing_ok=True)
    logger.info('Node 5 完成：%d 块，整图语义缓存命中=%s', len(blocks), semantic_info['cache_hit'])
    return manifest['extensions']['block_features']


def compute_block_features(state: PreprocessState, *, config: GlobalPreprocessConfig,
                           semantic_segmenter: Any = None) -> dict:
    summary = run_block_features(state['manifest_path'], config, semantic_segmenter=semantic_segmenter,
        logger=logging.getLogger(f"map_preprocess.{state['task_id']}"))
    return {'block_features': summary}
