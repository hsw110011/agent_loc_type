"""Node 8：退化区域空间融合（fuse_degraded_patches）。

读取 Node 7 的 degradation_extents.json，把每条已完成支路在最大确认尺寸下的窗口作为一个 Patch，
按原始 Patch 的正面积重叠融合（几何规则见 utils/patch_fusion.py）。只做确定性几何计算，不调用 VLM，
不改块、Node 6 和 Node 7 的结论；结果写入 state/degradation_regions.json 与 extensions.degradation_fusion。
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..global_config import GlobalPreprocessConfig, PIPELINE_VERSION
from ..io import load_manifest, read_json, relative_path, resolve_path, write_json
from .. import output_layout as layout
from ..state import PreprocessState
from ..utils import vlm_review_inputs as vi
from ..utils.patch_fusion import MIXED_TYPE, fuse_patches

# Rule 的中文结论 → 融合使用的类别；未列出的值按 unknown 处理，不进入融合
DEGRADATION_LABELS = {'退化': 'degradation', '纵向退化': 'longitudinal_degradation', '不退化': 'non_degraded', '未知': 'unknown'}
REGION_COLORS = {'degradation': (255, 70, 40), 'longitudinal_degradation': (40, 140, 255), MIXED_TYPE: (210, 60, 220)}
PATCH_COLOR, BBOX_COLOR = (0, 230, 255), (255, 220, 0)


def patch_id_of(extent: dict) -> str:
    return f"{extent['block_id']}_{extent['review_type']}"


def patch_from_extent(extent: dict, resolution: float) -> dict:
    """窗口与 Node 7 判断时的档位窗口相同；类别取最大确认尺寸那一档的结论，各档结论可能不同。
    基准范围不完整（stop_reason=base_incomplete）时用 Node 6 实际判断的范围，不声明完整基准窗口。"""
    size = extent['max_size_m']
    tier = next((item for item in extent['tiers'] if item['size_m'] == size), {})
    bbox = tier.get('bbox') or vi.review_window(extent, size, resolution)
    if extent.get('stop_reason') == 'base_incomplete':
        bbox = ((extent.get('node6') or {}).get('coverage') or {}).get('image_global_bbox') or bbox
    return {'patch_id': patch_id_of(extent), 'block_id': extent['block_id'], 'review_type': extent['review_type'], 'size_m': size,
            'bbox': bbox, 'degradation_type': DEGRADATION_LABELS.get(tier.get('degradation_type'), 'unknown')}


def patches_from_extents(extents: list[dict], resolution: float) -> tuple[list[dict], list[dict], list[dict]]:
    """返回 (Patch, 已确认退化却没有确认尺寸的支路, Node 6 未确认退化的支路)；最后一类本来就不参与融合，不算问题。"""
    patches, excluded, not_degraded = [], [], []
    for extent in extents:
        if extent['status'] == 'completed' and extent['max_size_m'] is not None:
            patches.append(patch_from_extent(extent, resolution))
        elif extent.get('skip_reason') == 'node6_not_confirmed':
            not_degraded.append({'patch_id': patch_id_of(extent), 'reason': 'node6_not_confirmed'})
        else:
            excluded.append({'patch_id': patch_id_of(extent), 'reason': extent.get('skip_reason') or 'no_confirmed_size'})
    return patches, excluded, not_degraded


def union_rings(region: dict) -> list:
    """union_polygon 的多边形列表，每项为 [外环, 洞…]；Node 9 共用。"""
    geometry = region['union_polygon']
    return geometry['coordinates'] if geometry['type'] == 'MultiPolygon' else [geometry['coordinates']]


def render_overview(manifest: dict, config: GlobalPreprocessConfig, regions: list[dict], output: Path) -> Path:
    """全图融合结果：并集按类型半透明填充并描边（洞不填充），青框为原始 Patch，黄框为仅用于索引的外接矩形。"""
    from PIL import Image, ImageDraw
    from ..utils.global_render import read_preview
    preview, scale = read_preview(manifest['image'], config)
    image = preview.convert('RGB')
    side = max(image.size)
    width = max(2, side//700)
    font = vi._font(max(12, side//160))

    def scaled(ring: list) -> list[tuple[float, float]]:
        return [(x*scale, y*scale) for x, y in ring]

    for region in regions:
        mask = Image.new('L', image.size, 0)
        painter = ImageDraw.Draw(mask)
        for exterior, *holes in union_rings(region):
            painter.polygon(scaled(exterior), fill=110)
            for hole in holes:
                painter.polygon(scaled(hole), fill=0)
        image.paste(REGION_COLORS[region['degradation_type']], (0, 0), mask)
    draw = ImageDraw.Draw(image)
    for edge in manifest['edges']:
        draw.line(scaled(edge['polyline_xy']), fill=(255, 255, 255), width=1)
    for region in regions:
        for patch in region['source_patches']:
            draw.rectangle([value*scale for value in patch['bbox']], outline=PATCH_COLOR, width=1)
        draw.rectangle([value*scale for value in region['bbox']], outline=BBOX_COLOR, width=width)
        for rings in union_rings(region):
            for ring in rings:
                draw.line(scaled(ring), fill=REGION_COLORS[region['degradation_type']], width=2*width, joint='curve')
        draw.text((region['bbox'][0]*scale+3, region['bbox'][1]*scale+1), f"{region['merged_id']} n={region['source_patch_count']}",
                  fill='white', font=font, stroke_width=2, stroke_fill='black')
    kinds = sorted({region['degradation_type'] for region in regions})
    items = [*((REGION_COLORS[kind], kind) for kind in kinds), (PATCH_COLOR, 'source patch'), (BBOX_COLOR, 'bbox (index only)')]
    vi.add_legend(image, items, 'Fused degraded regions (fill = union of confirmed windows)').save(output)
    return output


def run_degradation_fusion(manifest_path: str | Path, config: GlobalPreprocessConfig, *, logger: logging.Logger | None = None) -> dict:
    """融合 Node 7 的退化 Patch；返回摘要，完整结果写入 state/degradation_regions.json。"""
    logger = logger or logging.getLogger('map_preprocess.degradation_fusion')
    manifest_path = resolve_path(manifest_path)
    manifest = load_manifest(manifest_path)
    extents_path = manifest['files'].get('degradation_extents')
    if not extents_path:
        raise ValueError('需要先运行 Node 7（degradation_extents.json）')
    document = read_json(resolve_path(extents_path))
    resolution = document['resolution_m_per_px']
    patches, skipped, not_degraded = patches_from_extents(document['extents'], resolution)
    result = fuse_patches(patches, allowed_types=config.degradation_fusion_types, cross_type=config.degradation_fusion_cross_type)
    excluded, rejected, regions = skipped+result['excluded'], result['rejected'], result['regions']
    for item in rejected:
        logger.warning('Node 8：丢弃无效 Patch %s（%s）', item.get('patch_id'), item['reason'])
    task_dir = layout.task_dir_of_manifest(manifest_path)
    output = layout.state_dir(task_dir)/'degradation_regions.json'
    from ..utils.result_publication import protect_stage_outputs
    overview_path = layout.figure(task_dir, 'degradation_fusion')
    with protect_stage_outputs(manifest_path, (output, overview_path)):
        overview = render_overview(manifest, config, regions, overview_path)
        write_json(output, {'pipeline_version': PIPELINE_VERSION, 'task_id': manifest.get('task_id'), 'image': manifest['image']['path'], 'resolution_m_per_px': resolution,
                            'coordinate_system': 'global_pixel', 'overlap_rule': 'strict_positive_area',
                            'fusion_types': list(config.degradation_fusion_types), 'cross_type': config.degradation_fusion_cross_type,
                            'meaning': 'union_polygon 是已确认窗口的并集；bbox 仅用于裁剪、索引和可视化，其中超出并集的部分未经验证',
                            'count': len(regions), 'regions': regions, 'excluded': excluded, 'rejected': rejected,
                            'not_degraded': not_degraded})
        summary = {'status': 'completed', 'pipeline_version': PIPELINE_VERSION, 'patch_count': sum(region['source_patch_count'] for region in regions),
                   'region_count': len(regions), 'merged_region_count': sum(region['source_patch_count'] > 1 for region in regions),
                   'excluded_count': len(excluded), 'excluded_patch_ids': [item['patch_id'] for item in excluded],
                   'not_degraded_count': len(not_degraded), 'rejected_count': len(rejected), 'degradation_regions': relative_path(output),
                   'overview': relative_path(overview)}
        manifest.setdefault('extensions', {})['degradation_fusion'] = summary
        manifest['files'].update({'degradation_regions': relative_path(output), 'degradation_fusion_overview': relative_path(overview)})
        from ..utils.result_publication import publish_stage
        publish_stage(manifest_path, manifest, 'degradation_fusion')
    logger.info('Node 8 完成：%d 个 Patch → %d 个区域（%d 个由多个 Patch 融合），未进入融合 %d，Node 6 不退化 %d，无效 %d',
                summary['patch_count'], len(regions), summary['merged_region_count'], len(excluded), len(not_degraded), len(rejected))
    return summary


def fuse_degraded_patches(state: PreprocessState, *, config: GlobalPreprocessConfig) -> dict:
    expansion = (state.get('vlm_expansion') or {}).get('status')
    if expansion in ('skipped', 'dry_run'):  # Node 7 没有产出 degradation_extents.json
        return {'degradation_fusion': {'status': 'skipped', 'reason': f'node7_{expansion}'}}
    summary = run_degradation_fusion(state['manifest_path'], config, logger=logging.getLogger(f"map_preprocess.{state['task_id']}"))
    return {'degradation_fusion': summary}
