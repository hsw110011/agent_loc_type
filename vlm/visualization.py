"""Node 4：保存候选框图、原图 ROI 和逐区域输入包；不调用 VLM。"""
from __future__ import annotations

import copy
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageFont

from .candidates import validate_candidates, load_candidate_config
from .common import PositioningError, _digest, record_trace, validate_region, region_vertices, region_crop_box, write_json, resolve_path, relative_paths

PALETTE = [(255, 80, 70), (0, 220, 255), (255, 215, 0),
           (200, 100, 255), (60, 240, 140), (255, 140, 40)]


def _draw_boxes(image: Image.Image, areas: list[dict]) -> list[dict]:
    width, height = image.size
    thickness = max(2, round(max(width, height) / 600))
    font_size = max(12, round(max(width, height) / 100))
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
    font = ImageFont.truetype(str(font_path), font_size) if font_path.exists() else ImageFont.load_default(size=font_size)
    draw = ImageDraw.Draw(image)
    legend, occupied = [], []
    for index, area in enumerate(areas):
        validate_region(area['bbox'], width, height)
        points = region_vertices(area['bbox'])
        x1, y1, x2, y2 = region_crop_box(area['bbox'])
        color = PALETTE[index % len(PALETTE)]
        draw.line([tuple(p) for p in points] + [tuple(points[0])], fill=color, width=thickness, joint='curve')
        label = area['area_id']
        box = draw.textbbox((0, 0), label, font=font)
        padding = max(2, font_size // 6)
        label_width = min(width, box[2] - box[0] + 2 * padding)
        label_height = min(height, box[3] - box[1] + 2 * padding)
        label_x = max(0, min(x1, width - label_width))
        choices = [max(0, y1 - label_height), min(y1, height - label_height),
                   max(0, min(y2, height - label_height))]
        def overlap(y):
            return sum(max(0, min(label_x + label_width, rx2) - max(label_x, rx1)) *
                       max(0, min(y + label_height, ry2) - max(y, ry1)) for rx1, ry1, rx2, ry2 in occupied)
        label_y = min(choices, key=overlap)
        rect = (label_x, label_y, label_x + label_width - 1, label_y + label_height - 1)
        draw.rectangle(rect, fill=color)
        draw.text((label_x + padding - box[0], label_y + padding - box[1]), label, font=font, fill=(0, 0, 0))
        occupied.append((label_x, label_y, label_x + label_width, label_y + label_height))
        legend.append({'area_id': label, 'color_rgb': list(color), 'bbox': area['bbox'],
                       'label_pixel_box': list(rect)})
    return legend


def _crop_region(original: Image.Image, bbox) -> tuple[Image.Image, list[int], list[list[float]]]:
    """保留区域内原始 RGB 像素；外接裁剪框中的区域外像素透明且 RGB 置零。"""
    import numpy as np
    crop_box = region_crop_box(bbox)
    local = [[x - crop_box[0], y - crop_box[1]] for x, y in region_vertices(bbox)]
    crop = original.crop(tuple(crop_box))
    mask = np.zeros((crop.height, crop.width), dtype=np.uint8)
    xs = np.arange(crop.width, dtype=np.float64)[None, :] + 0.5
    # 按行分块，避免为大幅面 ROI 创建多份完整浮点网格。
    for top in range(0, crop.height, 128):
        bottom = min(crop.height, top + 128)
        ys = np.arange(top, bottom, dtype=np.float64)[:, None] + 0.5
        inside = np.ones((bottom - top, crop.width), dtype=bool)
        for i, a in enumerate(local):
            b = local[(i + 1) % 4]
            inside &= ((b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])) >= 0
        mask[top:bottom] = inside.astype(np.uint8) * 255
    roi = Image.new('RGBA', crop.size, (0, 0, 0, 0))
    roi.paste(crop, mask=Image.fromarray(mask))
    crop.close()
    return roi, crop_box, local


def draw_candidates(state: dict) -> dict:
    """Node 4 的业务产物只有统一 regions.json、全幅框图和 Area_n.png。"""
    started = time.monotonic()
    output = resolve_path(state['trace_dir']) / 'node4'
    runtime = resolve_path(state['runtime_dir'])
    try:
        if _digest(state['image_path']) != state['image_digest']:
            raise PositioningError('可视化前处理后的 TIFF 已改变')
        config = load_candidate_config(runtime / 'node3/prompt_system')
        schema = config['schema']
        width, height = state['image']['width'], state['image']['height']
        validate_candidates(state['result'], schema, width, height, state.get('max_regions'))
        view = state['candidate_view']
        validate_candidates(state['model_candidates'], schema, view['width'], view['height'], state.get('max_regions'))
        original_path = resolve_path(state['rgb_image_path'])
        original_digest = _digest(original_path)
        output.mkdir(parents=True, exist_ok=False)
        region_path = output / 'regions.json'
        annotated_path = output / 'bbox_satellite.png'
        with Image.open(original_path) as source:
            original = source.convert('RGB')
        if original.size != (width, height):
            original.close()
            raise PositioningError('原图尺寸与候选坐标不一致')
        try:
            annotated = original.copy()
            legend = _draw_boxes(annotated, state['result']['areas'])
            annotated.save(annotated_path)
            annotated.close()
            areas = []
            for area in state['result']['areas']:
                roi_path = output / (area['area_id'] + '.png')
                roi, crop_box, local = _crop_region(original, area['bbox'])
                roi.save(roi_path)
                roi.close()
                areas.append({**copy.deepcopy(area), 'roi_image_path': str(roi_path),
                              'roi_crop_bbox': crop_box, 'roi_polygon_local': local,
                              'roi_offset_xy': crop_box[:2], 'roi_outside_region': 'transparent_black'})
        finally:
            original.close()
        if _digest(original_path) != original_digest or _digest(state['image_path']) != state['image_digest']:
            raise PositioningError('可视化期间输入影像发生改变')
        geometry_rule = config['rules']['bbox_policy']
        regions = {'format_version': 2, 'task_id': state['task_id'], 'path_base': 'project_root',
                   'image': state['image'], 'input_view': view,
                   'primary_mode': state.get('primary_mode'), 'task_context': state.get('task_context', {}),
                   'geo_coordinates': state.get('geo_coordinates'),
                   'original_image_path': str(original_path), 'original_image_sha256': original_digest,
                   'annotated_image_path': str(annotated_path),
                   'bbox_coordinate_system': 'processed_image_pixel_edges_clockwise_quadrilateral',
                   'coordinate_note': 'bbox 是全幅处理图上的四个顺时针顶点；ROI 像素坐标加 roi_offset_xy 为全图坐标。',
                   'instructions': ['只按 area_id 读取对应区域，不重新编号或改变 bbox。',
                                    '全局图只添加区域框和编号，无道路 mask 或 polygon；区域颜色不表示适用性。',
                                    'ROI 区域内保留原始 RGB 像素，区域外透明，转换 RGB 后为黑色留白，不是真实地物。'],
                   'geometry_validation': {'passed': True, 'rule_id': geometry_rule['id'],
                                           'rule_name': geometry_rule['title'], 'boundary_contact_allowed': True},
                   'overlap_resolution': state.get('inference', {}).get('overlap_resolution'),
                   'legend': legend, 'areas': areas}
        write_json(region_path, regions)
        jobs = [{'area_id': area['area_id'], 'region_json_path': str(region_path)} for area in areas]
        artifacts = {'region_json_path': str(region_path), 'annotated_image_path': str(annotated_path),
                     'annotated_image_size': [width, height], 'visualization_scope': 'full_processed_image',
                     'roi_image_paths': [area['roi_image_path'] for area in areas]}
        traces = record_trace(state, 'draw_candidates', {'status': 'completed', 'candidate_count': len(areas),
                             'artifacts': artifacts, 'latency_seconds': time.monotonic() - started})
        return relative_paths({'artifacts': artifacts, 'region_json_path': str(region_path),
                               'parallel_inputs': jobs, **traces})
    except Exception as exc:
        failure = {'status': 'failed', 'error': str(exc)}
        write_json(runtime / 'node4/error.json', failure)
        record_trace(state, 'draw_candidates', failure)
        raise
