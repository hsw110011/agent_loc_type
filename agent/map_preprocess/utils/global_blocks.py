"""沿边弧长连续布块；只保存几何，读取源图时才裁剪。"""
from __future__ import annotations
import math
from pathlib import Path
import warnings
import numpy as np
from shapely.geometry import LineString
from ..io import load_manifest, open_image


def block_positions(length_m: float, config, *, warn: bool = True) -> list[float]:
    size, stride = config.block_size_m, config.block_size_m*(1-config.block_overlap_ratio)
    if length_m <= size:
        return [length_m/2]
    steps = math.ceil((length_m-size)/stride-1e-12)
    if config.long_edge_strategy == 'sparse3' and steps+1 > config.blocks_per_group:
        if warn:
            warnings.warn('sparse3 长边存在覆盖空洞', RuntimeWarning)
        return [size/2, length_m/2, length_m-size/2]
    # 将超出道路两端的范围平分，让中心始终在线上，同时维持固定步长。
    start = (length_m-steps*stride)/2
    return [start+i*stride for i in range(steps+1)]


def bbox_metadata(center: list[float], size_m: float, resolution: float, width: int, height: int) -> dict:
    side = size_m/resolution
    x, y = center
    bbox = [x-side/2, y-side/2, x+side/2, y+side/2]
    pixels = max(1, round(side))
    # 假设裁剪栅格就近对齐（最多半像素偏差），物理 BBox 和中心坐标保持精确。
    x0, y0 = math.floor(x-pixels/2+.5), math.floor(y-pixels/2+.5)
    crop = [x0, y0, x0+pixels, y0+pixels]
    valid = [max(0, crop[0]), max(0, crop[1]), min(width, crop[2]), min(height, crop[3])]
    return {'global_bbox': bbox, 'crop_bbox': crop, 'size_px': pixels, 'size_m': size_m,
            'valid_global_bbox': valid,
            'valid_window_bbox': [valid[0]-crop[0], valid[1]-crop[1], valid[2]-crop[0], valid[3]-crop[1]],
            'padding_ltrb': [valid[0]-crop[0], valid[1]-crop[1], crop[2]-valid[2], crop[3]-valid[3]]}


def make_block(edge: dict, arc: float, index: int, count: int, image: dict,
               tiles: list[dict], resolution: float, config, *, line: LineString | None = None) -> dict:
    """在给定弧长构造一个轻量块；复用于连续布块及去重后的保留块。"""
    if line is None:
        line = LineString(edge['polyline_xy'])
    point = line.interpolate(arc/resolution)
    center = [point.x, point.y]
    before = line.interpolate(max(0., arc/resolution-.5))
    after = line.interpolate(min(line.length, arc/resolution+.5))
    delta = np.array([after.x-before.x, after.y-before.y])
    norm = np.linalg.norm(delta)
    direction = (delta/norm).tolist() if norm else [1., 0.]
    tile = next((t['tile_id'] for t in tiles if t['x0'] <= point.x < t['x1'] and t['y0'] <= point.y < t['y1']), None)
    identifier = f"E{edge['edge_id']}_B{index}"
    return {'block_id': identifier, 'edge_id': edge['edge_id'], 'seq': index,
        'global_xy': center, 'arc_length_m': arc, 'arc_length_px': arc/resolution,
        'tangent_xy': direction, 'tangent_angle_rad': math.atan2(direction[1], direction[0]),
        'node_ids': [edge['u'], edge['v']], 'tile_id': tile, 'features': {},
        'predecessor_ids': [f"E{edge['edge_id']}_B{index-1}"] if index else [],
        'successor_ids': [f"E{edge['edge_id']}_B{index+1}"] if index+1 < count else [],
        **bbox_metadata(center, config.block_size_m, resolution, image['width'], image['height']),
        'context': bbox_metadata(center, config.context_size_m, resolution, image['width'], image['height'])
                   if config.context_size_m is not None else None}


def blocks_for_edge(edge: dict, image: dict, tiles: list[dict], resolution: float, config, *, warn: bool = True) -> tuple[list[dict], dict]:
    positions = block_positions(edge['length_m'], config, warn=warn)
    line = LineString(edge['polyline_xy'])
    blocks = [make_block(edge, arc, index, len(positions), image, tiles, resolution, config, line=line)
              for index, arc in enumerate(positions)]
    # 弧长支持区间是方块对道路覆盖的保守下界，弯道回返可能增加实际覆盖。
    intervals = [(max(0., p-config.block_size_m/2), min(edge['length_m'], p+config.block_size_m/2)) for p in positions]
    covered, end = 0., 0.
    for start, stop in intervals:
        covered += max(0., stop-max(start, end))
        end = max(end, stop)
    overlaps = [1-(b-a)/config.block_size_m for a, b in zip(positions, positions[1:])]
    coverage = min(1., covered/edge['length_m'])
    if warn and coverage < 1-1e-9:
        warnings.warn(f"边 {edge['edge_id']} 覆盖率不足 1: {coverage:.3f}", RuntimeWarning)
    if warn and overlaps and any(abs(r-config.block_overlap_ratio) > 1e-9 for r in overlaps):
        warnings.warn(f"边 {edge['edge_id']} 相邻名义弧长重叠率偏离配置", RuntimeWarning)
    return blocks, {'edge_id': edge['edge_id'], 'coverage_ratio': coverage,
                    'min_overlap_ratio': min(overlaps) if overlaps else None,
                    'max_overlap_ratio': max(overlaps) if overlaps else None, 'block_count': len(blocks)}


def connect_blocks(edges: list[dict], blocks: list[dict]) -> None:
    by_edge = {e['edge_id']: [] for e in edges}
    for block in blocks:
        by_edge[block['edge_id']].append(block)
    endpoints = {}
    for edge in edges:
        selected = by_edge[edge['edge_id']]
        for node, block, field in ((edge['u'], selected[0], 'predecessor_ids'),
                                    (edge['v'], selected[-1], 'successor_ids')):
            endpoints.setdefault(node, []).append((edge['edge_id'], block, field))
    for entries in endpoints.values():
        for edge_id, block, field in entries:
            block[field] = sorted(set(block[field]) | {other['block_id'] for other_edge, other, _ in entries
                                                      if other['block_id'] != block['block_id'] and
                                                      (other_edge != edge_id or len(by_edge[edge_id]) > 1)})


def group_blocks(edges: list[dict], blocks: list[dict], config) -> list[dict]:
    groups = []
    for edge in edges:
        selected = [b for b in blocks if b['edge_id'] == edge['edge_id']]
        start, index = 0, 0
        while start < len(selected):
            group = selected[start:start+config.blocks_per_group]
            boxes = [b['global_bbox'] for b in group]
            groups.append({'group_id': f"E{edge['edge_id']}_G{index}", 'edge_id': edge['edge_id'],
                'block_ids': [b['block_id'] for b in group],
                'global_bbox': [min(b[0] for b in boxes), min(b[1] for b in boxes),
                                max(b[2] for b in boxes), max(b[3] for b in boxes)]})
            if start+config.blocks_per_group >= len(selected):
                break
            start += config.blocks_per_group-config.group_overlap_blocks
            index += 1
    return groups


def _data(manifest: dict | str | Path) -> dict:
    return load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest


def _crop(data: dict, record: dict, mode: str) -> np.ndarray:
    if mode not in {'edge', 'reflect', 'constant'}:
        raise ValueError('未知 pad_mode')
    x0, y0, x1, y1 = record['crop_bbox']
    width, height = data['image']['width'], data['image']['height']
    xs, ys = np.arange(x0, x1), np.arange(y0, y1)
    valid_x, valid_y = (xs >= 0) & (xs < width), (ys >= 0) & (ys < height)
    def indices(values: np.ndarray, limit: int) -> np.ndarray:
        if mode != 'reflect' or limit == 1:
            return np.clip(values, 0, limit-1)
        reflected = values % (2*(limit-1))
        return np.minimum(reflected, 2*(limit-1)-reflected)
    xs, ys = indices(xs, width), indices(ys, height)
    # 读取映射后的最小源窗口；即使 1px 请求完全在边界外，也可正确填充。
    left, top = int(xs.min()), int(ys.min())
    with open_image(data['image']['path'], data['config']['pillow_max_image_pixels']) as image:
        pixels = image.read([left, top, int(xs.max())+1, int(ys.max())+1])
    result = pixels[(ys-top)[:, None], (xs-left)[None, :]].copy()
    if mode == 'constant':
        result[~(valid_y[:, None] & valid_x[None, :])] = 0
    return result


def crop_block(manifest: dict | str | Path, block_id: str, pad_mode: str | None = None) -> np.ndarray:
    data = _data(manifest)
    block = next(b for b in data['blocks'] if b['block_id'] == block_id)
    return _crop(data, block, pad_mode or data['config']['pad_mode'])


def crop_context(manifest: dict | str | Path, block_id: str) -> np.ndarray:
    data = _data(manifest)
    block = next(b for b in data['blocks'] if b['block_id'] == block_id)
    if block['context'] is None:
        raise ValueError('context_size_m 已关闭')
    return _crop(data, block['context'], data['config']['pad_mode'])


def crop_blocks_for_group(manifest: dict | str | Path, group_id: str) -> list[np.ndarray]:
    data = _data(manifest)
    group = next(g for g in data['groups'] if g['group_id'] == group_id)
    return [crop_block(data, identifier) for identifier in group['block_ids']]
