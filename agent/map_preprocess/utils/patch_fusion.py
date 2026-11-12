"""Node 8 的纯几何融合：按原始 Patch 的正面积重叠求连通分组，保留真实并集。

重叠关系只由输入 Patch 两两判断，分组完成后才计算并集和外接矩形，融合结果不参与任何重叠判断。
union_polygon 是输入 Patch 的精确并集，bbox 仅用于裁剪、索引和可视化。坐标统一为原图全局像素。
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Any, Iterable, Sequence

import shapely
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from ..global_config import FUSABLE_DEGRADATION_TYPES as DEFAULT_FUSION_TYPES

MIXED_TYPE = 'mixed'  # 仅跨类型融合时出现：组内含多种退化类型


def has_area_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    """两个 [xmin, ymin, xmax, ymax] 是否有严格为正的重叠面积；共边、共点都不算。"""
    overlap_width = min(a[2], b[2])-max(a[0], b[0])
    overlap_height = min(a[3], b[3])-max(a[1], b[1])
    return overlap_width > 0 and overlap_height > 0


def bbox_problem(bbox: Any) -> str | None:
    """返回 BBox 的问题代码；合法时返回 None。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 'bbox_malformed'
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in bbox):
        return 'bbox_not_finite'
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return 'bbox_empty'
    return None


def split_patches(patches: Iterable[Any], allowed_types: Iterable[str]) -> tuple[list[dict], list[dict], list[dict]]:
    """分成 (参与融合, 类型不参与融合, 无效)；后两者记录 patch_id 与原因，供节点落盘和告警。"""
    allowed = set(allowed_types)
    valid, excluded, rejected, seen = [], [], [], set()
    for patch in patches:
        if not isinstance(patch, dict) or not patch.get('patch_id') or not patch.get('degradation_type'):
            rejected.append({'patch': patch, 'reason': 'missing_required_field'})
            continue
        patch_id, kind = patch['patch_id'], patch['degradation_type']
        if kind not in allowed:
            excluded.append({'patch_id': patch_id, 'degradation_type': kind, 'reason': 'degradation_type_not_fused'})
            continue
        problem = bbox_problem(patch.get('bbox')) or ('duplicate_patch_id' if patch_id in seen else None)
        if problem:
            rejected.append({'patch_id': patch_id, 'bbox': patch.get('bbox'), 'reason': problem})
            continue
        seen.add(patch_id)
        valid.append(patch)
    return valid, excluded, rejected


def overlap_components(patches: Sequence[dict], *, cross_type: bool = False) -> list[list[dict]]:
    """并查集求连通分量：边只来自原始 Patch 两两的正面积重叠，孤立 Patch 自成一组。"""
    parent = list(range(len(patches)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    order = sorted(range(len(patches)), key=lambda index: patches[index]['bbox'][0])
    for position, i in enumerate(order):
        for j in order[position+1:]:
            if patches[j]['bbox'][0] >= patches[i]['bbox'][2]:
                break  # 按 xmin 升序，后面的 Patch 在 x 方向都不可能再有正重叠
            same_type = patches[i]['degradation_type'] == patches[j]['degradation_type']
            if (cross_type or same_type) and has_area_overlap(patches[i]['bbox'], patches[j]['bbox']):
                parent[find(i)] = find(j)
    groups: dict[int, list[dict]] = {}
    for index, patch in enumerate(patches):
        groups.setdefault(find(index), []).append(patch)
    return list(groups.values())


def _plain(value: float) -> int | float:
    """整数值输出为 int，其余保持 float，保证产物可直接 JSON 序列化。"""
    number = float(value)
    return int(number) if number.is_integer() else number


def _geojson(geometry: Any) -> dict:
    def convert(item: Any) -> Any:
        return [convert(value) for value in item] if isinstance(item, (list, tuple)) else _plain(item)
    data = mapping(geometry)
    return {'type': data['type'], 'coordinates': convert(data['coordinates'])}


def _fuse_group(members: list[dict]) -> dict:
    """单个分组的并集几何：L/T 形、带洞区域都原样保留，共线中间点去掉，环的起点和方向规范化。"""
    union = unary_union([box(*member['bbox']) for member in members])
    union = shapely.orient_polygons(union.simplify(0).normalize())
    xmin, ymin, xmax, ymax = union.bounds
    types = sorted({member['degradation_type'] for member in members})
    source_area = sum((m['bbox'][2]-m['bbox'][0])*(m['bbox'][3]-m['bbox'][1]) for m in members)
    return {'degradation_type': types[0] if len(types) == 1 else MIXED_TYPE,
            'source_degradation_types': types,
            'source_patch_ids': [member['patch_id'] for member in members],
            'source_patch_count': len(members),
            'bbox': [_plain(v) for v in (xmin, ymin, xmax, ymax)],
            'union_polygon': _geojson(union),
            'source_area_px2': _plain(source_area),
            'union_area_px2': _plain(union.area),
            'bbox_area_px2': _plain((xmax-xmin)*(ymax-ymin)),
            'source_patches': [{**member, 'bbox': [_plain(v) for v in member['bbox']]} for member in members]}


def fuse_patches(patches: Iterable[Any], *, allowed_types: Iterable[str] = DEFAULT_FUSION_TYPES,
                 cross_type: bool = False) -> dict:
    """返回 {'regions', 'excluded', 'rejected'}；相同输入（与顺序无关）得到相同的 regions。"""
    valid, excluded, rejected = split_patches(patches, allowed_types)
    canonical = sorted(valid, key=lambda patch: ([float(v) for v in patch['bbox']], str(patch['patch_id'])))
    groups = overlap_components(canonical, cross_type=cross_type)
    fused = sorted((_fuse_group(group) for group in groups), key=lambda region: (region['bbox'], region['source_patch_ids']))
    regions = [{'merged_id': f'Merged_{index}', **region} for index, region in enumerate(fused, 1)]
    return {'regions': regions, 'excluded': excluded, 'rejected': rejected}
