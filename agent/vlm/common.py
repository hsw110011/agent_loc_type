"""各节点共用的 JSON、图像校验及配置读取。"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from .client import VLMError

class PositioningError(VLMError):
    pass


from agent.paths import PROJECT_ROOT, resolve_path, relative_path


def relative_paths(value: Any) -> Any:
    """在状态与持久化边界统一保存相对路径，HTTP/data URL 和普通文本保持原样。"""
    if isinstance(value, dict):
        return {key: relative_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relative_paths(item) for item in value]
    if isinstance(value, Path):
        return relative_path(value)
    if isinstance(value, str) and value.startswith('/') and '\n' not in value:
        return relative_path(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(relative_paths(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def validate_bbox(bbox, width: int, height: int) -> None:
    if (not isinstance(bbox, list) or len(bbox) != 4 or
            any(isinstance(v, bool) or not isinstance(v, int) for v in bbox)):
        raise PositioningError('bbox 必须为四个整数 [x_min, y_min, x_max, y_max]')
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise PositioningError(f'bbox {bbox} 越界或面积不为正；图像尺寸 {width}×{height}')


def region_vertices(bbox) -> list[list[float]]:
    """简单多边形；兼容旧四顶点区域与历史 xyxy 可视化。"""
    if isinstance(bbox, list):
        if len(bbox) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox):
            x1, y1, x2, y2 = bbox
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if len(bbox) >= 3 and all(isinstance(p, list) and len(p) == 2 for p in bbox):
            return bbox
    raise PositioningError('bbox 必须为至少三个二维顶点组成的多边形')


def validate_region(bbox, width: int, height: int) -> None:
    from shapely.geometry import Polygon
    from shapely.validation import explain_validity
    points = region_vertices(bbox)
    for point in points:
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point):
            raise PositioningError('bbox 顶点坐标必须为有限数值')
        if not (0 <= point[0] <= width and 0 <= point[1] <= height):
            raise PositioningError(f'bbox 顶点 {point} 越界；图像尺寸 {width}×{height}')
    if len({tuple(point) for point in points}) != len(points):
        raise PositioningError('bbox 不能重复顶点或在末尾重复首点')
    signed_area = sum(a[0] * b[1] - b[0] * a[1]
                      for a, b in zip(points, points[1:] + points[:1])) / 2
    polygon = Polygon(points)
    if not polygon.is_valid:
        raise PositioningError(f'bbox 必须为不自交的简单多边形：{explain_validity(polygon)}')
    # 像素坐标 y 向下，正的有向面积表示视觉上的顺时针。
    if signed_area <= 0 or polygon.area <= 0:
        raise PositioningError('bbox 必须顺时针且面积为正，不能共线')


def regions_overlap(first, second) -> bool:
    """按实际简单多边形的相交面积判断；凹形合法，边界接触允许。"""
    from shapely.geometry import Polygon
    a, b = Polygon(region_vertices(first)), Polygon(region_vertices(second))
    return a.intersection(b).area > 1e-9


def polygon_pixel_mask(bbox, width: int, height: int, *, offset_xy=(0, 0)):
    """像素中心落在多边形内部的 mask；兼容凹形，供覆盖校验和 ROI 共用。"""
    from rasterio.features import geometry_mask
    from affine import Affine
    points = region_vertices(bbox)
    geometry = {'type': 'Polygon', 'coordinates': [points + points[:1]]}
    return geometry_mask([geometry], out_shape=(height, width),
                         transform=Affine.translation(*offset_xy), invert=True)


def region_crop_box(bbox) -> list[int]:
    """整数外接框仅用于 ROI 读取，不用于候选区域相交判定。"""
    points = region_vertices(bbox)
    return [math.floor(min(p[0] for p in points)), math.floor(min(p[1] for p in points)),
            math.ceil(max(p[0] for p in points)), math.ceil(max(p[1] for p in points))]


def read_config(path: Path) -> str:
    """配置未完成时明确报错，不使用隐式旧配置。"""
    try:
        text = resolve_path(path).read_text(encoding='utf-8').strip()
    except OSError as exc:
        raise PositioningError(f'无法读取配置：{path}') from exc
    if not text:
        raise PositioningError(f'配置尚未填写：{path}')
    return text


def _parse(text: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'重复 JSON 字段: {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'非法 JSON 数值: {value}')
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    except ValueError as exc:
        raise PositioningError(f'模型未返回有效 JSON: {exc}') from exc
    if not isinstance(value, dict):
        raise PositioningError('阶段输出必须为 JSON 对象')
    return value


def _digest(path: str) -> str:
    digest = hashlib.sha256()
    with resolve_path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _generation_schema(value):
    # 生成语法不强制 uniqueItems 等关键字；生成阶段只约束格式，
    # 去重和条件一致性仍由完整 JSON Schema 在程序中检查。
    if isinstance(value, dict):
        return {k: _generation_schema(v) for k, v in value.items() if k not in {'allOf', 'description', 'title', 'uniqueItems'}}
    if isinstance(value, list):
        return [_generation_schema(v) for v in value]
    return value



def record_trace(state: dict, node: str, details: dict) -> dict:
    """各节点共用一个任务目录，只记录元数据和路径，不写入图片 base64。"""
    from datetime import datetime, timezone
    directory = resolve_path(state['trace_dir'])
    trace_path = directory / 'trace.json'
    trace = json.loads(trace_path.read_text(encoding='utf-8')) if trace_path.exists() else {'events': []}
    trace['events'].append(relative_paths({'time': datetime.now(timezone.utc).isoformat(), 'node': node, **details}))
    temporary = directory / 'trace.json.tmp'
    temporary.write_text(json.dumps(trace, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(trace_path)
    (directory / 'trace.txt').write_text('\n\n'.join(
        f"[{event['time']}] {event['node']}\n" + json.dumps(event, ensure_ascii=False, indent=2)
        for event in trace['events']) + '\n', encoding='utf-8')
    return {'trace_json_path': str(trace_path), 'trace_txt_path': str(directory / 'trace.txt')}


def save_state(state: dict, filename: str = 'state.json') -> Path:
    """保存节点间传递的状态快照；图片通过路径传递，不写入 base64。"""
    path = resolve_path(state['trace_dir']) / filename
    bundle = relative_paths({key: value for key, value in state.items()
                             if key not in {'encoded_image', 'road_encoded_image'}})
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
    return path
