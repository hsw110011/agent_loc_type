"""参数的唯一默认值来源；JSON/YAML 使用同一份配置契约（只有整图 global_blocks 流程）。"""
from __future__ import annotations

import re

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreprocessConfig:
    """通用部分：输入影像、分辨率、道路模型推理、输出与出图。整图流程见 GlobalPreprocessConfig。"""
    output_dir: str = 'output/map_preprocess/global_blocks'
    fallback_resolution_m_per_px: float = 1.0
    zoom: float | None = None
    center_lat: float | None = None
    corner_lonlat: tuple[tuple[float, float], ...] = ()
    # 假设非正方形像元使用两轴长度的算术平均，下游仍保持原始像素网格。
    resolution_anisotropy_warning_ratio: float = 0.05
    max_tile_pixels: int = 32_000_000
    pillow_max_image_pixels: int = 100_000_000
    segmenter: str = 'magroad_scene'  # magroad_scene（默认）或 dummy（离线测试）
    # crop_block/crop_context 读取超出原图的部分时的填充方式（Node 5 不补边，直接裁掉）。
    pad_mode: str = 'edge'
    annotation_line_width: int = 1
    annotation_offset_px: float = 12.0
    overview_max_side: int = 4096
    debug: bool = False
    semantic_features_enabled: bool = False
    elevation_features_enabled: bool = False
    vlm_candidate_detection_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.corner_lonlat, (list, tuple)):
            raise ValueError('corner_lonlat 必须为列表')
        positive_ints = ('max_tile_pixels', 'pillow_max_image_pixels', 'annotation_line_width', 'overview_max_side')
        for name in positive_ints:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} 必须为合法整数')
        for name in ('fallback_resolution_m_per_px',):
            self._number(name, getattr(self, name), positive=True)
        for name in ('annotation_offset_px', 'resolution_anisotropy_warning_ratio'):
            self._number(name, getattr(self, name))
            if getattr(self, name) < 0:
                raise ValueError(f'{name} 不能为负')
        if self.segmenter not in {'dummy', 'magroad_scene'}:
            raise ValueError('segmenter 必须为 magroad_scene 或 dummy')
        if self.pad_mode not in {'edge', 'reflect', 'constant'}:
            raise ValueError('pad_mode 必须为 edge、reflect 或 constant（零填充）')
        if self.zoom is not None:
            self._number('zoom', self.zoom)
            if not 0 <= self.zoom <= 30:
                raise ValueError('zoom 必须在 [0, 30]')
        if self.center_lat is not None:
            self._number('center_lat', self.center_lat)
            if abs(self.center_lat) > 85.05112878:
                raise ValueError('center_lat 超出 Web Mercator 纬度范围')
        if self.corner_lonlat:
            if len(self.corner_lonlat) != 4:
                raise ValueError('corner_lonlat 必须为四个 [longitude, latitude]')
            for pair in self.corner_lonlat:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError('corner_lonlat 坐标必须包含两个数值')
                for value in pair:
                    self._number('corner_lonlat', value)
                if abs(pair[0]) > 180 or abs(pair[1]) > 85.05112878:
                    raise ValueError('corner_lonlat 超出经纬度范围')
        for name in ('debug', 'semantic_features_enabled', 'elevation_features_enabled',
                     'vlm_candidate_detection_enabled'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} 必须为布尔值')
        for name in ('output_dir',):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'{name} 必须为非空字符串')

    @staticmethod
    def _number(name: str, value: Any, positive: bool = False) -> None:
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f'{name} 必须为有限数值')
        if positive and value <= 0:
            raise ValueError(f'{name} 必须为正数')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None) -> 'GlobalPreprocessConfig':
    """读取 JSON/YAML 配置，返回整图流程配置；未知字段报错。"""
    from .global_config import GlobalPreprocessConfig
    if path is None:
        return GlobalPreprocessConfig()
    import yaml
    from .io import resolve_path
    try:
        raw = yaml.safe_load(resolve_path(path).read_text(encoding='utf-8'))
    except yaml.YAMLError as exc:
        raise ValueError(f'配置不是合法 JSON/YAML：{exc}') from exc
    if not isinstance(raw, dict):
        raise ValueError('配置必须是 JSON/YAML 对象')
    if any(not isinstance(key, str) for key in raw):
        raise ValueError('配置字段名必须是字符串')
    if raw.get('workflow', 'global_blocks') != 'global_blocks':
        raise ValueError('只支持 workflow: global_blocks（旧的切片/采样点流程已删除）')
    if len(re.findall(r'^run_until\s*:', resolve_path(path).read_text(encoding='utf-8'), re.M)) > 1:
        raise ValueError('run_until 只能保留一行不注释')
    raw.pop('road_topo', None)   # 道路拓扑参数由 utils/road_topo.py 读取
    unknown = set(raw) - {field.name for field in fields(GlobalPreprocessConfig)}
    if unknown:
        raise ValueError(f'未知配置项：{sorted(unknown)}')
    return GlobalPreprocessConfig(**raw)
