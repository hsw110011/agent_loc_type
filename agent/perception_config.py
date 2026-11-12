"""后续分析工具的配置，独立于四节点道路预处理。"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, fields
from typing import Any
from agent.paths import resolve_path


def _validate(config: Any) -> None:
    for name in ('threads', 'max_input_pixels', 'min_free_memory_mib'):
        if type(getattr(config,name)) is not int or getattr(config,name) <= 0:
            raise ValueError(f'{name} 必须为正整数')
    if not isinstance(config.device,str) or not config.device.strip():
        raise ValueError('device 必须为 auto/cpu/cuda[:n]')
    if config.device not in {'auto','cpu','cuda'} and not (config.device.startswith('cuda:') and config.device[5:].isdigit()):
        raise ValueError('device 必须为 auto/cpu/cuda[:n]')


@dataclass(frozen=True)
class DepthAnythingConfig:
    weights_dir: str = 'percection_tool/depth_anything/weights/depth_anything_v2_large'
    device: str = 'auto'
    threads: int = 4
    input_size: int = 518
    max_input_pixels: int = 16_000_000
    max_processed_pixels: int = 4_194_304
    min_free_memory_mib: int = 3072

    def __post_init__(self) -> None:
        _validate(self)
        for name in ('input_size','max_processed_pixels'):
            if type(getattr(self,name)) is not int or getattr(self,name) <= 0:
                raise ValueError(f'{name} 必须为正整数')
        if self.input_size % 14:
            raise ValueError('Depth Anything input_size 必须为 14 的倍数')
        if not isinstance(self.weights_dir,str) or not self.weights_dir.strip():
            raise ValueError('weights_dir 必须为非空路径')


@dataclass(frozen=True)
class SegEarthOV3Config:
    weights_dir: str = 'percection_tool/segearth_ov3/weights'
    device: str = 'auto'
    threads: int = 4
    tile_size: int = 1600
    stride: int = 1200
    score_threshold: float = .1
    max_input_pixels: int = 16_000_000
    min_free_memory_mib: int = 16384

    def __post_init__(self) -> None:
        _validate(self)
        if self.device == 'cpu':
            raise ValueError('SegEarth-OV3 使用 CUDA bfloat16 autocast，仅支持 CUDA')
        if type(self.tile_size) is not int or type(self.stride) is not int or not 0 < self.stride < self.tile_size:
            raise ValueError('需要 0 < stride < tile_size 的整数')
        if not isinstance(self.score_threshold,float) or not 0 < self.score_threshold < 1:
            raise ValueError('score_threshold 必须在 (0,1)')
        if not isinstance(self.weights_dir,str) or not self.weights_dir.strip():
            raise ValueError('weights_dir 必须为非空路径')


@dataclass(frozen=True)
class PerceptionToolsConfig:
    depth: DepthAnythingConfig = field(default_factory=DepthAnythingConfig)
    segearth_ov3: SegEarthOV3Config = field(default_factory=SegEarthOV3Config)

    @property
    def pipeline_semantic_config(self) -> SegEarthOV3Config:
        return self.segearth_ov3

    def to_dict(self) -> dict:
        return asdict(self)


def load_perception_config(path: str | None = None) -> PerceptionToolsConfig:
    if path is None:
        return PerceptionToolsConfig()
    import yaml
    try:
        raw = yaml.safe_load(resolve_path(path).read_text(encoding='utf-8'))
    except yaml.YAMLError as exc:
        raise ValueError(f'配置不是合法 JSON/YAML：{exc}') from exc
    if not isinstance(raw,dict) or set(raw)-{'depth','segearth_ov3'}:
        raise ValueError('配置仅接受 depth 和 segearth_ov3')
    settings = {}
    for name,cls in [('depth',DepthAnythingConfig),('segearth_ov3',SegEarthOV3Config)]:
        value = raw.get(name,{})
        if not isinstance(value,dict) or set(value)-{f.name for f in fields(cls)}:
            raise ValueError(f'{name} 配置不是对象或含未知字段')
        settings[name] = cls(**value)
    return PerceptionToolsConfig(**settings)
