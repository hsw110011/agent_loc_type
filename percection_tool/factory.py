"""按配置延迟选择工具，不包含模型实现。"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .interfaces import RoadGraphExtractor, DepthEstimator, SemanticPredictor

if TYPE_CHECKING:
    from agent.map_preprocess.global_config import GlobalPreprocessConfig
    from agent.perception_config import DepthAnythingConfig, SegEarthOV3Config


def create_graph_extractor(config: GlobalPreprocessConfig, resolution_m_per_px: float, output_dir: str,
                           *, magroad_checkpoint: str | None = None) -> RoadGraphExtractor:
    if config.segmenter == 'magroad_scene':
        if not magroad_checkpoint:
            raise ValueError('segmenter=magroad_scene 需要场景节点选出的 MaGRoad 权重')
        from .magroad.graph_adapter import MaGRoadGraphExtractor
        return MaGRoadGraphExtractor(config, resolution_m_per_px, output_dir, magroad_checkpoint)
    if config.segmenter == 'dummy':
        from .dummy import DummyRoadGraphExtractor
        return DummyRoadGraphExtractor(config.dummy_graph_path)
    raise ValueError('全局道路图只支持 magroad_scene 或 dummy')


def create_depth_estimator(config: DepthAnythingConfig | None = None) -> DepthEstimator:
    from .depth_anything import DepthAnythingEstimator
    return DepthAnythingEstimator(config)


def create_semantic_segmenter(config: SegEarthOV3Config | None = None) -> SemanticPredictor:
    from .segearth_ov3 import SegEarthOV3Segmenter
    return SegEarthOV3Segmenter(config)
