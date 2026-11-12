"""感知工具的轻量输入/输出协议，不导入具体模型。"""
from __future__ import annotations

from typing import Any, Protocol, TypedDict
from pathlib import Path
import numpy as np

class GraphSegmenter(Protocol):
    def predict_graph(self, image: np.ndarray, resolution_m_per_px: float,
                      valid_mask: np.ndarray | None = None) -> dict: ...






class RawRoadGraph(TypedDict):
    vertices: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    metadata: dict[str, Any]


class RoadGraphExtractor(Protocol):
    def extract(self, image_path_or_array: str | Path | np.ndarray) -> RawRoadGraph: ...


class DepthResult(TypedDict):
    depth: np.ndarray
    metadata: dict[str, Any]


class SemanticResult(TypedDict):
    mask: np.ndarray
    confidence: np.ndarray
    metadata: dict[str, Any]


class DepthEstimator(Protocol):
    def predict(self, image: np.ndarray, valid_mask: np.ndarray | None = None) -> DepthResult: ...
    def close(self) -> None: ...


class SemanticPredictor(Protocol):
    def predict(self, image: np.ndarray, valid_mask: np.ndarray | None = None) -> SemanticResult: ...
    def close(self) -> None: ...
