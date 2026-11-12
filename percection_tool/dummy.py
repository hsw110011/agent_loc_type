"""仅供合成图测试的感知替身。"""
from __future__ import annotations

from pathlib import Path
from agent.paths import read_json
from .interfaces import RawRoadGraph
import numpy as np


class DummyRoadGraphExtractor:
    """读取统一图 JSON；无文件时生成一条带 degree-2 中点的合成直路。"""
    def __init__(self, graph_path: str | None = None):
        self.graph_path = graph_path

    def extract(self, image_path_or_array: str | Path | np.ndarray) -> RawRoadGraph:
        if self.graph_path:
            return read_json(self.graph_path)
        if isinstance(image_path_or_array, np.ndarray):
            height, width = image_path_or_array.shape[:2]
        else:
            from PIL import Image
            with Image.open(image_path_or_array) as image:
                width, height = image.size
        points = [[width*f, height/2] for f in (.1, .5, .9)]
        return {'vertices': [{'node_id': str(i), 'xy': p} for i, p in enumerate(points)],
                'edges': [{'u': str(i), 'v': str(i+1), 'polyline_xy': points[i:i+2], 'confidence': 1.}
                          for i in range(2)], 'metadata': {'model': 'synthetic_dummy'}}
