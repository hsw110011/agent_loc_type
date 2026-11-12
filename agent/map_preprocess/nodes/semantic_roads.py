"""整图语义推理（流程语义模型，默认 SegEarth-OV3）：分窗推理后拼接，按输入哈希 + 模型身份 + 分辨率缓存。

语义路线（Node 3）和 Node 6/7 共用这份结果。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any
from uuid import uuid4
import zipfile

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..global_config import GlobalPreprocessConfig
from .. import output_layout as layout
from ..io import open_image, resolve_path
from ..state import PreprocessState

CACHE_VERSION = 1


def _windows(length: int, size: int) -> list[tuple[int, int]]:
    return [(start, min(length, start+size)) for start in range(0, length, size)]


def _read_working(path: str, config: GlobalPreprocessConfig, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """整图按工作网格 size=(w, h) 读取 RGB 与有效掩码（rasterio 降采样读取，不载入原分辨率整图）。"""
    with open_image(path, config.pillow_max_image_pixels) as source:
        if source.raster is not None:
            from rasterio.enums import Resampling
            rgb = source.raster.read([1, 2, 3], out_shape=(3, size[1], size[0]), resampling=Resampling.average)
            valid = source.raster.dataset_mask(out_shape=(size[1], size[0]), resampling=Resampling.nearest) > 0
            return rgb.transpose(1, 2, 0), valid
        image = source.pillow.convert('RGB').resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image), np.ones((size[1], size[0]), bool)


def semantic_cache_key(state: PreprocessState, config: GlobalPreprocessConfig, segmenter: Any) -> tuple[str, Any, tuple[int, int], float]:
    """整图语义的缓存键 (key, 模型指纹, 工作网格 (w, h), 缩放)，不加载图像；指纹为 None（注入的模型未声明身份）时不可缓存。"""
    from agent.perception_config import load_perception_config
    from .block_features import _model_fingerprint
    perception = load_perception_config(config.perception_config)
    width, height = state['image']['width'], state['image']['height']
    # 只降采样：原图比 semantic_roads_resolution_m 更粗时按原分辨率推理；差 1% 以内（0.04999 对 0.05）不重采样
    scale = min(1., state['resolution_m_per_px']/config.semantic_roads_resolution_m)
    scale = 1. if scale > .99 else scale
    size = (max(1, round(width*scale)), max(1, round(height*scale)))
    fingerprint = _model_fingerprint(segmenter, perception.pipeline_semantic_config)
    key = hashlib.sha256(json.dumps({'version': CACHE_VERSION, 'input_sha256': state['input_sha256'],
        'size': size, 'model': fingerprint, 'window': config.semantic_roads_window_px}, sort_keys=True).encode()).hexdigest()
    return key, fingerprint, size, scale


def whole_image_semantic(state: PreprocessState, config: GlobalPreprocessConfig, segmenter: Any,
                         logger: logging.Logger) -> tuple[np.ndarray, np.ndarray, dict]:
    """整图语义：返回工作网格上的 (labels uint8, confidence float32, 信息)；分窗推理避免超过模型输入上限。"""
    from agent.perception_config import load_perception_config
    from .block_features import _release_cuda
    perception = load_perception_config(config.perception_config)
    key, fingerprint, size, scale = semantic_cache_key(state, config, segmenter)
    cache = None
    if fingerprint is not None and config.cache_enabled:
        root = resolve_path(config.cache_dir)/'semantic_whole'
        cache = layout.find_cache(root, key, '.npz') or root/layout.cache_name(state['image_path'], key, '.npz')
    info = {'working_size': list(size), 'resolution_m': state['resolution_m_per_px']/scale, 'cache_key': key[:24],
            'model': fingerprint}
    if cache is not None and cache.is_file():
        try:
            with np.load(cache, allow_pickle=False) as data:
                return data['labels'], data['confidence'].astype(np.float32), {**info, 'cache_hit': True}
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):  # 缓存损坏（如写入时断电）时重新推理并覆盖
            pass
    rgb, valid = _read_working(state['image_path'], config, size)
    labels = np.full((size[1], size[0]), 255, np.uint8)
    confidence = np.zeros((size[1], size[0]), np.float32)
    window, margin = config.semantic_roads_window_px, config.semantic_roads_window_margin_px
    tool = segmenter
    if tool is None:
        from percection_tool.factory import create_semantic_segmenter
        tool = create_semantic_segmenter(perception.pipeline_semantic_config)
    try:
        grid = [(y, x) for y in _windows(size[1], window) for x in _windows(size[0], window)]
        for (y0, y1), (x0, x1) in tqdm(grid, desc='整图语义', unit='窗', dynamic_ncols=True, disable=None):
            # 带 margin 推理、只保留中心区域，避免窗口边缘效应
            a, b = max(0, y0-margin), min(size[1], y1+margin)
            c, d = max(0, x0-margin), min(size[0], x1+margin)
            if not valid[a:b, c:d].any():
                continue
            output = tool.predict(np.ascontiguousarray(rgb[a:b, c:d]), valid_mask=valid[a:b, c:d])
            labels[y0:y1, x0:x1] = output['mask'][y0-a:y1-a, x0-c:x1-c]
            confidence[y0:y1, x0:x1] = output['confidence'][y0-a:y1-a, x0-c:x1-c]
        logger.info('整图语义推理完成：%s 窗口 %d', size, window)
    finally:
        tool.close()
        _release_cuda()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(f'.{uuid4().hex}.npz')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, labels=labels, confidence=confidence.astype(np.float16))
            handle.flush()
            os.fsync(handle.fileno())  # 先落盘再改名：写入时断电也不会留下半个缓存
        temporary.replace(cache)
    return labels, confidence, {**info, 'cache_hit': False}
