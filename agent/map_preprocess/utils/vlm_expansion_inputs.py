"""Node 7 输入：数据覆盖足迹、多档框线、高度可比性检查。不加载模型、不调用 VLM。

覆盖足迹与“内部无数据”分开处理：只有与栅格边缘连通的空白才算地图外，内部空洞保留给数据质量规则。
部分 DOM 没有声明掩码，地图外是纯黑/纯白填充，所以 RGB 足迹同时看掩码和填充色。
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import block_features as bf
from . import vlm_review_inputs as vi

COVERAGE_MAX_SIDE = 2048     # 足迹网格长边上限；约 0.5~1 m/格，足够判断 64~100 m 窗口
FILL_WHITE = 254             # 三通道都不低于该值、且与边缘连通才视为白色填充
# 框线样式按档位顺序：基准实线、其后短虚线、点线。(实线段, 间隔) 为图像像素，None 为实线。
TIER_STYLES = (('实线', None), ('短虚线', (14, 8)), ('点线', (3, 6)))


def _outside(blank: np.ndarray) -> np.ndarray:
    """与栅格边缘连通的空白；其余空白是地图内部的空洞。"""
    count, labels = cv2.connectedComponents(blank.astype(np.uint8), connectivity=4)
    if count <= 1:
        return np.zeros(blank.shape, bool)
    edge = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    return np.isin(labels, edge[edge > 0])


def rgb_blank(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (~mask) | (rgb.max(axis=2) == 0) | (rgb.min(axis=2) >= FILL_WHITE)


def _read_rgb_grid(manifest: dict, shape: tuple[int, int]) -> np.ndarray:
    from ..io import open_image
    height, width = shape
    with open_image(manifest['image']['path'], manifest['config']['pillow_max_image_pixels']) as source:
        if source.raster is not None:
            from rasterio.enums import Resampling
            rgb = source.raster.read([1, 2, 3], out_shape=(3, height, width), resampling=Resampling.nearest)
            mask = source.raster.dataset_mask(out_shape=(height, width), resampling=Resampling.nearest) > 0
            return rgb_blank(rgb.transpose(1, 2, 0), mask)
        rgb = np.asarray(source.pillow.convert('RGB').resize((width, height), Image.Resampling.NEAREST))
        return rgb_blank(rgb, np.ones(shape, bool))


def coverage_grids(manifest: dict, labels: np.ndarray, with_dsm: bool) -> dict:
    """整图足迹（True = 地图内）：rgb、semantic，有 DSM 时再加 dsm；网格为原图 factor 倍下采样。"""
    from .dsm import read_block_dsm
    width, height = manifest['image']['width'], manifest['image']['height']
    factor = max(1, math.ceil(max(width, height)/COVERAGE_MAX_SIDE))
    shape = (math.ceil(height/factor), math.ceil(width/factor))
    semantic = cv2.resize(labels, shape[::-1], interpolation=cv2.INTER_NEAREST) == bf.IGNORE
    grids = {'factor': factor, 'rgb': ~_outside(_read_rgb_grid(manifest, shape)), 'semantic': ~_outside(semantic)}
    if with_dsm:
        dsm = read_block_dsm({**manifest['image']['dsm'], 'block_grid_factor': factor}, [0, 0, width, height])
        grids['dsm'] = ~_outside(~np.isfinite(dsm))
    return grids


def required_sources(review_type: str) -> tuple[str, str]:
    return ('rgb', 'semantic') if review_type == 'semantic' else ('rgb', 'dsm')


def uncovered(grids: dict, review_type: str, window: list[int], image_size: tuple[int, int]) -> list[str]:
    """窗口未完整落入的数据源；空列表表示该支路所需数据完整覆盖窗口。"""
    x0, y0, x1, y1 = window
    if x0 < 0 or y0 < 0 or x1 > image_size[0] or y1 > image_size[1]:
        return ['raster_bounds']
    f = grids['factor']
    view = np.s_[y0//f:-(-y1//f), x0//f:-(-x1//f)]
    return [name for name in required_sources(review_type) if not grids[name][view].all()]


def frame_box(window: list[int], outer: list[int], px: int) -> list[float]:
    """档位窗口在图像上的框（outer 为图像覆盖范围）；最外层框内缩，避免画在图像边缘外。"""
    scale = px/(outer[2]-outer[0])
    box = [(window[0]-outer[0])*scale, (window[1]-outer[1])*scale, (window[2]-outer[0])*scale, (window[3]-outer[1])*scale]
    return [min(max(v, 2.), px-3.) for v in box]


def _pattern_rect(draw: ImageDraw.ImageDraw, box: list[float], color, width: int, pattern: tuple[int, int] | None) -> None:
    x0, y0, x1, y1 = box
    for (a, b), (c, d) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        if pattern is None:
            draw.line([(a, b), (c, d)], fill=color, width=width)
            continue
        length, (on, off) = math.hypot(c-a, d-b), pattern
        step = 0.
        while step < length:
            end = min(step+on, length)
            draw.line([(a+(c-a)*step/length, b+(d-b)*step/length), (a+(c-a)*end/length, b+(d-b)*end/length)],
                      fill=color, width=width)
            step += on+off


def tier_style(index: int) -> tuple[str, tuple[int, int] | None]:
    return TIER_STYLES[min(index, len(TIER_STYLES)-1)]


def draw_tier_frames(image: Image.Image, frames: list[tuple[float, list[float]]]) -> Image.Image:
    """白芯黑边细框 + 不同线型 + 尺寸标签；白芯黑边组合不与语义或高度配色混淆，框内部不填色。"""
    draw = ImageDraw.Draw(image)
    font = vi._font(max(12, image.width//64))
    for index, (size, box) in enumerate(frames):
        _, pattern = tier_style(index)
        _pattern_rect(draw, box, (0, 0, 0), 3, pattern)
        _pattern_rect(draw, box, (255, 255, 255), 1, pattern)
        draw.text((box[0]+5, box[1]+4), f'{size:g} m', fill='white', font=font, stroke_width=2, stroke_fill='black')
    return image


