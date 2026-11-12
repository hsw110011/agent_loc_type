"""Node 6 VLM 复核输入：以块中心为中心、边长 = 当前块尺寸的窗口，渲染 RGB / 语义 / 相对高度图，
并计算窗口内的语义与高度统计、相对高度网格。

内部计算使用统一分析网格；发送前统一裁掉地图外部分，各图尺寸相同、像素一一对应，可以为长方形。
送入 VLM 的图不画图例（避免改变图像尺寸），颜色含义由 block_review 的 prompt 说明；add_legend 仅用于全图总览。
不加载模型、不调用 VLM。
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import block_features as bf
from .block_feature_vis import height_colormap, semantic_color

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
# Node 6 局部 DSM 伪彩色：固定物理范围上的 Viridis 连续色标（与 block_feature_vis.HEIGHT_ANCHORS 同一组锚点），无数据白色
DSM_NODATA_RGB = (255, 255, 255)
DSM_TICKS = ((0., '深紫色'), (.25, '深蓝色'), (.5, '蓝绿色/青绿色'), (.75, '绿色'), (1., '亮黄色'))


def review_window(block: dict, extent_m: float, resolution: float) -> list[int]:
    """以块中心（global_xy）为中心、边长 extent_m 的窗口（原图像素，可超出原图）。"""
    side = round(extent_m/resolution)
    x, y = block['global_xy']
    x0, y0 = round(x-side/2), round(y-side/2)
    return [x0, y0, x0+side, y0+side]


def _clip(bbox: list[int], width: int, height: int) -> tuple[list[int], tuple[int, int]] | tuple[None, None]:
    x0, y0, x1, y1 = bbox
    clipped = [max(0, x0), max(0, y0), min(width, x1), min(height, y1)]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None, None
    return clipped, (clipped[0]-x0, clipped[1]-y0)


def _image_box(clipped: list[int], window: list[int], out_px: int) -> list[int]:
    scale = out_px/(window[2]-window[0])
    a, b = round((clipped[0]-window[0])*scale), round((clipped[1]-window[1])*scale)
    c, d = round((clipped[2]-window[0])*scale), round((clipped[3]-window[1])*scale)
    # 非空地图范围即使小于一个分析像素，也至少保留一列/一行。
    a, b = min(a, out_px-1), min(b, out_px-1)
    c, d = max(a+1, min(c, out_px)), max(b+1, min(d, out_px))
    return [a, b, c, d]


def valid_crop(window: list[int], image_size: tuple[int, int], out_px: int) -> tuple[list[int], list[int]] | None:
    """返回地图内原图 bbox 和对应分析网格裁剪框；不保留地图外画布。"""
    clipped, _ = _clip(window, *image_size)
    if clipped is None:
        return None
    box = _image_box(clipped, window, out_px)
    return (clipped, box) if box[2] > box[0] and box[3] > box[1] else None


def _place(array: np.ndarray, clipped: list[int], window: list[int], out_px: int, fill, nearest: bool) -> np.ndarray:
    """内部统一分析网格；发送前用同一裁剪框移除地图外填充。"""
    a, b, c, d = _image_box(clipped, window, out_px)
    shape = (out_px, out_px, *array.shape[2:])
    canvas = np.full(shape, fill, array.dtype)
    if c > a and d > b:
        resized = cv2.resize(array, (c-a, d-b), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
        canvas[b:d, a:c] = resized
    return canvas


def read_rgb(manifest: dict, window: list[int], out_px: int) -> np.ndarray:
    from ..io import open_image
    width, height = manifest['image']['width'], manifest['image']['height']
    clipped, _ = _clip(window, width, height)
    if clipped is None:
        return np.zeros((out_px, out_px, 3), np.uint8)
    with open_image(manifest['image']['path'], manifest['config']['pillow_max_image_pixels']) as source:
        rgb = source.read(clipped)
        valid = source.valid_mask(clipped)
    rgb = np.where(valid[..., None], rgb, 0).astype(np.uint8)
    return _place(rgb, clipped, window, out_px, 0, nearest=False)


def crop_labels(labels: np.ndarray, image_size: tuple[int, int], window: list[int], out_px: int) -> np.ndarray:
    """从整图语义（工作网格）裁出窗口并最近邻缩放到分析网格；超出原图为 IGNORE。"""
    width, height = image_size
    clipped, _ = _clip(window, width, height)
    if clipped is None:
        return np.full((out_px, out_px), bf.IGNORE, np.uint8)
    sx, sy = labels.shape[1]/width, labels.shape[0]/height
    a, b = int(math.floor(clipped[0]*sx)), int(math.floor(clipped[1]*sy))
    c, d = max(a+1, int(math.ceil(clipped[2]*sx))), max(b+1, int(math.ceil(clipped[3]*sy)))
    return _place(np.ascontiguousarray(labels[b:d, a:c]), clipped, window, out_px, bf.IGNORE, nearest=True)


def read_dsm(dsm: dict, image_size: tuple[int, int], window: list[int], out_px: int) -> np.ndarray:
    """窗口 DSM（米，float32），最近邻缩放到分析网格，避免 NaN 扩散；超出原图或无数据为 NaN。"""
    from .dsm import read_block_dsm
    clipped, _ = _clip(window, *image_size)
    if clipped is None:
        return np.full((out_px, out_px), np.nan, np.float32)
    return _place(read_block_dsm(dsm, clipped), clipped, window, out_px, np.nan, nearest=True)


def _r(value):
    return None if value is None else round(float(value), 1)


def semantic_stats(labels: np.ndarray) -> dict | None:
    """窗口内语义各类别占比；只给数值，不含退化判断。"""
    pct = bf.class_pct(labels)
    return {f'{k}_pct': _r(v) for k, v in pct.items()} if pct else None


def dsm_reference(dsm: np.ndarray, percentile: float) -> tuple[float | None, float]:
    """参考高程 H_ref：有效 DSM 的百分位（米）及有效像元占比（%）；没有有效值返回 None。只是着色基准，不是路面高程。"""
    finite = np.isfinite(dsm)
    if not finite.any():
        return None, 0.
    return float(np.percentile(dsm[finite], percentile)), _r(finite.mean()*100)


def dsm_color(relative: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    """DSM − H_ref（米）按固定范围线性映射到 Viridis，超出范围取端点颜色；不加阴影，同一高差在任何块都是同一颜色。"""
    low, high = value_range
    image = height_colormap((relative-low)/(high-low))
    image[~np.isfinite(relative)] = DSM_NODATA_RGB
    return image


def dsm_legend(value_range: tuple[float, float]) -> list[dict]:
    """色标刻度（米）及颜色名，记录在 metadata，不发给模型。"""
    low, high = value_range
    colors = height_colormap(np.array([t for t, _ in DSM_TICKS]))
    return [{'value_m': round(low+(high-low)*t, 3), 'color': name, 'rgb': [int(c) for c in colors[k]]}
            for k, (t, name) in enumerate(DSM_TICKS)]


def dsm_stats(relative: np.ndarray, value_range: tuple[float, float]) -> dict:
    """局部 DSM 图统计：无数据、低于/高于色标范围（饱和）的像元占比（%），以及相对高差分位数（米）。"""
    finite = np.isfinite(relative)
    values = relative[finite]
    stats = {'nodata_pct': _r((~finite).mean()*100),
             'below_range_pct': _r(np.count_nonzero(values < value_range[0])*100/relative.size),
             'above_range_pct': _r(np.count_nonzero(values > value_range[1])*100/relative.size)}
    for q in (5, 50, 95):
        stats[f'p{q}_m'] = round(float(np.percentile(values, q)), 2) if values.size else None
    return stats


def _font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def _dashed_rect(draw: ImageDraw.ImageDraw, box: list[float], color, width: int, dash: int) -> None:
    x0, y0, x1, y1 = box
    for (a, b), (c, d) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = math.hypot(c-a, d-b)
        steps = max(1, int(length//(2*dash)))
        for k in range(steps):
            t0, t1 = 2*k*dash/length, min(1., (2*k+1)*dash/length)
            draw.line([(a+(c-a)*t0, b+(d-b)*t0), (a+(c-a)*t1, b+(d-b)*t1)], fill=color, width=width)


def draw_center(image: Image.Image, center_xy: tuple[float, float] | None = None) -> Image.Image:
    """只叠加区域中心十字（黑边白芯），不画道路线；裁剪后区域中心不必在图像中间。"""
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    cx, cy = center_xy if center_xy is not None else (image.width/2, image.height/2)
    r = max(6, min(image.size)//64)
    for width, color in ((5, (0, 0, 0)), (2, (255, 255, 255))):
        draw.line([(cx-r, cy), (cx+r, cy)], fill=color, width=width)
        draw.line([(cx, cy-r), (cx, cy+r)], fill=color, width=width)
    return image


def add_legend(image: Image.Image, items: list[tuple[tuple[int, int, int], str]], title: str) -> Image.Image:
    font = _font(max(12, image.width//52))
    measure = ImageDraw.Draw(image)
    # 图例项按宽度自动换行，避免超出图像被截断
    rows, row, x = [], [], 8
    for color, text in items:
        width = font.size+14+int(measure.textlength(text, font=font))
        if row and x+width > image.width-8:
            rows.append(row)
            row, x = [], 8
        row.append((color, text, x))
        x += width
    rows.append(row)
    line = font.size+8
    canvas = Image.new('RGB', (image.width, image.height+line*(len(rows)+1)+8), 'white')
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, image.height+4), title, fill='black', font=font)
    for k, row in enumerate(rows):
        y = image.height+line*(k+1)+6
        for color, text, x in row:
            draw.rectangle([x, y, x+font.size, y+font.size], fill=color, outline='black')
            draw.text((x+font.size+4, y-1), text, fill='black', font=font)
    return canvas


def save_png(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
