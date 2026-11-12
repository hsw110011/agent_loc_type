"""Node 5 每块处理结果的可视化（PIL + numpy，无 matplotlib 依赖）。

输出 rgb / semantic_color / semantic_overlay，有 DSM 时再输出 height（块内去偏相对高度，米）/ height_bands，
按 max_side 等比缩小；语义图和分段图最近邻缩放，其余 LANCZOS。DSM 网格比 RGB 粗，统一放大到同一宽度。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .block_features import BUILDING, IGNORE, OTHER, PLANT, ROAD, band_index

# road 红（白色叠在灰白路面上看不出，改红；分割工具自身输出仍为白）、building 蓝、plant 绿，other 品红，无效像元深灰。
CLASS_RGB = {ROAD: (255, 0, 0), BUILDING: (0, 0, 255), PLANT: (0, 255, 0), OTHER: (255, 0, 255)}
INVALID_RGB = (40, 40, 40)
# 块内相对高度 0（本块最低）→ 1（>= feature_height_vis_max_m）的顺序色带（近似 viridis）。
HEIGHT_ANCHORS = np.array([[0., 68, 1, 84], [.25, 59, 82, 139], [.5, 33, 145, 140],
                           [.75, 94, 201, 98], [1., 253, 231, 37]])
SEMANTIC_FILES = ('rgb', 'semantic_color', 'semantic_overlay')
HEIGHT_FILES = ('dsm_raw', 'height', 'height_bands')


def band_colors(count: int) -> list[tuple[int, int, int]]:
    """count 个高度分段的离散颜色（色带上等距取色，低→高）。"""
    middle = (np.arange(count)+.5)/count
    return [tuple(int(c) for c in row) for row in height_colormap(middle)]


def height_colormap(values: np.ndarray) -> np.ndarray:
    result = np.empty((*values.shape, 3), np.uint8)
    finite = np.isfinite(values)
    clipped = np.clip(values[finite], HEIGHT_ANCHORS[0, 0], HEIGHT_ANCHORS[-1, 0])
    for channel in range(3):
        result[..., channel][finite] = np.rint(np.interp(clipped, HEIGHT_ANCHORS[:, 0], HEIGHT_ANCHORS[:, channel+1]))
    result[~finite] = INVALID_RGB
    return result


def semantic_color(labels: np.ndarray) -> np.ndarray:
    result = np.empty((*labels.shape, 3), np.uint8)
    result[:] = INVALID_RGB
    for index, color in CLASS_RGB.items():
        result[labels == index] = color
    return result


def _resize(array: np.ndarray, side: int, nearest: bool = False, height: int | None = None) -> Image.Image:
    image = Image.fromarray(array)
    size = (side, height if height is not None else round(image.height*side/image.width))
    if image.size == size:
        return image
    return image.resize(size, Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)


def save_block_visuals(directory: Path, rgb: np.ndarray, labels: np.ndarray, relative_m: np.ndarray | None,
                       bands_m: tuple[float, ...], vis_max_m: float, max_side: int | None,
                       raw_dsm: np.ndarray | None = None) -> dict[str, Path]:
    """relative_m：DSM 网格上的块内相对高度（从 0 开始）（米，无效为 NaN）；None 表示没有 DSM，不输出高度图。
    height_bands 与 utils.block_features.height_band_pct 的分段规则一致。"""
    directory.mkdir(parents=True, exist_ok=True)
    for name in HEIGHT_FILES:  # 上次有 DSM、这次没有时清掉旧高度图
        (directory/f'{name}.png').unlink(missing_ok=True)
    side = rgb.shape[1] if max_side is None else min(max_side, rgb.shape[1])
    tall = round(rgb.shape[0]*side/rgb.shape[1])
    valid = labels != IGNORE
    color = semantic_color(labels)
    overlay = rgb.copy()
    overlay[valid] = np.rint(.5*rgb[valid]+.5*color[valid]).astype(np.uint8)
    overlay[~valid] = INVALID_RGB
    images = {'rgb': _resize(rgb, side), 'semantic_color': _resize(color, side, nearest=True),
              'semantic_overlay': _resize(overlay, side)}
    if raw_dsm is not None:
        # 原始 DSM：按本块有效值的最低→最高拉伸着色（绝对高程，未减路面），无效为深灰。
        values = raw_dsm[np.isfinite(raw_dsm)]
        low, high = (float(values.min()), float(values.max())) if values.size else (0., 1.)
        images['dsm_raw'] = _resize(height_colormap((raw_dsm-low)/max(high-low, 1e-6)), side, nearest=True, height=tall)
    if relative_m is not None:
        finite = np.isfinite(relative_m)
        banded = np.empty((*relative_m.shape, 3), np.uint8)
        banded[:] = INVALID_RGB
        banded[finite] = np.array(band_colors(len(bands_m)+1), np.uint8)[band_index(relative_m[finite], bands_m)]
        # DSM 网格较粗：放大到与 RGB 缩略图同尺寸，最近邻保持像元边界清晰。
        images['height'] = _resize(height_colormap(relative_m/vis_max_m), side, nearest=True, height=tall)
        images['height_bands'] = _resize(banded, side, nearest=True, height=tall)
    paths = {}
    for name, image in images.items():
        paths[name] = directory/f'{name}.png'
        image.save(paths[name])
    return paths


# 全局退化图：块按退化类型半透明填色。
DEGRADATION_RGB = {'both': (230, 0, 0), 'semantic': (255, 150, 0), 'height': (0, 110, 255)}
DEGRADATION_LABEL = {'both': 'semantic + height degraded', 'semantic': 'semantic degraded (little road, mostly plant)',
                     'height': 'height degraded (DSM: flat or below road)'}


def degradation_kind(block: dict) -> str | None:
    d = (block.get('features') or {}).get('degradation') or {}
    semantic, height = bool(d.get('semantic_degraded')), bool(d.get('height_degraded'))
    return 'both' if semantic and height else 'semantic' if semantic else 'height' if height else None


def render_degradation_overview(preview: Image.Image, scale: float, blocks: list[dict], edges: list[dict],
                                output: Path, title: str = '') -> Path:
    """全图块 + 退化标记：道路白线；全部块细白框；退化块按类型半透明填色、粗彩色框并标块 ID。

    先画全部填色、再画全部边框，重叠处不会互相遮挡边框。
    """
    from PIL import ImageFont
    base = preview.convert('RGBA')
    side = max(base.size)
    bold = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    try:
        font, head = ImageFont.truetype(bold, max(12, side//220)), ImageFont.truetype(bold, max(14, side//120))
    except OSError:
        font = head = ImageFont.load_default()
    line_width = max(2, side//900)
    rect = lambda b: [v*scale for v in b['valid_global_bbox']]
    kinds = [degradation_kind(b) for b in blocks]
    fill = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(fill)
    for edge in edges:
        draw.line([(x*scale, y*scale) for x, y in edge['polyline_xy']], fill=(255, 255, 255, 170), width=2*line_width)
    for block, kind in zip(blocks, kinds):
        if kind is not None:
            draw.rectangle(rect(block), fill=DEGRADATION_RGB[kind]+(90,))
    image = Image.alpha_composite(base, fill)
    lines = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(lines)
    for block, kind in zip(blocks, kinds):
        if kind is None:
            draw.rectangle(rect(block), outline=(255, 255, 255, 230), width=line_width)
    for block, kind in zip(blocks, kinds):
        if kind is not None:
            draw.rectangle(rect(block), outline=DEGRADATION_RGB[kind]+(255,), width=2*line_width)
    for block, kind in zip(blocks, kinds):
        if kind is not None:
            x0, y0, x1, y1 = rect(block)
            draw.text(((x0+x1)/2, (y0+y1)/2), block['block_id'], fill=(255, 255, 255, 255), font=font, anchor='mm',
                      stroke_width=2, stroke_fill=(0, 0, 0, 255))
    image = Image.alpha_composite(image, lines).convert('RGB')
    counts = {k: kinds.count(k) for k in DEGRADATION_RGB}
    items = [(DEGRADATION_RGB[k], f'{DEGRADATION_LABEL[k]}: {counts[k]}') for k in DEGRADATION_RGB]
    items.append(((255, 255, 255), f'not degraded (white box): {kinds.count(None)}'))
    band = head.size+font.size+30
    canvas = Image.new('RGB', (image.width, image.height+band), 'white')
    canvas.paste(image, (0, band))
    legend = ImageDraw.Draw(canvas)
    degraded = len(blocks)-kinds.count(None)
    legend.text((10, 6), f'{title}   {degraded}/{len(blocks)} blocks degraded'.strip(), fill='black', font=head)
    x, y = 10, head.size+16
    for color, text in items:
        legend.rectangle([x, y, x+font.size, y+font.size], fill=color, outline='black')
        legend.text((x+font.size+6, y-2), text, fill='black', font=font)
        x += font.size+16+int(legend.textlength(text, font=font))+20
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output
