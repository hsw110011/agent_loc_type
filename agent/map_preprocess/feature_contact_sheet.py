"""块级总结拼图：每块一行，最右列展示占比、Node 5 初筛标准和结论。

Node 5 会自动为全部退化块输出到 visualization/06_degraded_block_sheets/；也可手动运行：
python -m agent.map_preprocess.feature_contact_sheet --manifest .../state/manifest.json --degraded
python -m agent.map_preprocess.feature_contact_sheet --manifest ... --count 10 --seed 0
python -m agent.map_preprocess.feature_contact_sheet --manifest ... --block-ids E0_B0 E12_B1
"""
from __future__ import annotations

import argparse
import operator
from pathlib import Path
import random
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agent.map_preprocess import output_layout as layout
from agent.map_preprocess.global_config import GlobalPreprocessConfig
from agent.map_preprocess.io import load_manifest, relative_path, resolve_path
from agent.map_preprocess.utils.block_feature_vis import (CLASS_RGB, DEGRADATION_RGB, HEIGHT_ANCHORS, band_colors,
                                                          degradation_kind, height_colormap)
from agent.map_preprocess.utils.block_features import band_labels

SEMANTIC_PANELS = (('rgb', 'RGB 原图'), ('semantic_color', '语义分割 Semantic'))
HEIGHT_PANELS = (('height', '相对路面高度 Height (m)'), ('height_bands', '高度分段 Height bands'))
BAND_LEVELS = ('路面 road', '低矮 low', '高 high')
KIND_ZH = {'both': '语义 + 高度退化', 'semantic': '语义退化', 'height': '高度退化'}
CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-{}.ttc'
DEJAVU = '/usr/share/fonts/truetype/dejavu/DejaVuSans{}.ttf'
RED, BLUE = (200, 0, 0), (0, 70, 160)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for path, index in ((CJK.format('Bold' if bold else 'Regular'), 2), (DEJAVU.format('-Bold' if bold else ''), 0)):
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default()


def _settings(manifest: dict) -> dict:
    """优先读取本次 Node 5 的分段、显示范围和阈值。"""
    summary = manifest.get('extensions', {}).get('block_features') or {}
    defaults = GlobalPreprocessConfig()
    config = {**(manifest.get('config') or {}), **(summary.get('settings') or {})}
    thresholds = {key: summary.get(key, config.get(f'feature_{key}', getattr(defaults, f'feature_{key}')))
                  for key in ('semantic_max_road_pct', 'semantic_min_plant_pct',
                              'height_flat_min_pct', 'height_below_max_pct', 'step_height_m')}
    return {**thresholds, 'semantic_judged': summary.get('semantic_judged', summary.get('height_source') != 'dsm'),
            'has_dsm': summary.get('height_source') == 'dsm',
            'bands_m': tuple(summary.get('height_bands_m') or defaults.feature_dsm_height_bands_m),
            'vis_max_m': summary.get('height_vis_max_m', defaults.feature_height_vis_max_m)}


def _band_name(label: str, index: int, count: int) -> str:
    return f'{BAND_LEVELS[index]}  {label}' if count == 3 else label


def _legend(width: int, font: ImageFont.ImageFont, settings: dict) -> Image.Image:
    image = Image.new('RGB', (width, 40), 'white')
    draw = ImageDraw.Draw(image)
    x = 10
    for name, index in (('道路 road', 0), ('建筑 building', 1), ('植被 plant', 2), ('其他 other', 3)):
        draw.rectangle([x, 11, x+18, 29], fill=CLASS_RGB[index], outline='black')
        draw.text((x+24, 20), name, fill='black', font=font, anchor='lm')
        x += 24+int(draw.textlength(name, font=font))+22
    if not settings['has_dsm']:
        draw.text((x+20, 20), '未提供 DSM：只判语义退化', fill=RED, font=font, anchor='lm')
        return image
    x += 20
    draw.text((x, 20), '路面 0 m', fill='black', font=font, anchor='lm')
    x += int(draw.textlength('路面 0 m', font=font))+8
    bar = height_colormap(np.linspace(HEIGHT_ANCHORS[0, 0], HEIGHT_ANCHORS[-1, 0], 200)[None].repeat(18, 0))
    image.paste(Image.fromarray(bar), (x, 11))
    text = f"≥ {settings['vis_max_m']:g} m"
    draw.text((x+208, 20), text, fill='black', font=font, anchor='lm')
    x += 208+int(draw.textlength(text, font=font))+30
    labels = band_labels(settings['bands_m'])
    for index, (label, color) in enumerate(zip(labels, band_colors(len(labels)))):
        label = _band_name(label, index, len(labels))
        draw.rectangle([x, 11, x+18, 29], fill=color, outline='black')
        draw.text((x+24, 20), label, fill='black', font=font, anchor='lm')
        x += 24+int(draw.textlength(label, font=font))+22
    return image


def _metric_rows(block: dict, settings: dict, labels: list[str]) -> list[tuple]:
    features = block.get('features') or {}
    diagnosis = features.get('degradation') or {}
    classes = features.get('class_pct') or {}
    bands = features.get('height_band_pct') or {}
    semantic_heading = '语义占比 · 两项同时满足' if settings['semantic_judged'] else '语义占比 · 未启用'
    rows = [(semantic_heading, None, '', False)]
    for key, label, symbol, threshold, compare in (
        ('road', '道路', '<', settings['semantic_max_road_pct'], operator.lt),
        ('building', '建筑', '', None, None),
        ('plant', '植被', '>', settings['semantic_min_plant_pct'], operator.gt),
        ('other', '其他', '', None, None),
    ):
        value = diagnosis.get(f'semantic_{key}_pct', classes.get(key))
        active = settings['semantic_judged'] and threshold is not None
        criterion = f'{symbol} {threshold:g}%' if threshold is not None else '—'
        hit = active and value is not None and compare(value, threshold)
        rows.append((label, '—' if value is None else f'{value:.1f}%', criterion, hit))
    if labels:
        rows.append(('高程占比 · 任一项满足', None, '', False))
        reasons = diagnosis.get('height_reasons') or []
        for index, label in enumerate(labels):
            value = bands.get(label)
            criterion = f"> {settings['height_flat_min_pct']:g}%" if index == 0 else '—'
            rows.append((label, '—' if value is None else f'{value:.1f}%', criterion, index == 0 and 'flat' in reasons))
        value = diagnosis.get('height_below_pct')
        rows.append(('阶跃低于路面', '—' if value is None else f'{value:.1f}%',
                     f"> {settings['height_below_max_pct']:g}%", 'below_road' in reasons))
    kind = degradation_kind(block)
    assessed = any(diagnosis.get(key) is not None for key in ('semantic_degraded', 'height_degraded'))
    status = KIND_ZH[kind] if kind else '未触发' if assessed else '无法判断'
    rows.append(('Node 5 初筛', status, '', kind is not None))
    if 'height' in block.get('review_types_available', ()):
        reference = features.get('road_reference') or {}
        stats = features.get('height_stats') or {}
        if reference.get('valid') is False:
            rows.append(('送高程复核', '路面基准无效', '', False))
        elif stats.get('center_step_above'):
            rows.append(('送高程复核', '中心向上阶跃', f"≥ {settings['step_height_m']:g} m", False))
    return rows


def render_sheet(manifest: dict, blocks: list[dict], target: Path, title: str = '', thumb: int = 400) -> Path:
    settings = _settings(manifest)
    panels = SEMANTIC_PANELS+(HEIGHT_PANELS if settings['has_dsm'] else ())
    title_font, head_font, text_font, bold_font = _font(24, True), _font(18, True), _font(15), _font(15, True)
    gap, header = 10, 34
    labels = band_labels(settings['bands_m']) if settings['has_dsm'] else []
    metric_width = max(350, thumb)
    width = len(panels)*(thumb+gap)+metric_width+2*gap
    metrics = [_metric_rows(block, settings, labels) for block in blocks]
    row_height = max(thumb, (1+max(map(len, metrics), default=0))*24)+28+header+2*gap
    title_h = 44 if title else 0
    legend = _legend(width, text_font, settings)
    sheet = Image.new('RGB', (width, title_h+legend.height+row_height*len(blocks)), 'white')
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((10, title_h//2), title, fill='black', font=title_font, anchor='lm')
    sheet.paste(legend, (0, title_h))
    y = title_h+legend.height
    res = manifest['resolution_m_per_px']
    for block, rows in zip(blocks, metrics):
        kind = degradation_kind(block)
        draw.rectangle([0, y, width, y+header-6], fill=(235, 235, 235))
        if kind is not None:
            draw.rectangle([0, y, 10, y+header-6], fill=DEGRADATION_RGB[kind])
        x0, y0, x1, y1 = block['valid_global_bbox']
        draw.text((18, y+(header-6)//2), f"{block['block_id']}   所属边 E{block['edge_id']}   "
                  f"{(x1-x0)*res:.0f} m × {(y1-y0)*res:.0f} m   中心像素 ({block['global_xy'][0]:.0f}, "
                  f"{block['global_xy'][1]:.0f})", fill='black', font=head_font, anchor='lm')
        vis = block.get('feature_vis') or {}
        top = y+header
        for column, (name, label) in enumerate(panels):
            left = column*(thumb+gap)+gap
            draw.text((left, top+10), label, fill=BLUE, font=bold_font, anchor='lm')
            if name in vis and resolve_path(vis[name]).is_file():
                with Image.open(resolve_path(vis[name])) as image:
                    # 贴边块已裁掉图外部分，可能是长方形：等比缩放后居中放入方格
                    image = image.convert('RGB')
                    scale = thumb/max(image.size)
                    image = image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))),
                                         Image.Resampling.LANCZOS)
                    sheet.paste(image, (left+(thumb-image.width)//2, top+24+(thumb-image.height)//2))
        left = len(panels)*(thumb+gap)+gap
        value_x, criterion_x = left+metric_width-116, left+metric_width-8
        draw.text((left+8, top+12), '指标', fill=BLUE, font=bold_font, anchor='lm')
        draw.text((value_x, top+12), '数值 / 结果', fill=BLUE, font=bold_font, anchor='rm')
        draw.text((criterion_x, top+12), '退化标准', fill=BLUE, font=bold_font, anchor='rm')
        for index, (label, value, criterion, hit) in enumerate(rows, start=1):
            middle = top+12+index*24
            section = value is None
            fill = (229, 237, 247) if section else (255, 236, 236) if hit else (247, 248, 250) if index % 2 else 'white'
            draw.rectangle([left, middle-12, left+metric_width, middle+11], fill=fill)
            color = RED if hit else BLUE if section else 'black'
            draw.text((left+8, middle), label, fill=color,
                      font=bold_font if section or hit else text_font, anchor='lm')
            if not section:
                draw.text((value_x if criterion else criterion_x, middle), value, fill=color, font=text_font, anchor='rm')
                draw.text((criterion_x, middle), criterion, fill=color, font=text_font, anchor='rm')
        y += row_height
        draw.line([0, y-gap//2, width, y-gap//2], fill=(200, 200, 200), width=1)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)
    return target


def write_sheets(manifest: dict, blocks: list[dict], directory: Path, prefix: str, *,
                 per_sheet: int = 5, title: str = '') -> list[Path]:
    """按 per_sheet 分页输出；先删除同前缀的旧页，避免残留上次运行的图。"""
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob(f'{prefix}_*.png'):
        old.unlink()
    pages = -(-len(blocks)//per_sheet)
    paths = []
    for page in range(pages):
        chunk = blocks[page*per_sheet:(page+1)*per_sheet]
        first = page*per_sheet+1
        label = f'{title}   第 {page+1}/{pages} 页（第 {first}–{first+len(chunk)-1} 块，共 {len(blocks)} 块）'
        paths.append(render_sheet(manifest, chunk, directory/f'{prefix}_{page+1:02d}.png', label.strip()))
    return paths


def degraded_blocks(manifest: dict) -> list[dict]:
    return [b for b in manifest['blocks'] if ((b.get('features') or {}).get('degradation') or {}).get('degraded')]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--degraded', action='store_true', help='只输出退化块（全部）')
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--block-ids', nargs='*')
    parser.add_argument('--per-sheet', type=int, default=5)
    parser.add_argument('--output-dir', help=f'默认任务目录 visualization/{layout.FIGURES["contact_sheets"]}/')
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    blocks = [b for b in manifest['blocks'] if b.get('features')]
    if not blocks:
        print('manifest 尚无 Node 5 features', file=sys.stderr)
        return 1
    if args.block_ids:
        lookup = {b['block_id']: b for b in blocks}
        missing = [i for i in args.block_ids if i not in lookup]
        if missing:
            print(f'未知块：{missing}', file=sys.stderr)
            return 1
        chosen, tag, title = [lookup[i] for i in args.block_ids], 'selected', '指定块 Selected blocks'
    elif args.degraded:
        chosen, tag, title = degraded_blocks(manifest), 'degraded', '退化块汇总 Degraded blocks'
    else:
        chosen = sorted(random.Random(args.seed).sample(blocks, min(args.count, len(blocks))), key=lambda b: b['block_id'])
        tag, title = f'seed{args.seed}', f'随机抽样 Random sample (seed {args.seed})'
    directory = (resolve_path(args.output_dir) if args.output_dir
                 else layout.figure(layout.task_dir_of_manifest(args.manifest), 'contact_sheets'))
    for path in write_sheets(manifest, chosen, directory, tag, per_sheet=args.per_sheet,
                             title=f"{title} — {Path(manifest['image']['path']).stem}"):
        print(relative_path(path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
