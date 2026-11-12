"""Node 5 指标的中英文定义：表格导出、人工检查和后续 VLM 提示词共用这一份。

列名 `column` 是扁平化后的字段名（CSV/表格/提示词中使用）；`path` 是 manifest
blocks[].features 中的位置。所有指标只描述当前块（默认 64m × 64m）在原图内的整块（超出原图处直接裁掉），不看道路两侧，不与邻块或全图比较。
"""
from __future__ import annotations

from typing import Any

from .global_config import GlobalPreprocessConfig
from .utils.block_features import band_labels

CLASS_ZH = {'road': '道路', 'building': '建筑', 'plant': '植被', 'other': '其他（低置信度）'}
_DEFAULTS = GlobalPreprocessConfig()
EDGES = tuple(_DEFAULTS.feature_dsm_height_bands_m)
BANDS = tuple(band_labels(EDGES))
BAND_ZH = dict(zip(BANDS, ('路面/地面', '低矮物', '建筑或树'))) if len(BANDS) == 3 else {band: band for band in BANDS}
FLAT = _DEFAULTS.feature_height_flat_min_pct
BELOW = _DEFAULTS.feature_height_below_max_pct
MAX_ROAD = _DEFAULTS.feature_semantic_max_road_pct
MIN_PLANT = _DEFAULTS.feature_semantic_min_plant_pct
BAND_LIST = '/'.join(BANDS)


def _definitions() -> list[dict[str, Any]]:
    items = []
    for name, zh in CLASS_ZH.items():
        items.append({
            'column': f'class_{name}', 'path': ('class_pct', name), 'group': 'semantic', 'unit': '%', 'range': '0-100',
            'name_en': f'Block {name} ratio', 'name_zh': f'整块{zh}占比',
            'definition_en': f'Share of valid pixels in the whole block labelled {name}. The four class ratios sum to 100.'
                             + (' "other" = pixels whose top-class confidence is below 0.5.' if name == 'other' else ''),
            'definition_zh': f'整块全部有效像元中{zh}像元的比例，四类合计 100。'
                             + ('“其他”指模型最高类置信度低于 0.5 的像元。' if name == 'other' else '')})
    for band in BANDS:
        items.append({
            'column': f'band_{band}', 'path': ('height_band_pct', band), 'group': 'height', 'unit': '%', 'range': '0-100',
            'name_en': f'Height band {band}', 'name_zh': f'高度分段 {band}（{BAND_ZH[band]}）',
            'definition_en': f'Share of DSM pixels (at or above road level) whose height above the block-centre road (metres) falls in {band}. '
                             f'The {len(BANDS)} bands sum to 100. Empty when no DSM is given.',
            'definition_zh': f'DSM 减去块中心路面高程后的相对高度（米）落在 {band} 的像元比例，只统计 0 m 及以上像元，{len(BANDS)} 段合计 100。'
                             '未提供 DSM 时为空。'})
    items += [
        {'column': 'base_dsm_m', 'path': ('height_stats', 'base_dsm_m'), 'group': 'height', 'unit': 'm', 'range': 'DSM',
         'name_en': 'Road reference (DSM)', 'name_zh': '0 点（中心路面高程）',
         'definition_en': 'Median valid DSM of road pixels in a centred 5 m square when the centre has valid road DSM, otherwise in a 10 m square. Empty if no reference is available.',
         'definition_zh': '中心有有效路面高程时，取中心 5 m × 5 m 内有效 road 像元的 DSM 中位数；否则扩大到 10 m × 10 m。找不到基准时为空。'},
        {'column': 'below_base_pct', 'path': ('height_stats', 'below_base_pct'), 'group': 'height', 'unit': '%', 'range': '0-100',
         'name_en': 'Below road level (all)', 'name_zh': '低于路面像元占比（合计）',
         'definition_en': 'Share of valid DSM pixels more than {0:g} m below the road reference; not counted in the height bands '
                          '(-{0:g} to 0 m is treated as road noise and counted as 0 m).'.format(_DEFAULTS.feature_below_base_tolerance_m),
         'definition_zh': '低于 0 点（路面）超过 {0:g} m 的有效 DSM 像元比例；这些像元不算有效像元，不参与高度分段'
                          '（-{0:g}~0 m 视为路面噪声，计为 0 m）。'.format(_DEFAULTS.feature_below_base_tolerance_m)},
                {'column': 'step_below_pct', 'path': ('height_stats', 'step_below_pct'), 'group': 'height', 'unit': '%', 'range': '0-100',
         'name_en': 'Step below road', 'name_zh': '阶跃低于路面占比',
         'definition_en': 'Share of DSM pixels below the road that are only reached across a step (>= {0:g} m height change within '
                          '{1:g} m), e.g. embankment slopes, bridges, ditches: unreasonably low. Used by the below_road rule.'
                          .format(_DEFAULTS.feature_step_height_m, _DEFAULTS.feature_step_run_m),
         'definition_zh': '低于路面、且必须越过台阶（{1:g} m 内高差 >= {0:g} m）才能从路面到达的像元比例，如路堤边坡、桥下、坎、沟：'
                          '不合理的低。“低于路面”退化规则只看此项。'.format(_DEFAULTS.feature_step_height_m, _DEFAULTS.feature_step_run_m)},
        {'column': 'slope_below_pct', 'path': ('height_stats', 'slope_below_pct'), 'group': 'height', 'unit': '%', 'range': '0-100',
         'name_en': 'Slope below road', 'name_zh': '缓坡低于路面占比',
         'definition_en': 'Share of DSM pixels below the road that connect to the road surface without a step (continuous slope): '
                          'treated as normal ground and counted as 0 m.',
         'definition_zh': '低于路面、但与路面连续相接（缓坡、无台阶）的像元比例：视为正常地面，计为 0 m（0-1m 段）。'},
{'column': 'terrain_slope_pct', 'path': ('height_stats', 'terrain_slope_pct'), 'group': 'height', 'unit': '%', 'range': '>=0',
         'name_en': 'Terrain slope', 'name_zh': '地形坡度',
         'definition_en': 'Slope of the fitted block ground plane (rise/run, %); only when feature_dsm_detrend is on, else empty.',
         'definition_zh': '块内拟合地面平面的坡度（高差/水平距离，%）；仅 feature_dsm_detrend 开启时有值，默认为空。'},
        {'column': 'p95_m', 'path': ('height_stats', 'p95_m'), 'group': 'height', 'unit': 'm', 'range': '>=0',
         'name_en': 'Relative height P95', 'name_zh': '相对高度 P95',
         'definition_en': '95th percentile of de-biased height in the block (metres).',
         'definition_zh': '本块去偏后相对高度的 95% 分位数（米）。'},
        {'column': 'dsm_nodata_pct', 'path': ('height_stats', 'nodata_pct'), 'group': 'height', 'unit': '%', 'range': '0-100',
         'name_en': 'DSM no-data share', 'name_zh': 'DSM 无效像元占比',
         'definition_en': 'Share of block pixels without valid DSM values.', 'definition_zh': '本块内没有有效 DSM 值的像元比例。'}]
    items += [
        {'column': 'semantic_degraded', 'path': ('degradation', 'semantic_degraded'), 'group': 'degradation', 'unit': 'bool', 'range': 'yes/no',
         'name_en': 'Semantic degraded', 'name_zh': '语义退化',
         'definition_en': f'Judged only without DSM (empty when a DSM is given). Yes when road < {MAX_ROAD:g}% and plant > {MIN_PLANT:g}% of the block: vegetation covers the block '
                          'and almost no road is visible (e.g. tree canopy, field), so semantic matching has little to anchor on.',
         'definition_zh': f'仅无 DSM 时判断（有 DSM 时为空）。整块道路占比 < {MAX_ROAD:g}% 且植被占比 > {MIN_PLANT:g}% 时为是：植被覆盖、几乎看不到道路'
                          '（如树冠、农田），语义匹配缺少可区分的参照。'},
        {'column': 'height_degraded', 'path': ('degradation', 'height_degraded'), 'group': 'degradation', 'unit': 'bool', 'range': 'yes/no',
         'name_en': 'Height degraded', 'name_zh': '高度退化',
         'definition_en': f'Yes (needs DSM) when the {BANDS[0]} band takes more than {FLAT:g}% (block almost as flat as the road), '
                          f'or more than {BELOW:g}% of DSM pixels are below road level across a step (embankment / viaduct; continuous slopes do not count). A large share of '
                          'tall objects (>3 m) is not degraded. Empty (not judged) when no DSM is given.',
         'definition_zh': f'需要 DSM：{BANDS[0]} 段占比超过 {FLAT:g}%（整块几乎和路面一样平），或阶跃低于路面的像元超过 {BELOW:g}%'
                          '（路堤/高架，四周隔着台阶比路低；缓坡低于不算）时为是；高段（>2m）占比大不算退化。未提供 DSM 时为空（不判断）。'},
        {'column': 'height_reasons', 'path': ('degradation', 'height_reasons'), 'group': 'degradation', 'unit': 'name', 'range': 'flat/below_road',
         'name_en': 'Height degradation reasons', 'name_zh': '高度退化原因',
         'definition_en': 'flat = lowest band above the limit; below_road = too many pixels below road level across a step.',
         'definition_zh': 'flat：最低段超过阈值；below_road：阶跃低于路面的像元过多（缓坡低于不算）。'},
        {'column': 'degraded', 'path': ('degradation', 'degraded'), 'group': 'degradation', 'unit': 'bool', 'range': 'yes/no',
         'name_en': 'Send to VLM', 'name_zh': '送大模型复核',
         'definition_en': 'Height degraded when a DSM is given, otherwise semantic degraded. All such blocks go to the VLM; '
                          'whether the road is drivable is not considered.',
         'definition_zh': '有 DSM 时等于高度退化，无 DSM 时等于语义退化；这些块全部送大模型复核，不区分车辆能否通行。'},
    ]
    return items


METRICS = _definitions()


def flatten(block: dict) -> dict[str, Any]:
    """把一个块的 features 展开成 {column: value}。"""
    features = block['features']
    row: dict[str, Any] = {'block_id': block['block_id'], 'edge_id': block['edge_id']}
    for item in METRICS:
        value: Any = features
        for key in item['path']:
            value = None if value is None else value.get(key)
        row[item['column']] = ';'.join(value) if isinstance(value, list) else value
    return row
