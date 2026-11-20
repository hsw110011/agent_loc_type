"""Node 5 纯 numpy 指标：块内语义占比与 DSM 块内去偏相对高度分段占比；每块独立，不与邻块或全图比较。

不加载模型，不读写文件；输入为单块的标签/DSM 数组。
"""
from __future__ import annotations

import numpy as np

CLASSES = ('road', 'building', 'plant', 'other')
ROAD, BUILDING, PLANT, OTHER, IGNORE = 0, 1, 2, 3, 255


def semantic_labels(mask: np.ndarray, confidence: np.ndarray, min_confidence: float) -> np.ndarray:
    """最高类置信度低于阈值的有效像素记为 other，模型直接输出的 other 和无效像素 255 保持不变。"""
    labels = mask.astype(np.uint8, copy=True)
    labels[(labels != IGNORE) & (confidence < min_confidence)] = OTHER
    return labels


def class_pct(labels: np.ndarray, region: np.ndarray | None = None) -> dict[str, float] | None:
    selected = labels != IGNORE
    if region is not None:
        selected &= region
    total = int(selected.sum())
    if not total:
        return None
    counts = np.bincount(labels[selected], minlength=4)[:4]
    return {name: float(count*100/total) for name, count in zip(CLASSES, counts)}


def band_labels(edges_m: tuple[float, ...]) -> list[str]:
    """相对路面高度分段名（米）：边界 (1, 3) → ['0-1m', '1-3m', '>3m']（0 m 以下不计入）。"""
    edges = [0., *edges_m]
    return [*(f'{a:g}-{b:g}m' for a, b in zip(edges, edges[1:])), f'>{edges[-1]:g}m']


def band_index(height_m: np.ndarray, edges_m: tuple[float, ...]) -> np.ndarray:
    """每个像元（高度 >= 0）所属分段编号：左闭右开，[0, e0) 为 0 段。"""
    return np.searchsorted(np.asarray(edges_m, float), height_m, side='right')


def height_band_pct(height_m: np.ndarray, edges_m: tuple[float, ...]) -> dict[str, float] | None:
    """相对路面高度（米，>= 0；负值/NaN 不计）的分段占比（%），总和为 100；无有效像元为 None。"""
    values = height_m[np.isfinite(height_m) & (height_m >= 0)]
    if not values.size:
        return None
    counts = np.bincount(band_index(values, edges_m), minlength=len(edges_m)+1)
    return {name: float(count*100/values.size) for name, count in zip(band_labels(edges_m), counts)}


def ground_plane(dsm: np.ndarray, finite: np.ndarray, cells: int, percentile: float = 5.,
                 reject_m: float = 1., rounds: int = 3) -> np.ndarray | None:
    """块内地形坡面：分 cells×cells 格，每格取 percentile 分位数作地面采样点，最小二乘拟合平面。
    初值用最低 25% 的采样点，之后每轮保留不高出平面 reject_m 的点重新拟合（剔除整格被建筑/树冠覆盖的点）。
    采样点不足 3 个返回 None。"""
    h, w = dsm.shape
    samples = []
    for i in range(cells):
        for j in range(cells):
            y0, y1, x0, x1 = i*h//cells, (i+1)*h//cells, j*w//cells, (j+1)*w//cells
            mask = finite[y0:y1, x0:x1]
            if mask.sum() >= max(4, mask.size//10):
                samples.append(((x0+x1-1)/2, (y0+y1-1)/2, np.percentile(dsm[y0:y1, x0:x1][mask], percentile)))
    if len(samples) < 3:
        return None
    points = np.array(samples)
    # 初值只用最低的 25% 采样点：块内多数被建筑/树冠覆盖时，避免平面被抬到屋顶/冠层。
    keep = points[:, 2] <= np.percentile(points[:, 2], 25)
    if keep.sum() < 3:
        keep = np.argsort(points[:, 2]) < 3
    for _ in range(rounds):
        design = np.c_[points[keep, :2], np.ones(keep.sum())]
        coef = np.linalg.lstsq(design, points[keep, 2], rcond=None)[0]
        residual = points[:, 2]-np.c_[points[:, :2], np.ones(len(points))]@coef
        updated = residual <= reject_m
        if updated.sum() < 3 or (updated == keep).all():
            break
        keep = updated
    yy, xx = np.mgrid[0:h, 0:w]
    return (coef[0]*xx+coef[1]*yy+coef[2]).astype(np.float32)


def step_mask(dsm: np.ndarray, m_per_px: float, step_m: float, run_m: float) -> np.ndarray:
    """台阶像元：水平 run_m 窗口内（max − min）高差 >= step_m；无效像元也视为台阶（不可连通）。

    连续的坡（例如 10%~30% 的路面/地形坡度）在 1 m 内高差远小于 0.5 m，不会成为台阶；
    路堤边坡、桥边、坎、墙、沟沿等直接阶跃会成为台阶。
    """
    import cv2
    finite = np.isfinite(dsm)
    if not finite.any():
        return ~finite
    k = max(3, int(round(run_m/m_per_px)) | 1)
    kernel = np.ones((k, k), np.uint8)
    filled_hi = np.where(finite, dsm, -np.inf).astype(np.float32)
    filled_lo = np.where(finite, dsm, np.inf).astype(np.float32)
    local_max = cv2.dilate(filled_hi, kernel)
    local_min = cv2.erode(filled_lo, kernel)
    difference = np.zeros_like(dsm)
    np.subtract(local_max, local_min, out=difference, where=finite)
    return ~finite | (difference >= step_m)


def continuous_region(dsm: np.ndarray, seed: np.ndarray, m_per_px: float, step_m: float, run_m: float) -> np.ndarray:
    """从 seed（块中心路面）出发、不越过台阶能连通到的像元（8 连通）。

    seed 全部落在台阶上时，逐步放大 seed 范围找最近的非台阶像元。
    """
    import cv2
    walls = step_mask(dsm, m_per_px, step_m, run_m)
    open_ = (~walls).astype(np.uint8)
    count, labels = cv2.connectedComponents(open_, connectivity=8)
    grow = seed.copy()
    for _ in range(6):
        hits = np.unique(labels[grow & (open_ > 0)])
        hits = hits[hits > 0]
        if hits.size:
            return np.isin(labels, hits)
        grow = cv2.dilate(grow.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return np.zeros_like(seed)


def road_reference(dsm: np.ndarray, labels: np.ndarray, center_rc: tuple[float, float],
                   m_per_px: float, settings, cleaned_road: np.ndarray | None) -> tuple[np.ndarray | None, dict]:
    """按 5、10、20 m 方窗验证路面样本，每档优先语义道路。"""
    row, col = center_rc
    ir, ic = int(np.floor(row+.5)), int(np.floor(col+.5))
    h, w = dsm.shape
    yy, xx = np.ogrid[:h, :w]
    dy, dx = (yy-row)*m_per_px, (xx-col)*m_per_px
    road = labels == ROAD
    finite = np.isfinite(dsm)
    windows = []
    for size_m, source in ((5, 'center_road'), (10, 'expanded_road'), (20, 'expanded_road')):
        half = size_m/2
        window = (dy >= -half) & (dy < half) & (dx >= -half) & (dx < half)
        windows.append((size_m, source, window & road))
        if cleaned_road is not None:
            windows.append((size_m, 'cleaned_road', window & cleaned_road))

    attempts = []
    for size_m, source, mask in windows:
        seed = mask & finite
        sample_count, valid_count = int(mask.sum()), int(seed.sum())
        valid_pct = 100.*valid_count/sample_count if sample_count else None
        iqr = None
        reasons = []
        if not sample_count:
            reasons.append('no_road_samples')
        else:
            if not valid_count:
                reasons.append('road_dsm_no_data')
            if valid_count < settings.feature_road_ref_min_samples:
                reasons.append('too_few_valid_samples')
            if valid_pct < settings.feature_road_ref_min_valid_pct:
                reasons.append('low_valid_sample_coverage')
            if valid_count >= settings.feature_road_ref_min_samples:
                q25, q75 = np.percentile(dsm[seed], [25, 75])
                iqr = float(q75-q25)
                if iqr > settings.feature_road_ref_max_iqr_m:
                    reasons.append('unstable_road_elevation')
        attempts.append({'source': source, 'window_m': size_m, 'sample_count': sample_count, 'valid_count': valid_count,
                         'valid_pct': valid_pct, 'iqr_m': iqr, 'reasons': reasons})
        if not reasons:
            at_center = 0 <= ir < h and 0 <= ic < w and seed[ir, ic]
            offset_m = 0. if at_center else float(np.hypot(dy, dx)[seed].min())
            return seed, {'valid': True, 'source': source, 'offset_m': offset_m, 'window_m': size_m, 'attempts': attempts}
    return None, {'valid': False, 'source': None, 'offset_m': None, 'window_m': None, 'attempts': attempts}


def dsm_height_features(dsm: np.ndarray, labels: np.ndarray, settings, m_per_px: float = 1.,
                        center_rc: tuple[float, float] | None = None,
                        cleaned_road: np.ndarray | None = None) -> dict:
    """DSM（米）→ 相对块中心路面的高度及其分段占比；labels 已缩放到 DSM 块网格，m_per_px 为该网格分辨率。

    0 点 = 语义路面的 DSM 中位数（见 road_reference）：块中心（道路拓扑线上的点，center_rc 为其在块网格中的行列）
    依次尝试中心 5、10、20 m 方窗，每档语义道路失败后尝试清洗道路。
    无有效路面基准时跳过相对高度统计，保留取样诊断。
    相对高度 = DSM − 0 点，单位米、不缩放；[-feature_below_base_tolerance_m, 0) 的路面噪声视为 0 m。

    低于路面（< −tolerance）的像元按高度变化是否连续分两类（feature_step_height_m / feature_step_run_m）：
      - 缓坡低于：从块中心路面出发不越过台阶（run_m 内高差 >= step_m）即可连通，视为正常地面，计为 0 m（最低段，默认 0-0.5m）；
      - 阶跃低于：隔着台阶才到达（路堤边坡、桥下、坎、沟），为不合理的低，不参与分段，记入 step_below_pct。
    below_base_pct = 两类合计（信息）；高度退化的“低于路面”只看 step_below_pct。
    可选 feature_dsm_detrend 先减去块内地形平面。下划线键只用于可视化。
    """
    from .dsm import valid_dsm_elevation

    dsm = np.where(valid_dsm_elevation(dsm), dsm, np.nan)
    h, w = dsm.shape
    center = center_rc if center_rc is not None else ((h-1)/2, (w-1)/2)
    seed, quality = road_reference(dsm, labels, center, m_per_px, settings, cleaned_road)
    result = {'height_band_pct': None, 'height_stats': None, 'road_reference': quality, '_relative': None}
    if seed is None:
        return result
    reference = quality['source']
    inside = np.ones(dsm.shape, dtype=bool) if reference == 'cleaned_road' else labels != IGNORE
    finite = np.isfinite(dsm) & inside
    nodata_pct = float(np.count_nonzero(inside & ~finite)*100/inside.sum())
    plane = ground_plane(dsm, finite, settings.feature_dsm_ground_cells) if settings.feature_dsm_detrend else None
    slope, surface = None, dsm
    if plane is not None:
        gy, gx = np.gradient(plane)
        slope = float(np.hypot(gx.mean(), gy.mean())/m_per_px*100)
        surface = dsm-plane
    base = float(np.median(surface[seed]))
    relative = np.where(finite, surface-base, np.nan).astype(np.float32)
    # 0 m 以下不算有效像元；0 点是路面中位数，与路面同高的像元约一半会因 DSM 噪声略低于 0，
    # 因此 [-tolerance, 0) 视为 0 m（计入最低段），低于 -tolerance 才剔除。
    below = finite & (relative < -settings.feature_below_base_tolerance_m)
    region = continuous_region(surface, seed, m_per_px,
                               settings.feature_step_height_m, settings.feature_step_run_m)
    # 只检查道路拓扑中心。路侧高物不应触发道路中央遮挡提示。
    ir, ic = (int(np.floor(v+.5)) for v in center)
    center_step_above = False
    if (0 <= ir < h and 0 <= ic < w and np.isfinite(surface[ir, ic])
            and surface[ir, ic]-base >= settings.feature_step_height_m and not region[ir, ic]):
        # 核实路面样本到中心存在实际向上台阶，避免仅因 NoData 断开触发。
        from scipy.ndimage import label

        low_surface = np.isfinite(surface) & (surface < base+settings.feature_step_height_m)
        components, _ = label(np.isfinite(surface), structure=np.ones((3, 3), dtype=bool))
        center_step_above = bool(np.any(seed & low_surface & (components == components[ir, ic])))
    slope_below = below & region
    step_below = below & ~region
    relative[step_below] = np.nan
    relative = np.where(np.isfinite(relative), np.maximum(relative, 0.), np.nan).astype(np.float32)
    counted = np.isfinite(relative)
    raw = dsm[finite]
    stats = {'dsm_min_m': float(raw.min()), 'dsm_max_m': float(raw.max()), 'base_ref': reference, 'base_dsm_m': base if plane is None else float(np.median(dsm[seed])),
             'base_offset_m': quality['offset_m'], 'center_step_above': center_step_above,
             'below_base_pct': float(np.count_nonzero(below)*100/finite.sum()),
             'step_below_pct': float(np.count_nonzero(step_below)*100/finite.sum()),
             'slope_below_pct': float(np.count_nonzero(slope_below)*100/finite.sum()),
             'terrain_slope_pct': slope, 'nodata_pct': nodata_pct,
             'median_m': None, 'p95_m': None, 'max_m': None}
    if counted.any():
        p50, p95 = np.percentile(relative[counted], [50, 95])
        stats.update(median_m=float(p50), p95_m=float(p95), max_m=float(np.nanmax(relative)))
    result.update({'height_band_pct': height_band_pct(relative, tuple(settings.feature_dsm_height_bands_m)),
                   'height_stats': stats, '_relative': relative,
                   '_step_below': step_below, '_slope_below': slope_below})
    return result
