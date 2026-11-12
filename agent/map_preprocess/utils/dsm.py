"""DSM（数字表面模型，单位米）与 RGB 的对齐检查和按块读取。

道路拓扑和块只在 RGB 上计算；DSM 不参与拓扑，只按块的地理范围裁剪出对应高程。
两种对齐方式：
- geo：RGB 与 DSM 都有 CRS/地理变换（可不同 CRS、不同分辨率），按地理坐标重投影到块网格；
- pixel：两者都没有地理信息时，要求尺寸完全一致，按像素直接对应。
块网格：块在 RGB 上的像素范围，按 factor（≈ DSM 一个像元覆盖的 RGB 像元数）下采样，
避免把粗分辨率 DSM 插值到 RGB 原分辨率。
"""
from __future__ import annotations

import math
from pathlib import Path
import warnings

import numpy as np

from ..io import relative_path, resolve_path

# DSM 常见的未声明无效值（-9999、-32768、3.4e38 等）：绝对值超过该值一律视为无效。
ABS_MAX_M = 1.0e4


def _open(path: str | Path):
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rasterio.errors.NotGeoreferencedWarning)
        return rasterio.open(resolve_path(path))


def _georeferenced(dataset) -> bool:
    return dataset.crs is not None and not dataset.transform.is_identity


def inspect_dsm(dsm_path: str | Path, rgb_path: str | Path, rgb_m_per_px: float) -> dict:
    """Node 1：检查 DSM 与 RGB 能否对齐，返回写入 manifest image.dsm 的元数据；不能对齐时报错。"""
    try:
        import rasterio
        from rasterio.warp import transform, transform_bounds
    except ImportError as exc:
        raise ImportError('读取 DSM 需要 rasterio') from exc
    source = resolve_path(dsm_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with _open(source) as dsm, _open(rgb_path) as rgb:
        if dsm.count < 1:
            raise ValueError('DSM 没有波段')
        if not np.issubdtype(np.dtype(dsm.dtypes[0]), np.number):
            raise ValueError(f'DSM 波段类型不是数值：{dsm.dtypes[0]}')
        info = {'path': relative_path(source), 'width': dsm.width, 'height': dsm.height, 'dtype': dsm.dtypes[0],
                'nodata': None if dsm.nodata is None or not math.isfinite(dsm.nodata) else float(dsm.nodata),
                'vertical_unit': 'm (assumed)'}
        if _georeferenced(dsm) and _georeferenced(rgb):
            left, bottom, right, top = transform_bounds(dsm.crs, rgb.crs, *dsm.bounds)
            r = rgb.bounds
            overlap = max(0., min(right, r.right)-max(left, r.left))*max(0., min(top, r.top)-max(bottom, r.bottom))
            ratio = overlap/((r.right-r.left)*(r.top-r.bottom))
            if ratio <= 0:
                raise ValueError('DSM 与 RGB 的地理范围没有重叠')
            # DSM 中心一个像元在 RGB 像素网格上的边长（两轴平均）→ DSM 分辨率（米/像元）。
            c, row = dsm.width/2, dsm.height/2
            xs, ys = zip(*(dsm.transform @ (c+dx, row+dy) for dx, dy in ((0, 0), (1, 0), (0, 1))))
            xs, ys = transform(dsm.crs, rgb.crs, list(xs), list(ys))
            inverse = ~rgb.transform
            (ax, ay), (bx, by), (cx, cy) = (inverse @ (x, y) for x, y in zip(xs, ys))
            rgb_px = (math.hypot(bx-ax, by-ay)+math.hypot(cx-ax, cy-ay))/2
            info.update({'alignment': 'geo', 'crs': dsm.crs.to_string(), 'rgb_crs': rgb.crs.to_string(),
                         'rgb_transform': list(rgb.transform)[:6], 'overlap_ratio': round(ratio, 6)})
        elif not _georeferenced(dsm) and not _georeferenced(rgb):
            if (dsm.width, dsm.height) != (rgb.width, rgb.height):
                raise ValueError(f'RGB 与 DSM 都没有地理信息，只能按像素对齐，但尺寸不同：'
                                 f'RGB {rgb.width}x{rgb.height}，DSM {dsm.width}x{dsm.height}')
            rgb_px = 1.
            info.update({'alignment': 'pixel', 'crs': None, 'overlap_ratio': 1.})
        else:
            raise ValueError('RGB 与 DSM 必须同时有地理信息（CRS + 变换），或同时没有（按像素对齐）')
    info['rgb_px_per_dsm_px'] = round(rgb_px, 6)
    info['m_per_px'] = round(rgb_px*rgb_m_per_px, 6)
    # 下采样倍数：块网格每个像元约等于一个 DSM 像元（DSM 比 RGB 细时不上采样，按 RGB 网格读）。
    info['block_grid_factor'] = max(1, int(round(rgb_px)))
    return info


def read_block_dsm(dsm: dict, bbox: list[int] | tuple[int, int, int, int]) -> np.ndarray:
    """读取块（RGB 像素范围 bbox）对应的 DSM，float32 米，无效/无覆盖为 NaN。

    输出网格 = bbox 按 block_grid_factor 下采样，形状 (ceil(h/f), ceil(w/f))。
    """
    from rasterio.enums import Resampling
    x0, y0, x1, y1 = bbox
    factor = dsm['block_grid_factor']
    shape = (math.ceil((y1-y0)/factor), math.ceil((x1-x0)/factor))
    out = np.full(shape, np.nan, np.float32)
    with _open(dsm['path']) as source:
        nodata = source.nodata
        if dsm['alignment'] == 'pixel':
            from rasterio.windows import Window
            data = source.read(1, window=Window(x0, y0, x1-x0, y1-y0), out_shape=shape,
                               resampling=Resampling.bilinear, masked=True)
            out = data.astype(np.float32).filled(np.nan)
        else:
            from affine import Affine
            from rasterio.crs import CRS
            from rasterio.warp import reproject
            import rasterio
            target = Affine(*dsm["rgb_transform"]) @ Affine.translation(x0, y0) @ Affine.scale(factor)
            reproject(source=rasterio.band(source, 1), destination=out, src_nodata=nodata, dst_nodata=np.nan,
                      dst_transform=target, dst_crs=CRS.from_string(dsm['rgb_crs']), resampling=Resampling.bilinear)
    out[~np.isfinite(out) | (np.abs(out) > ABS_MAX_M)] = np.nan
    if nodata is not None and math.isfinite(nodata):
        out[out == nodata] = np.nan
    return out


def downsample_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """语义标签最近邻缩放到 DSM 块网格（与 read_block_dsm 的网格一致）。"""
    from PIL import Image
    if labels.shape == shape:
        return labels
    return np.asarray(Image.fromarray(labels).resize((shape[1], shape[0]), Image.Resampling.NEAREST))
