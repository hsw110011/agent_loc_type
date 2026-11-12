"""原始像素网格上的标量地面分辨率；不重投影、不重采样。"""
from __future__ import annotations

import math
from typing import Any
import warnings

from ..config import PreprocessConfig


def web_mercator_resolution(zoom: float, center_lat: float) -> float:
    if not math.isfinite(zoom) or not 0 <= zoom <= 30 or not math.isfinite(center_lat) or abs(center_lat) > 85.05112878:
        raise ValueError('非法 Web Mercator zoom/center_lat')
    return 156543.03392 * math.cos(math.radians(center_lat)) / 2**zoom


def calculate_resolution(raster: Any, width: int, height: int,
                         config: PreprocessConfig) -> tuple[float, str, dict]:
    """优先 CRS+affine；经纬度 CRS 使用中心纬度处椭球一阶米制近似。"""
    if raster is not None and raster.crs is not None:
        transform, crs = raster.transform, raster.crs
        vx, vy = (transform.a, transform.d), (transform.b, transform.e)
        if crs.is_geographic:
            _, latitude = transform * (width / 2, height / 2)
            # 使用 CRS 自带的椭球参数（GDAL WKT），无 pyproj 依赖。
            import re
            match = re.search(r'(?:SPHEROID|ELLIPSOID)\s*\[\s*"[^"]+"\s*,\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)', crs.to_wkt())
            if match and -90 < latitude < 90:
                major, inverse_flattening = map(float, match.groups())
                flattening = 1 / inverse_flattening if inverse_flattening else 0
                e2 = flattening * (2 - flattening)
                phi = math.radians(latitude)
                denominator = 1 - e2 * math.sin(phi)**2
                meters_lon = math.pi / 180 * major / math.sqrt(denominator) * math.cos(phi)
                meters_lat = math.pi / 180 * major * (1-e2) / denominator**1.5
                rx = math.hypot(vx[0]*meters_lon, vx[1]*meters_lat)
                ry = math.hypot(vy[0]*meters_lon, vy[1]*meters_lat)
            else:
                rx = ry = 0
        elif crs.is_projected:
            _, meters_per_unit = crs.linear_units_factor
            rx, ry = math.hypot(*vx)*meters_per_unit, math.hypot(*vy)*meters_per_unit
        else:
            rx = ry = 0
        if all(math.isfinite(value) and value > 0 for value in (rx, ry)):
            resolution = (rx+ry)/2
            if abs(rx-ry)/resolution > config.resolution_anisotropy_warning_ratio:
                warnings.warn('GeoTIFF 两轴分辨率不同：标量 m/px 使用算术平均，距离为近似值', RuntimeWarning)
            return resolution, 'geotiff', {'resolution_xy_m_per_px': [rx, ry],
                'crs': crs.to_string(), 'transform': list(transform)[:6],
                'method': 'center_latitude_ellipsoid_approximation' if crs.is_geographic else 'projected_crs_units'}
        warnings.warn('GeoTIFF CRS/transform 无法确定有效分辨率，继续回退', RuntimeWarning)
    latitude = config.center_lat
    if latitude is None and config.corner_lonlat:
        latitude = sum(pair[1] for pair in config.corner_lonlat) / len(config.corner_lonlat)
    if config.zoom is not None and latitude is not None:
        return web_mercator_resolution(config.zoom, latitude), 'web_mercator', {
            'zoom': config.zoom, 'center_lat': latitude, 'method': 'web_mercator_zoom'}
    if config.corner_lonlat:
        # 假设四角按左上、右上、右下、左下排列，覆盖范围是未缩放的 Web Mercator 截图。
        # 没有 zoom 时用四边长度推算地面像素大小，跨日期变更线取最近经度分支。
        radius = 156543.03392 * 256 / (2*math.pi)
        longitude0 = config.corner_lonlat[0][0]
        projected = [(radius*math.radians(longitude0+(lon-longitude0+180) % 360-180),
                      radius*math.log(math.tan(math.pi/4+math.radians(lat)/2)))
                     for lon, lat in config.corner_lonlat]
        rx = (math.dist(projected[0], projected[1])+math.dist(projected[3], projected[2]))/(2*width)
        ry = (math.dist(projected[0], projected[3])+math.dist(projected[1], projected[2]))/(2*height)
        scale = math.cos(math.radians(latitude))
        if rx > 0 and ry > 0:
            return (rx+ry)/2*scale, 'web_mercator', {'center_lat': latitude,
                'method': 'web_mercator_corners', 'resolution_xy_m_per_px': [rx*scale, ry*scale]}
        warnings.warn('四角范围退化，使用配置分辨率回退', RuntimeWarning)
    warnings.warn('无法从地理信息或 zoom/纬度计算分辨率，使用 fallback_resolution_m_per_px', RuntimeWarning)
    return config.fallback_resolution_m_per_px, 'fallback', {'method': 'configured_fallback'}
