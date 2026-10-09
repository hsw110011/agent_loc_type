"""Node 1：实际重采样至 0.2 m/pixel，验证 TIFF，保存路径与 trace。"""
from __future__ import annotations
import base64
import io
import json
import math
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling, ColorInterp
from rasterio.transform import from_origin, Affine
from rasterio.warp import calculate_default_transform, reproject, transform as crs_transform

from .client import VLMClient
from .common import PositioningError, _digest, record_trace, resolve_path, relative_path, relative_paths

TARGET_GSD = 0.2


def _gauss_crs(src):
    """由影像中心经纬度确定 WGS84 三度带，比例因子 1，东坐标不加带号。"""
    center_x, center_y = src.transform * (src.width / 2, src.height / 2)
    longitude, latitude = crs_transform(src.crs, 'EPSG:4326', [center_x], [center_y])
    lon, lat = longitude[0], latitude[0]
    if not math.isfinite(lon) or not math.isfinite(lat) or not -180 <= lon <= 180 or not -90 < lat < 90:
        raise PositioningError('无法由输入 CRS 计算合法影像中心经纬度')
    zone = math.floor(lon / 3 + 0.5)
    central_meridian = zone * 3
    projection = (f'+proj=tmerc +lat_0=0 +lon_0={central_meridian} +k=1 '
                  '+x_0=500000 +y_0=0 +datum=WGS84 +units=m +no_defs')
    crs = rasterio.crs.CRS.from_string(projection)
    return crs, {'datum': 'WGS84', 'band_width_degrees': 3, 'zone': zone,
                 'central_meridian_degrees': central_meridian, 'scale_factor': 1.0,
                 'false_easting_m': 500000.0, 'false_northing_m': 0.0,
                 'zone_prefix_in_easting': False, 'axis_order': ['easting', 'northing'],
                 'proj_string': projection, 'crs_wkt': crs.to_wkt(),
                 'image_center_lonlat': [lon, lat]}


def lonlat_to_pixel(longitude: float, latitude: float, state: dict) -> dict:
    """经纬度 → Gauss(E,N) → 连续像素(col,row)，采用像元边缘坐标。"""
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in (longitude, latitude)) or not -180 <= longitude <= 180 or not -90 < latitude < 90:
        raise ValueError('经纬度不合法；顺序必须为 longitude, latitude')
    metadata = state['coordinate_transform']
    if not metadata['gauss']:
        raise PositioningError('无地理 CRS 的普通 TIFF 不支持经纬度到 Gauss/pixel 转换')
    easting, northing = crs_transform('EPSG:4326', metadata['gauss']['crs_wkt'], [longitude], [latitude])
    affine = Affine(*metadata['pixel_transform'])
    column, row = ~affine * (easting[0], northing[0])
    return {'longitude': longitude, 'latitude': latitude, 'easting_m': easting[0],
            'northing_m': northing[0], 'column': column, 'row': row,
            'inside_image': 0 <= column < metadata['width'] and 0 <= row < metadata['height']}


def pixel_to_lonlat(column: float, row: float, state: dict) -> dict:
    """像元边缘坐标 → Gauss(E,N) → 经纬度；像元中心需自行加 0.5。"""
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in (column, row)):
        raise ValueError('像素坐标必须是有限数值')
    metadata = state['coordinate_transform']
    if not metadata['gauss']:
        raise PositioningError('普通 TIFF 的本地网格没有地理 CRS')
    easting, northing = Affine(*metadata['pixel_transform']) * (column, row)
    longitude, latitude = crs_transform(metadata['gauss']['crs_wkt'], 'EPSG:4326', [easting], [northing])
    return {'column': column, 'row': row, 'easting_m': easting, 'northing_m': northing,
            'longitude': longitude[0], 'latitude': latitude[0]}


def _coordinate_metadata(src, crs, affine, width, height, gauss):
    metadata = {'chain': ['WGS84_longitude_latitude', 'Gauss_easting_northing_m', 'pixel_column_row'] if gauss else ['local_metric_grid', 'pixel_column_row'],
                'gauss': gauss, 'pixel_transform': list(affine)[:6], 'width': width, 'height': height,
                'pixel_convention': 'pixel edges; center of pixel (col,row) is (col+0.5,row+0.5)',
                'target_resolution_gauss_m': [TARGET_GSD, TARGET_GSD] if gauss else None,
                'source_resolution_gauss_m_at_center': None, 'reference_points': []}
    if not gauss:
        return metadata
    col, row = src.width / 2, src.height / 2
    source_points = [src.transform * point for point in [(col, row), (col + 1, row), (col, row + 1)]]
    lon, lat = crs_transform(src.crs, 'EPSG:4326', [p[0] for p in source_points], [p[1] for p in source_points])
    east, north = crs_transform('EPSG:4326', crs, lon, lat)
    metadata['source_resolution_gauss_m_at_center'] = {
        'column_step_m': math.hypot(east[1]-east[0], north[1]-north[0]),
        'row_step_m': math.hypot(east[2]-east[0], north[2]-north[0]),
        'measurement_location': 'source image center'}
    coords = [(0,0),(width,0),(width,height),(0,height),(width/2,height/2)]
    state = {'coordinate_transform': metadata}
    for label, point in zip(['top_left','top_right','bottom_right','bottom_left','center'], coords):
        reference = pixel_to_lonlat(*point, state)
        roundtrip = lonlat_to_pixel(reference['longitude'], reference['latitude'], state)
        error = math.hypot(point[0]-roundtrip['column'], point[1]-roundtrip['row'])
        if error > 1e-4:
            raise PositioningError(f'经纬度/Gauss/像素往返误差过大：{error} pixel')
        metadata['reference_points'].append({'label': label, **reference, 'roundtrip_error_pixels': error})
    return metadata


def _target_grid(src, source_gsd, max_pixels):
    if src.crs:
        crs, gauss = _gauss_crs(src)
        affine, width, height = calculate_default_transform(
            src.crs, crs, src.width, src.height, *src.bounds, resolution=TARGET_GSD)
        mode = 'wgs84_3degree_gauss'
    else:
        if (isinstance(source_gsd, bool) or not isinstance(source_gsd, (int, float))
                or not math.isfinite(source_gsd) or source_gsd <= 0):
            raise PositioningError('普通 TIFF 缺少 CRS，必须用 gsd_m/--gsd-m 提供真实原始 m/pixel')
        width = math.ceil(src.width * source_gsd / TARGET_GSD)
        height = math.ceil(src.height * source_gsd / TARGET_GSD)
        crs, gauss = None, None
        affine = from_origin(0, src.height * source_gsd, TARGET_GSD, TARGET_GSD)
        mode = 'local_metric_grid_from_supplied_gsd'
    if width * height > max_pixels:
        raise PositioningError(f'0.2 m/pixel 输出预计 {width}×{height}，超过 max_output_pixels={max_pixels}；请先裁剪实际研究区域')
    coordinates = _coordinate_metadata(src, crs, affine, width, height, gauss)
    return crs, affine, width, height, mode, coordinates


def _rgb_preview(dataset, path):
    """保持 uint8 RGB；其它数值类型按有效像元分位数拉伸，记录拉伸范围。"""
    rgb = [dataset.colorinterp.index(color) + 1 for color in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
           if color in dataset.colorinterp]
    if len(rgb) != 3:
        rgb = [1, 2, 3] if dataset.count >= 3 else [1, 1, 1]
    data = dataset.read(rgb)
    valid = dataset.dataset_mask() > 0
    limits = []
    if data.dtype != np.uint8:
        scaled = np.zeros(data.shape, dtype=np.uint8)
        for index, band in enumerate(data):
            pixels = band[valid & np.isfinite(band)]
            if not pixels.size:
                raise PositioningError('影像没有可用于 RGB 预览的有效像元')
            low, high = np.percentile(pixels, (2, 98))
            scaled[index] = np.clip(np.nan_to_num((band - low) * 255 / max(float(high - low), 1e-12)), 0, 255).astype(np.uint8)
            limits.append([float(low), float(high)])
        data = scaled
    data[:, ~valid] = 0
    Image.fromarray(data.transpose(1, 2, 0)).save(path)
    return {'bands': rgb, 'stretch_percentiles': [2, 98] if limits else None, 'stretch_limits': limits}


def read_picture(state: dict, *, client: VLMClient | None = None) -> dict:
    """不依赖 prompt 系统；输出 image_path 指向重采样后的真实 TIFF。"""
    source = resolve_path(state['image_path'])
    if source.suffix.lower() not in {'.tif', '.tiff'}:
        raise PositioningError('输入必须是 .tif 或 .tiff 文件；不能只修改扩展名')
    with source.open('rb') as stream:
        if stream.read(4) not in {b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+'}:
            raise PositioningError('输入文件内容不是 TIFF/BigTIFF')
    if not isinstance(state.get('primary_mode'), str) or not state['primary_mode'].strip():
        raise ValueError('必须提供指定定位方式 primary_mode')
    if state.get('geo_coordinates') is not None and not isinstance(state['geo_coordinates'], dict):
        raise ValueError('geo_coordinates 必须为空或对象')
    limit = state.get('max_regions')
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError('max_regions 必须为空或正整数')
    context = state.get('task_context', {})
    if not isinstance(context, dict):
        raise ValueError('task_context 必须为对象')
    fixed = {'platform': 'unmanned_ground_vehicle', 'height_reference': 'DSM'}
    if any(key in context and context[key] != value for key, value in fixed.items()):
        raise ValueError('task_context 与无人车/DSM 约定不一致')
    context = {'target_accuracy_m': None, 'available_sensors': None, 'available_maps': None, **context, **fixed}
    max_pixels = state.get('max_output_pixels', 64_000_000)
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels <= 0:
        raise ValueError('max_output_pixels 必须为正整数')
    root = resolve_path(state.get('output_dir') or Path(__file__).parent / 'output')
    task_id = state.get('task_id') or datetime.now().strftime('run_%Y%m%d_%H%M%S_') + uuid4().hex[:8]
    if not isinstance(task_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', task_id):
        raise ValueError('task_id 必须为 1～80 位字母、数字、下划线或连字符')
    digest = _digest(str(source))
    with rasterio.open(source) as src:
        if src.driver != 'GTiff':
            raise PositioningError('输入文件的实际驱动不是 GTiff')
        orientation = src.tags(ns='TIFF').get('TIFFTAG_ORIENTATION', src.tags().get('TIFFTAG_ORIENTATION', '1'))
        if str(orientation) != '1':
            raise PositioningError('输入 TIFF 非标准方向，请先纠正方向与地理变换')
        crs, transform, width, height, mode, coordinates = _target_grid(src, state.get('gsd_m'), max_pixels)
        if state.get('geo_coordinates') is not None:
            supplied = state['geo_coordinates']
            longitude = supplied.get('longitude', supplied.get('lon'))
            latitude = supplied.get('latitude', supplied.get('lat'))
            if longitude is None or latitude is None:
                raise ValueError('geo_coordinates 必须包含 longitude/latitude（也支持 lon/lat）')
            coordinates['input_coordinate'] = lonlat_to_pixel(longitude, latitude, {'coordinate_transform': coordinates})
        else:
            coordinates['input_coordinate'] = None
        source_info = {'path': str(source), 'sha256': digest, 'width': src.width, 'height': src.height,
                       'crs': str(src.crs) if src.crs else None, 'transform': list(src.transform)[:6],
                       'resolution_in_crs_units': list(src.res), 'supplied_gsd_m': state.get('gsd_m'),
                       'band_count': src.count, 'dtype': list(src.dtypes)}
        directory = root / task_id
        directory.mkdir(parents=True, exist_ok=False)
        output = directory / 'satellite_0p2m.tiff'
        preview = directory / 'satellite_rgb.png'
        trace_state = {**state, 'trace_dir': str(directory)}
        try:
            profile = src.profile.copy()
            # GDAL 解码 JPEG/YCbCr 后返回 RGB；DEFLATE 输出不能沿用 JPEG 专属 photometric。
            profile.pop('photometric', None)
            profile.update(driver='GTiff', crs=crs, transform=transform, width=width, height=height,
                           compress='deflate', tiled=True, blockxsize=256, blockysize=256, BIGTIFF='IF_SAFER')
            # 无 CRS 时只建立本地米制网格；不编造地理坐标。
            source_crs = src.crs or rasterio.crs.CRS.from_epsg(3857)
            destination_crs = crs or source_crs
            source_transform = src.transform if src.crs else from_origin(0, src.height * state['gsd_m'], state['gsd_m'], state['gsd_m'])
            with rasterio.open(output, 'w', **profile) as dst:
                for band in src.indexes:
                    reproject(rasterio.band(src, band) if src.crs else src.read(band), rasterio.band(dst, band),
                              src_transform=source_transform, src_crs=source_crs,
                              dst_transform=transform, dst_crs=destination_crs,
                              src_nodata=src.nodata, dst_nodata=src.nodata, resampling=Resampling.bilinear)
                valid = np.zeros((height, width), dtype=np.uint8)
                reproject(src.dataset_mask(), valid, src_transform=source_transform, src_crs=source_crs,
                          dst_transform=transform, dst_crs=destination_crs, resampling=Resampling.nearest)
                dst.write_mask(valid)
                dst.colorinterp = src.colorinterp
                dst.update_tags(target_gsd_m=TARGET_GSD, source_image_path=relative_path(source), resampling='bilinear', grid_mode=mode)
            with rasterio.open(output) as verified:
                if (verified.driver != 'GTiff' or output.suffix != '.tiff'
                        or verified.crs != crs or not verified.transform.almost_equals(transform)
                        or not np.allclose(verified.res, (TARGET_GSD, TARGET_GSD), atol=1e-9, rtol=0)):
                    raise PositioningError('输出 TIFF 格式或 0.2 m/pixel 分辨率验证失败')
                rgb_info = _rgb_preview(verified, preview)
            if _digest(str(source)) != digest:
                raise PositioningError('处理过程中源 TIFF 发生变化')
            coordinates_path = directory / 'coordinate_transform.json'
            coordinates_path.write_text(json.dumps(coordinates, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            details = {'status': 'completed', 'source': source_info,
                       'coordinate_transform': coordinates, 'coordinate_transform_path': str(coordinates_path),
                       'output': {'path': str(output), 'rgb_path': str(preview), 'width': width, 'height': height,
                                  'format': 'GTiff', 'is_tiff': True, 'gsd_m': TARGET_GSD,
                                  'crs': str(crs) if crs else None, 'transform': list(transform)[:6], 'grid_mode': mode},
                       'rgb_conversion': rgb_info,
                       'note': '重采样改变像素网格，不会增加源影像的真实空间细节。'}
            traces = record_trace(trace_state, 'read_picture', details)
        except Exception as exc:
            record_trace(trace_state, 'read_picture', {'status': 'failed', 'error': str(exc), 'source': source_info})
            raise
    vlm = client or VLMClient()
    encoded = vlm._prepare_image(preview)
    with Image.open(io.BytesIO(base64.b64decode(encoded.split(',', 1)[1]))) as prepared:
        sent_width, sent_height = prepared.size
    return relative_paths({'source_image_path': str(source), 'source_image_digest': digest, 'path_base': 'project_root',
            'image_path': str(output), 'rgb_image_path': str(preview), 'image_digest': _digest(str(output)),
            'image': {'image_id': state.get('image_id', source.name), 'width': width, 'height': height,
                      'coordinate_system': 'original_image_pixels', 'acquisition_date': state.get('acquisition_date'), 'gsd_m': TARGET_GSD},
            'gsd_m': TARGET_GSD, 'task_context': context, 'max_regions': limit,
            'encoded_image': encoded, 'input_view': {'width': sent_width, 'height': sent_height,
                        'coordinate_system': 'normalized_0_1000', 'max_image_pixels': vlm.config.max_image_pixels},
            'task_id': task_id, 'trace_dir': str(directory), 'raster_metadata': details,
            'coordinate_transform': coordinates, 'coordinate_transform_path': str(coordinates_path), **traces})
