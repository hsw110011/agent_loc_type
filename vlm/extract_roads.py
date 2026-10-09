"""Node 2：本地 PathFinder 分块推理、地理对齐 mask 和半透明增强。"""
from __future__ import annotations
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.features import shapes
from rasterio.warp import transform_geom

from .common import PositioningError, _digest, record_trace, resolve_path, relative_paths


ROAD_THRESHOLD = 0.1


def _select_device(requested: str, tile_size: int) -> tuple[str, dict]:
    """auto 优先 GPU；显存下限是运行前估计，实际 OOM 仍由 CUDA 明确报错。"""
    import torch
    if requested != 'auto':
        return requested, {'requested': requested, 'selected': requested, 'reason': 'explicit_user_selection'}
    if not torch.cuda.is_available():
        return 'cpu', {'requested': 'auto', 'selected': 'cpu', 'reason': 'cuda_unavailable'}
    free, total = torch.cuda.mem_get_info(0)
    minimum = int(2 * 1024**3 * max(1, (tile_size / 512)**2))
    device = 'cuda:0' if free >= minimum else 'cpu'
    return device, {'requested': 'auto', 'selected': device,
                   'reason': 'gpu_memory_available' if device.startswith('cuda') else 'insufficient_free_gpu_memory',
                   'gpu_name': torch.cuda.get_device_name(0), 'free_memory_mib': round(free / 2**20),
                   'total_memory_mib': round(total / 2**20), 'estimated_minimum_free_mib': round(minimum / 2**20)}

def _predict_mask(rgb, checkpoint, device, tile_size, overlap, threads):
    import torch
    repository = Path(__file__).resolve().parent / 'PathFinder'
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from PATHFinder.model.decoder import PATHFinder

    torch.set_num_threads(threads)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise PositioningError('所选 CUDA 设备不可用；请使用 pathfinder_device=cpu')
    # 第三方 pretrained=True 分支只创建 mit_b5 架构，不下载额外权重。
    model = PATHFinder(num_classes=2, pretrained=True)
    try:
        checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    except RuntimeError:
        checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    weights = checkpoint_data.get('model_state_dict', checkpoint_data)
    checkpoint_info = {'format': 'training_checkpoint' if 'model_state_dict' in checkpoint_data else 'state_dict',
                       'weights_key': 'model_state_dict' if 'model_state_dict' in checkpoint_data else None,
                       'epoch': checkpoint_data.get('epoch') if isinstance(checkpoint_data.get('epoch'), int) else None}
    model.load_state_dict(weights, strict=True)
    del checkpoint_data, weights
    model.to(device).eval()
    height, width = rgb.shape[:2]
    probabilities = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint16)
    stride = tile_size - overlap
    def starts(length):
        return sorted(set(range(0, max(length - tile_size, 0) + 1, stride)) | {max(0, length - tile_size)})
    tiles = 0
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    with torch.inference_mode():
        for top in starts(height):
            for left in starts(width):
                tile = rgb[top:top + tile_size, left:left + tile_size]
                th, tw = tile.shape[:2]
                padded = np.full((tile_size, tile_size, 3), 128, dtype=np.float32)
                padded[:th, :tw] = tile
                # 与官方 preprocessing.py 相同的 RGB mean/std，无整体缩图。
                tensor = torch.from_numpy(((padded - mean) / std).transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
                logits, _, _ = model(tensor)
                scores = logits.softmax(dim=1)[0, 1, :th, :tw].cpu().numpy()
                probabilities[top:top + th, left:left + tw] += scores
                counts[top:top + th, left:left + tw] += 1
                tiles += 1
    mask = (probabilities / np.maximum(counts, 1) > ROAD_THRESHOLD).astype(np.uint8)
    return mask, tiles, checkpoint_info


def _save_polygons(mask, profile, directory, enhanced, threshold):
    """按像元边缘提取完整 polygon（含孔洞），不简化、不删小区域。"""
    polygons = []
    overlay = Image.fromarray(enhanced)
    boundary = Image.new('L', overlay.size, 0)
    draw = ImageDraw.Draw(overlay)
    boundary_draw = ImageDraw.Draw(boundary)
    line_width = max(1, round(max(overlay.size) / 700))
    geo_features = []
    transform = profile['transform']
    for index, (geometry, value) in enumerate(shapes(mask, mask=mask.astype(bool), connectivity=4), 1):
        identifier = f'road_{index:04d}'
        polygons.append({'polygon_id': identifier, 'geometry': geometry})
        # 第一环为外边界，其余为孔洞边界；输出闭合环。
        for ring in geometry['coordinates']:
            points = [tuple(point) for point in ring]
            draw.line(points, fill=(255, 220, 0), width=line_width)
            boundary_draw.line(points, fill=255, width=line_width)
        if profile['crs']:
            coordinates = [[[*(transform * (x, y))] for x, y in ring]
                           for ring in geometry['coordinates']]
            world = {'type': 'Polygon', 'coordinates': coordinates}
            geo_features.append({'type': 'Feature', 'properties': {'polygon_id': identifier, 'threshold': threshold},
                                 'geometry': transform_geom(profile['crs'], 'EPSG:4326', world, precision=10)})
    pixel_path = directory / 'road_polygons.json'
    pixel_path.write_text(json.dumps({'coordinate_system': 'processed_image_pixel_edges',
        'image_size': list(overlay.size), 'connectivity': 4, 'threshold': threshold,
        'polygon_count': len(polygons), 'polygons': polygons}, ensure_ascii=False, indent=2), encoding='utf-8')
    geo_path = None
    if profile['crs']:
        geo_path = directory / 'road_polygons.geojson'
        geo_path.write_text(json.dumps({'type': 'FeatureCollection', 'features': geo_features},
                                      ensure_ascii=False, indent=2), encoding='utf-8')
    boundary_path = directory / 'road_boundary.png'
    boundary.save(boundary_path)
    overlay.save(directory / 'road_enhanced.png')
    return overlay, {'road_polygons_path': str(pixel_path),
                     'road_polygons_geojson_path': str(geo_path) if geo_path else None,
                     'road_boundary_path': str(boundary_path), 'polygon_count': len(polygons),
                     'polygon_coordinate_system': 'processed_image_pixel_edges',
                     'geojson_coordinate_system': 'EPSG:4326' if geo_path else None,
                     'visualization': 'green_mask_with_yellow_polygon_boundaries'}


def extract_roads(state: dict) -> dict:
    """与 Node 1 输出同尺寸/同 transform；auto 优先 GPU，支持已有对齐 mask。"""
    directory = resolve_path(state['trace_dir'])
    started = time.monotonic()
    try:
        if _digest(state['image_path']) != state['image_digest']:
            raise PositioningError('PathFinder 推理前处理后的 TIFF 已改变')
        with Image.open(resolve_path(state['rgb_image_path'])) as source:
            rgb = np.array(source.convert('RGB'))
        with rasterio.open(resolve_path(state['image_path'])) as raster:
            profile = raster.profile.copy()
            valid = raster.dataset_mask() > 0
            expected_size = (raster.height, raster.width)
        if rgb.shape[:2] != expected_size:
            raise PositioningError('RGB 输入与处理后 TIFF 尺寸不一致')
        supplied = state.get('road_mask_path')
        if supplied:
            supplied_path = resolve_path(supplied)
            with rasterio.open(supplied_path) as source:
                if (source.height, source.width) != expected_size:
                    raise PositioningError('传入 Road Mask 必须与处理后 TIFF 同尺寸')
                if source.crs and (source.crs != profile['crs'] or not source.transform.almost_equals(profile['transform'])):
                    raise PositioningError('传入 Road Mask 的地理坐标与处理后 TIFF 不一致')
                mask = (source.read(1) > 0).astype(np.uint8)
            details = {'status': 'provided_mask', 'source': str(supplied_path)}
        else:
            checkpoint = resolve_path(state.get('pathfinder_checkpoint') or Path(__file__).resolve().parent / 'PathFinder' / 'pathfinder.pth')
            if not checkpoint.is_file():
                raise PositioningError(f'PathFinder 权重不存在：{checkpoint}')
            requested_device = state.get('pathfinder_device', 'auto')
            tile_size, overlap = state.get('tile_size', 512), state.get('tile_overlap', 64)
            threads = state.get('pathfinder_threads', 4)
            if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 256 or tile_size % 32:
                raise ValueError('tile_size 必须是至少 256 的 32 倍数')
            if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < tile_size:
                raise ValueError('tile_overlap 必须满足 0 <= overlap < tile_size')
            if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
                raise ValueError('pathfinder_threads 必须为正整数')
            device, selection = _select_device(requested_device, tile_size)
            print(f'PathFinder device: {device}; reason: {selection["reason"]}', file=sys.stderr)
            mask, tiles, checkpoint_info = _predict_mask(rgb, checkpoint, device, tile_size, overlap, threads)
            details = {'status': 'completed', 'model': 'PathFinder', 'checkpoint': str(checkpoint),
                       'device': device, 'device_selection': selection, 'tile_size': tile_size, 'overlap': overlap,
                       'tile_count': tiles, 'threshold': ROAD_THRESHOLD, 'road_class': 1,
                       'checkpoint_info': checkpoint_info,
                       'normalization': {'mean': [123.675,116.28,103.53], 'std': [58.395,57.12,57.375]}}
        mask[~valid] = 0
        mask_tiff = directory / 'road_mask.tiff'
        mask_png = directory / 'road_mask.png'
        enhanced_path = directory / 'road_enhanced.png'
        profile.update(count=1, dtype='uint8', nodata=None, photometric='minisblack')
        with rasterio.open(mask_tiff, 'w', **profile) as target:
            target.write(mask, 1)
            target.write_mask(valid.astype(np.uint8) * 255)
            target.update_tags(road_class=1, background_class=0, gsd_m=0.2)
        Image.fromarray(mask * 255).save(mask_png)
        enhanced = rgb.copy()
        road = mask.astype(bool)
        enhanced[road] = (rgb[road] * 0.65 + np.array([0,255,100]) * 0.35).astype(np.uint8)
        overlay, polygon_info = _save_polygons(mask, profile, directory, enhanced, details.get('threshold'))
        details.update(polygon_info)
        with rasterio.open(mask_tiff) as verified:
            if ((verified.height, verified.width) != expected_size or verified.crs != profile['crs']
                    or verified.res != (0.2, 0.2) or not verified.transform.almost_equals(profile['transform'])):
                raise PositioningError('Road Mask 地理对齐验证失败')
        if _digest(state['image_path']) != state['image_digest']:
            raise PositioningError('PathFinder 推理期间输入 TIFF 已改变')
        details.update({'input_image_path': state['image_path'], 'rgb_image_path': state['rgb_image_path'],
                        'road_mask_path': str(mask_tiff), 'road_mask_png_path': str(mask_png),
                        'enhanced_image_path': str(enhanced_path), 'width': expected_size[1], 'height': expected_size[0],
                        'road_pixel_count': int(mask.sum()), 'valid_pixel_count': int(valid.sum()),
                        'latency_seconds': round(time.monotonic() - started, 3)})
        traces = record_trace(state, 'extract_roads', details)
        sent = overlay.resize((state['input_view']['width'], state['input_view']['height']), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        sent.save(output, format='PNG')
        return relative_paths({'road_mask_path': str(mask_tiff), 'road_mask_png_path': str(mask_png),
                'road_enhanced_path': str(enhanced_path), 'road_extraction': details,
                **{key: polygon_info[key] for key in ('road_polygons_path', 'road_polygons_geojson_path', 'road_boundary_path')},
                'road_encoded_image': 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode('ascii'), **traces})
    except Exception as exc:
        record_trace(state, 'extract_roads', {'status': 'failed', 'error': str(exc), 'latency_seconds': time.monotonic() - started})
        raise
