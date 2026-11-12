"""用不同的官方 MaGRoad 权重对整幅 GeoTIFF 提取道路图，并输出对比。

每个权重按其训练 GSD 重采样推理（WildRoad 0.3 m/px，Global-Scale/SpaceNet 1 m/px），
节点坐标映射回原图像素网格并附经纬度；概率图按工作网格保存为带地理参考的 GeoTIFF。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np

from agent.paths import relative_path, resolve_path, write_json

DEFAULT_IMAGE = 'input_geo_img/miluo_dom.tif'
COLORS = {'wildroad': (255, 80, 40), 'globalscale': (40, 200, 255), 'spacenet': (255, 230, 0)}


def draw_graph(canvas: np.ndarray, nodes_xy: np.ndarray, edges: list[list[int]], scale: float,
               color: tuple[int, int, int], width: int = 3) -> np.ndarray:
    import cv2
    out = np.ascontiguousarray(canvas.copy())
    points = np.round(nodes_xy*scale).astype(np.int32)
    for a, b in edges:
        cv2.line(out, tuple(map(int, points[a])), tuple(map(int, points[b])), color, width, cv2.LINE_AA)
    for point in points:
        cv2.circle(out, tuple(map(int, point)), width+1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, tuple(map(int, point)), width, color, -1, cv2.LINE_AA)
    return out


def parse_variant(text: str) -> tuple[str, str, dict[str, float]]:
    """'globalscale' 或 'globalscale:ROAD=0.2,TOPO=0.3' → (权重, 输出名, 阈值覆盖)。"""
    from percection_tool.magroad import parse_checkpoint_spec
    return parse_checkpoint_spec(text)


def write_probability(path: Path, probability: np.ndarray, crs, transform) -> None:
    import rasterio
    with rasterio.open(path, 'w', driver='GTiff', width=probability.shape[1], height=probability.shape[0],
                       count=1, dtype='uint8', crs=crs, transform=transform, compress='deflate') as target:
        target.write((np.clip(probability, 0, 1)*255).astype(np.uint8), 1)
        target.update_tags(representation='probability_x255')


def run(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', default=DEFAULT_IMAGE)
    parser.add_argument('--checkpoints', nargs='+', default=['wildroad', 'globalscale', 'spacenet'],
                        help='权重名，可附阈值覆盖，如 globalscale:ROAD=0.2,ITSC=0.1,TOPO=0.3')
    parser.add_argument('--output-dir', help='默认 output/perception_tools/magroad/<图片名>/<时间戳>')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--model-resolution', type=float,
                        help='覆盖所有权重的推理 GSD（m/px）；默认用各自训练 GSD')
    parser.add_argument('--keep-black', action='store_true',
                        help='GeoTIFF 无掩码时不把大片纯黑区域当作无效区')
    parser.add_argument('--preview-edge', type=int, default=2400, help='对比图长边像素')
    args = parser.parse_args(argv)

    import cv2
    from rasterio.transform import Affine
    from agent.map_preprocess.config import PreprocessConfig
    from agent.map_preprocess.io import open_image
    from agent.map_preprocess.utils.resolution import calculate_resolution
    from percection_tool.magroad import MaGRoadExtractor, black_border_invalid

    variants = [parse_variant(item) for item in args.checkpoints]
    with open_image(args.image, 0) as source:
        if source.raster is None or source.raster.crs is None:
            raise ValueError('输入没有地理参考')
        width, height = source.width, source.height
        resolution, origin, resolution_meta = calculate_resolution(source.raster, width, height, PreprocessConfig())
        if origin != 'geotiff':
            raise ValueError(f'分辨率来源为 {origin}，不是 geotiff')
        crs, transform = source.raster.crs, source.raster.transform
        bbox = [0, 0, width, height]
        rgb, valid = source.read(bbox), source.valid_mask(bbox)
    valid_source = 'dataset_mask'
    if valid.all() and not args.keep_black:
        # 无 nodata/alpha 的正射图（如河西）外缘为纯黑：大片 (0,0,0) 视为无效，开运算去掉零星暗像素。
        black = black_border_invalid(rgb)
        if black is not None:
            valid, valid_source = ~black, 'black_border_fallback'
        del black

    output = resolve_path(args.output_dir or
                          f'output/perception_tools/magroad/{Path(args.image).stem}/{datetime.now():%Y%m%d_%H%M%S}')
    output.mkdir(parents=True, exist_ok=True)
    scale = args.preview_edge/max(width, height)
    preview = cv2.resize(rgb, (round(width*scale), round(height*scale)), interpolation=cv2.INTER_AREA)
    panels, summary = [], {'image': relative_path(args.image), 'source_size': [width, height],
                           'resolution_m_per_px': resolution, 'resolution_source': origin,
                           'resolution_metadata': resolution_meta, 'valid_fraction': float(valid.mean()), 'valid_source': valid_source,
                           'runs': {}}
    for name, label, overrides in variants:
        logging.info('MaGRoad %s 推理开始', label)
        tool = MaGRoadExtractor(name, device=args.device, model_resolution_m_per_px=args.model_resolution,
                                threshold_overrides=overrides)
        try:
            result = tool.predict_graph(rgb, resolution, valid_mask=valid)
        finally:
            tool.close()
        meta = result['metadata']
        target = output/label
        target.mkdir(exist_ok=True)
        nodes = np.asarray(result['nodes_xy'], dtype=float).reshape(-1, 2)
        lonlat = [list(transform*(x, y)) for x, y in nodes]
        sx, sy = meta['scale_to_source_xy']
        working_transform = transform*Affine.scale(sx, sy)
        write_probability(target/'road_probability.tif', result['road_probability'], crs, working_transform)
        write_probability(target/'keypoint_probability.tif', result['keypoint_probability'], crs, working_transform)
        cv2.imwrite(str(target/'road_probability.png'), (result['road_probability']*255).astype(np.uint8))
        working_overlay = draw_graph(result['working_rgb'], nodes/[sx, sy], result['edges'], 1.0, COLORS.get(name, (255, 0, 0)), 2)
        cv2.imwrite(str(target/'graph_working_grid.png'), cv2.cvtColor(working_overlay, cv2.COLOR_RGB2BGR))
        panel = draw_graph(preview, nodes, result['edges'], scale, COLORS.get(name, (255, 0, 0)), 3)
        cv2.imwrite(str(target/'graph_preview.png'), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        graph = {'coordinate_system': 'source pixel-edge coordinates (x, y); lonlat via GeoTIFF affine',
                 'crs': crs.to_string(), 'nodes_xy': nodes.tolist(), 'nodes_lonlat': lonlat,
                 'edges': result['edges'], 'edge_scores': result['edge_scores'],
                 'directed_predictions': result['directed_predictions'], 'metadata': meta}
        write_json(target/'graph.json', graph)
        features = [{'type': 'Feature', 'properties': {'score': score, 'source': a, 'target': b},
                     'geometry': {'type': 'LineString', 'coordinates': [lonlat[a], lonlat[b]]}}
                    for (a, b), score in zip(result['edges'], result['edge_scores'])]
        (target/'roads.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': features}), encoding='utf-8')
        lengths = [np.hypot(*(nodes[a]-nodes[b]))*resolution for a, b in result['edges']]
        summary['runs'][label] = {**{key: meta[key] for key in (
            'working_size', 'model_resolution_m_per_px', 'patch_size', 'patch_count', 'patch_stride_px',
            'node_count', 'edge_count', 'thresholds', 'official_thresholds', 'peak_cuda_allocated_bytes',
            'inference_seconds_including_model_load', 'config_overrides')},
            'checkpoint_load': meta['checkpoint_load'], 'total_edge_length_m': float(sum(lengths)),
            'outputs': relative_path(target)}
        thresholds = meta['thresholds']
        text = (f'{label}  {meta["model_resolution_m_per_px"]} m/px  nodes={meta["node_count"]} edges={meta["edge_count"]}'
                f'  thr i/r/t={thresholds["ITSC_THRESHOLD"]:g}/{thresholds["ROAD_THRESHOLD"]:g}/{thresholds["TOPO_THRESHOLD"]:g}')
        cv2.putText(panel, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(panel, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
        logging.info('MaGRoad %s 完成：%d 节点 %d 边', label, meta['node_count'], meta['edge_count'])
    if panels:
        cv2.imwrite(str(output/'comparison.jpg'), cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    write_json(output/'summary.json', summary)
    return output


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    try:
        print(run(argv))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f'MaGRoad 推理失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
