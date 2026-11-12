"""前四节点的整图实现；state 仅包含路径和轻量元数据。"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Callable
from uuid import uuid4
import networkx as nx
import warnings
import numpy as np
from PIL import Image
from shapely.geometry import LineString

from ..global_config import GlobalPreprocessConfig
from ..io import file_sha256, fsync_path, open_image, read_json, relative_path, resolve_path, write_json
from .. import output_layout as layout
from ..state import PreprocessState
from ..utils.resolution import calculate_resolution
from ..utils.dsm import inspect_dsm
from ..utils import global_road_graph as roads
from ..utils import road_refine, road_skeleton, semantic_route
from ..utils.global_blocks import block_positions, blocks_for_edge, connect_blocks, group_blocks, crop_block
from ..utils.cross_edge_dedup import cross_edge_dedup
from ..utils.branch_review import BranchReviewer, PassThroughReviewer, RuleBasedReviewer, validated_decisions
from ..utils.global_render import render_block_layout, render_global, render_road_topology
from percection_tool.interfaces import RoadGraphExtractor
from percection_tool.factory import create_graph_extractor


def load_image_and_resolution(state: PreprocessState, *, config: GlobalPreprocessConfig) -> dict:
    with open_image(state['image_path'], config.pillow_max_image_pixels) as source:
        width, height = source.width, source.height
        resolution, origin, metadata = calculate_resolution(source.raster, width, height, config)
    if config.tile_grid > min(width, height):
        raise ValueError('tile_grid 超过图像尺寸，无法生成非空区域')
    tiles = []
    for row in range(config.tile_grid):
        for col in range(config.tile_grid):
            tiles.append({'tile_id': f'T{row}_{col}', 'x0': col*width//config.tile_grid,
                'y0': row*height//config.tile_grid, 'x1': (col+1)*width//config.tile_grid,
                'y1': (row+1)*height//config.tile_grid})
    image = {'path': state['image_path'], 'width': width, 'height': height,
             'coordinate_convention': 'global_pixel_edges_xy', **metadata}
    # DSM 可选：只检查与 RGB 能否对齐并记录元数据，拓扑仍只用 RGB；Node 5 据此按块裁剪高程。
    if state.get('dsm_path'):
        image['dsm'] = inspect_dsm(state['dsm_path'], state['image_path'], resolution)
    return {'image': image,
            'tiles': tiles, 'resolution_m_per_px': resolution, 'resolution_source': origin}


def extract_road_graph(state: PreprocessState, *, config: GlobalPreprocessConfig,
                       extractor: RoadGraphExtractor | None = None) -> dict:
    fingerprint = {'cache_version': 1, 'input_sha256': file_sha256(state['image_path']),
                   'resolution': state['resolution_m_per_px'], 'segmenter': config.segmenter}
    magroad_checkpoint = None
    if config.segmenter == 'magroad_scene' and extractor is None:
        from percection_tool.magroad.inference import CHECKPOINTS, parse_checkpoint_spec
        scene = state.get('scene_type') or {}
        magroad_checkpoint = scene.get('magroad_checkpoint')
        if not magroad_checkpoint:
            raise ValueError('segmenter=magroad_scene 需要先运行 Node 1b（classify_scene_type）')
        name, _, overrides = parse_checkpoint_spec(magroad_checkpoint)
        # 权重 SHA256 已固定在 CHECKPOINTS 中（加载时校验），这里不重复哈希 GB 级文件。
        fingerprint['magroad'] = {'adapter_version': 1, 'checkpoint': name, 'threshold_overrides': overrides,
                                  'checkpoint_sha256': CHECKPOINTS[name].sha256,
                                  'max_tile_pixels': config.max_tile_pixels}
    if config.dummy_graph_path:
        fingerprint['dummy_graph_sha256'] = file_sha256(config.dummy_graph_path)
    cache_enabled = config.cache_enabled
    if extractor is not None:
        # 假设注入者用 cache_key 显式声明实现/参数版本；否则不猜测其确定性。
        if not isinstance(getattr(extractor, 'cache_key', None), str):
            cache_enabled = False
        else:
            fingerprint['extractor'] = extractor.cache_key
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    root = resolve_path(config.cache_dir)
    directory = (layout.find_cache(root, key) or root/layout.cache_name(state['image_path'], key)) if cache_enabled else layout.state_dir(state['output_dir'])/'road_extraction_raw'
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory/'raw_graph.json'
    hit = False
    if cache_enabled and cache.is_file():
        try:
            payload = read_json(cache)
            files = payload.get('graph', {}).get('metadata', {}).get('files', {})
            if payload.get('fingerprint') == fingerprint and all(resolve_path(p).is_file() for p in files.values()):
                raw = payload['graph']
                roads.raw_to_graph(raw)
                hit = True
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            warnings.warn(f'道路图缓存无效，将重新提取：{exc}', RuntimeWarning)
    if not hit:
        if extractor is None:
            extractor = create_graph_extractor(config, state['resolution_m_per_px'], str(directory),
                                               magroad_checkpoint=magroad_checkpoint)
        raw = extractor.extract(resolve_path(state['image_path']))
        graph = roads.raw_to_graph(raw)
        width, height = state['image']['width'], state['image']['height']
        if any(not (0 <= x <= width and 0 <= y <= height)
               for *_, edge in graph.edges(data=True) for x, y in edge['polyline_xy']):
            raise ValueError('原始图折线坐标超出全局图像')
        temporary = cache.with_name(f'.{uuid4().hex}.json')
        write_json(temporary, {'fingerprint': fingerprint, 'graph': raw})
        fsync_path(temporary)  # 先落盘再改名：写入时断电也不会留下半个缓存
        temporary.replace(cache)
    target = layout.state_dir(state['output_dir'])/'raw_road_graph.json'
    write_json(target, raw)
    logging.getLogger(f"map_preprocess.{state['task_id']}").info('全局图缓存命中=%s key=%s', hit, key)
    return {'raw_road_graph_path': relative_path(target), 'road_graph_cache': {'hit': hit, 'key': key,
                'enabled': cache_enabled, 'path': relative_path(cache)}, 'input_sha256': fingerprint['input_sha256']}


def _constraint_filter(graph: nx.MultiGraph, state: PreprocessState, config: GlobalPreprocessConfig) -> list[dict]:
    resolution = state['resolution_m_per_px']
    route = config.route_polyline
    if route and config.route_coordinate_system == 'lonlat':
        metadata = state['image']
        if 'crs' not in metadata or 'transform' not in metadata:
            raise ValueError('经纬度 route_polyline 需要输入图像的 CRS/affine')
        try:
            from rasterio.warp import transform
            from affine import Affine
        except ImportError as exc:
            raise ValueError('经纬度约束转换需要 rasterio') from exc
        xs, ys = transform('EPSG:4326', metadata['crs'], [p[0] for p in route], [p[1] for p in route])
        inverse = ~Affine(*metadata['transform'])
        route = [inverse*(x, y) for x, y in zip(xs, ys)]
    corridor = LineString(route).buffer(config.route_buffer_m/resolution) if route and config.route_buffer_m > 0 else None
    centerline = LineString(route) if route else None
    mask = None
    if config.roi_mask:
        with Image.open(resolve_path(config.roi_mask)) as source:
            if source.size != (state['image']['width'], state['image']['height']):
                raise ValueError('roi_mask 必须与源图同尺寸、同像素网格')
            if source.width*source.height > config.max_tile_pixels:
                raise ValueError('roi_mask 超过 max_tile_pixels')
            mask = np.asarray(source.convert('L')) > 0
    rejected = set()
    for edge in roads.edge_records(graph):
        line = LineString(edge['polyline_xy'])
        if centerline is not None and not (corridor if corridor is not None else centerline).covers(line):
            rejected.add(edge['edge_id'])
            continue
        if mask is not None:
            # 假设 ROI 采用 <=0.5px 采样近似，检查整条边而不是仅检查端点。
            count = max(1, math.ceil(line.length/min(.5, config.constraint_sample_spacing_m/resolution)))
            radius = config.route_buffer_m/resolution
            for arc in np.linspace(0, line.length, count+1):
                point = line.interpolate(arc)
                x, y = min(mask.shape[1]-1, math.floor(point.x)), min(mask.shape[0]-1, math.floor(point.y))
                if mask[y, x]:
                    continue
                x0, y0 = max(0, math.floor(x-radius)), max(0, math.floor(y-radius))
                x1, y1 = min(mask.shape[1], math.ceil(x+radius)+1), min(mask.shape[0], math.ceil(y+radius)+1)
                yy, xx = np.ogrid[y0:y1, x0:x1]
                if not np.any(mask[y0:y1, x0:x1] & ((xx-x)**2+(yy-y)**2 <= radius**2)):
                    rejected.add(edge['edge_id'])
                    break
    return roads.drop_edges(graph, rejected, 'external_constraint')


def _road_support_fn(state: PreprocessState, config: GlobalPreprocessConfig) -> Callable[[list], float] | None:
    """折线 -> 沿线模型概率支持度；分割器没有保存概率图（如 dummy）时返回 None，依赖概率的规则跳过。"""
    cache = state.get('road_graph_cache')
    path = resolve_path(cache['path']).with_name('road_probability.npy') if cache else None
    if config.road_duplicate_min_support <= 0 or path is None or not path.is_file():
        return None
    probability = np.load(path, allow_pickle=False)
    size = (state['image']['width'], state['image']['height'])
    return lambda polyline: road_refine.road_support(polyline, probability, size, config.road_duplicate_prob,
                                                     state['resolution_m_per_px'])


def _topology_from_masks(state: PreprocessState, config: GlobalPreprocessConfig, semantic_segmenter) -> tuple:
    """掩码拓扑：MaGRoad 概率上采样到原图后和语义分割融合，生成路网；不使用 MaGRoad 的稀疏图。"""
    import cv2
    from .semantic_roads import whole_image_semantic
    from ..utils import road_topo
    from . import road_topo_cache as topo_cache
    logger = logging.getLogger(f"map_preprocess.{state['task_id']}")
    cache = state.get('road_graph_cache')
    path = resolve_path(cache['path']).with_name('road_probability.npy') if cache else None
    if path is None or not path.is_file():
        raise ValueError('road_topo_enabled 需要道路提取模型保存的 road_probability.npy（dummy 分割器没有）')
    resolution = state['resolution_m_per_px']
    size = (state['image']['width'], state['image']['height'])
    settings = road_topo.load_config(resolve_path(config.road_topo_config))
    metadata = state['image']
    geotiff = state['image_path'] if 'crs' in metadata and 'transform' in metadata else None
    out_dir, vis_dir = layout.state_dir(state['output_dir'])/'road_topo', layout.figure(state['output_dir'], 'road_topo_stages')
    # 拓扑只依赖 MaGRoad 概率、整图语义和 road_topo 参数/代码：同一输入再次运行直接读缓存（见 road_topo_cache.py）
    key = topo_cache.cache_key(state, config, semantic_segmenter, geotiff is not None)
    cached = topo_cache.load(config, key, out_dir, vis_dir, logger) if key else None
    semantic_hit = None
    if cached is not None:
        source, records, removed = cached
    else:
        labels, _, semantic_info = whole_image_semantic(state, config, semantic_segmenter, logger)
        semantic_hit = semantic_info['cache_hit']
        if labels.shape != size[::-1]:
            labels = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
        probability = np.load(path, allow_pickle=False).astype(np.float32)
        if probability.shape != size[::-1]:
            probability = cv2.resize(probability, size, interpolation=cv2.INTER_LINEAR)
        source, records, removed = road_topo.run_arrays(probability, labels, resolution, out_dir, settings,
            geotiff=geotiff, vis_out=vis_dir, log=logger.info)
        del labels, probability
        if key:
            topo_cache.save(config, key, state['image_path'], (source, records, removed), out_dir, vis_dir, logger)
    graph = road_topo.to_pipeline_graph(source, resolution, size)
    stats = {'road_topo': {'steps': [{k: v for k, v in r.items() if k != 'detail'} for r in records],
                           'cache_hit': cached is not None, 'cache_key': key[:24] if key else None,
                           'semantic_cache_hit': semantic_hit, 'after': roads.graph_stats(graph)}}
    if config.road_max_shared_gap_m > 0:
        stats['crowded_junctions'] = {'splits': road_refine.split_crowded_junctions(graph, config.road_max_shared_gap_m, resolution, config.road_max_node_degree),
                                      'after': roads.graph_stats(graph)}
    tolerance = settings.smooth.simplify_m/resolution
    dropped = []
    for index, points in enumerate(removed):
        line = np.clip(road_topo._douglas_peucker(points, tolerance)+.5, 0., size).tolist()
        dropped.append({'edge_id': f'road_topo_{index}', 'polyline_xy': line, 'length_m': roads.length_px(line)*resolution, 'reason': 'road_topo'})
    return graph, stats, dropped, None


def _legacy_topology(state: PreprocessState, config: GlobalPreprocessConfig, semantic_segmenter) -> tuple:
    """旧流程：从 MaGRoad 稀疏图逐步清洗。road_topo_enabled=false 和 dummy 分割器使用。"""
    raw = read_json(state['raw_road_graph_path'])
    source = roads.raw_to_graph(raw)
    graph = roads.compress_degree_two(source, state['resolution_m_per_px'])
    stats = {'compress_degree_two': {'before': {'edges': source.number_of_edges(),
        'total_length_m': sum(roads.length_px(d['polyline_xy'])*state['resolution_m_per_px'] for *_, d in source.edges(data=True))},
        'after': roads.graph_stats(graph)}}
    dropped = []
    def stage(name: str, operation: Callable[[], list[dict]]) -> None:
        before = roads.graph_stats(graph)
        dropped.extend(operation())
        stats[name] = {'before': before, 'after': roads.graph_stats(graph)}
    stage('small_components', lambda: roads.remove_small_components(graph, config.min_component_length_m))
    stage('spurs', lambda: roads.prune_short_spurs(graph, config.min_spur_length_m))
    # 过近路口合并：默认阈值为块边长的 1/4，约一个路口的宽度
    junction_m = config.junction_merge_dist_m if config.junction_merge_dist_m is not None else config.block_size_m/4
    stage('close_junctions', lambda: roads.merge_close_junctions(graph, junction_m, state['resolution_m_per_px']))
    # 判重需要“保留高分边”，先算一次候选分数，合并后再按最终连接关系评分。
    resolution = state['resolution_m_per_px']
    for *_, edge in graph.edges(data=True):
        edge['edge_score'] = roads.score_edge(edge, graph, config)
    stage('parallel_merge', lambda: road_refine.merge_parallel_roads(graph, config.parallel_merge_dist_m,
        config.parallel_max_angle_deg, config.parallel_min_overlap, resolution))
    # 去并行时支路改接到保留边上，可能又形成过近路口
    stage('parallel_close_junctions', lambda: roads.merge_close_junctions(graph, junction_m, resolution))
    # 先连后删：能接上的短路不算悬空
    connected = road_refine.connect_near_roads(graph, config.road_connect_dist_m, resolution)
    stats['connect_roads'] = {**connected, 'after': roads.graph_stats(graph)}
    stage('short_roads', lambda: road_refine.remove_short_roads(graph, config.short_road_max_m,
        config.small_loop_max_reach_m, resolution))
    image_size = (state['image']['width'], state['image']['height'])
    support_fn = _road_support_fn(state, config) if config.road_through_only else None
    def through_network(close_node_m: float, check_duplicates: bool = True) -> list[dict]:
        return road_refine.enforce_through_network(graph, close_node_m=close_node_m,
            small_cycle_m=config.small_cycle_max_length_m, border_margin_m=config.road_border_margin_m,
            image_size=image_size, resolution=resolution, road_support_fn=support_fn if check_duplicates else None,
            duplicate_min_support=config.road_duplicate_min_support) if config.road_through_only else []
    stage('through_network', lambda: through_network(config.road_close_node_dist_m))
    labels = None
    if config.road_semantic_route_enabled:
        # 语义路线和路面环判断共用整图语义
        from .semantic_roads import whole_image_semantic
        labels, _, semantic_info = whole_image_semantic(state, config, semantic_segmenter,
                                                        logging.getLogger(f"map_preprocess.{state['task_id']}"))
    cache = state.get('road_graph_cache')
    probability_path = resolve_path(cache['path']).with_name('road_probability.npy') if cache else None
    probability = np.load(probability_path, allow_pickle=False) if probability_path is not None and probability_path.is_file() else None
    if labels is not None and probability is not None and config.road_skeleton_fill_min_length_m > 0:
        # MaGRoad 仍是主路网：只补它漏掉、概率强、两端能接入的路段
        filled = road_skeleton.fill_from_skeleton(graph, probability, labels, image_size, resolution, config.road_skeleton_fill_min_length_m)
        stats['skeleton_fill'] = {**filled, 'after': roads.graph_stats(graph)}
        if filled['added']:
            stage('through_network_after_fill', lambda: through_network(config.road_close_node_dist_m))
    def paved_ring_pass() -> list[dict]:
        return [*semantic_route.collapse_paved_rings(graph, labels, image_size, config.road_paved_ring_min_road,
                                                     config.road_paved_ring_max_perimeter_m, resolution),
                *semantic_route.merge_wide_junctions(graph, labels, image_size, resolution)]
    if labels is not None and config.road_paved_ring_min_road > 0:
        def paved_rings() -> list[dict]:
            changes = paved_ring_pass()
            return [*changes, *through_network(config.road_close_node_dist_m)] if changes else []
        stage('paved_rings', paved_rings)
    if config.road_through_only:
        # 合并后支路贴着干路走到路口：在真实分岔处建节点，重合段只留一条
        stats['split_shared'] = {**road_refine.split_shared_stretches(graph, config.road_shared_min_length_m, resolution),
                                 'after': roads.graph_stats(graph)}
    if labels is not None and config.road_paved_ring_min_road > 0:
        # 分岔处建的节点落在宽路面上时仍是同一个路口
        stage('wide_junctions_after_split', lambda: semantic_route.merge_wide_junctions(graph, labels, image_size, resolution))
    if labels is not None:
        # 拓扑定稿后沿语义道路中线走，连接关系不变；随后的平滑负责曲线顺滑
        routed = semantic_route.route_along_semantic(graph, labels, image_size, resolution,
            config.road_semantic_route_corridor_m, config.road_semantic_route_node_snap_m, config.road_close_node_dist_m, probability)
        stats['semantic_route'] = {**routed, 'semantic_cache_hit': semantic_info['cache_hit'], 'after': roads.graph_stats(graph)}
        # 路线贴到路面中心后，环内部才准（如环岛外圈），再查一次
        if config.road_through_only:
            # 路线收到路中线后，支路可能沿干路走一段才到节点；宽路口里各路线本来就挤在一起，不算分岔
            stats['split_shared_after_route'] = road_refine.split_shared_stretches(graph, config.road_shared_min_length_m, resolution,
                skip=lambda point: semantic_route.is_wide_road(point, labels, image_size, resolution))
        if config.road_paved_ring_min_road > 0:
            stage('paved_rings_after_route', paved_ring_pass)
        del labels
    if config.road_max_shared_gap_m > 0:
        stage('tiny_edges', lambda: roads.merge_close_junctions(graph, config.road_max_shared_gap_m, resolution))
        stats['crowded_junctions'] = {'splits': road_refine.split_crowded_junctions(graph, config.road_max_shared_gap_m, resolution, config.road_max_node_degree),
                                      'after': roads.graph_stats(graph)}
    # 拓扑定稿后才平滑：节点不动，折角只留在节点处
    smoothed = road_refine.smooth_roads(graph, config.road_smooth_sigma_m, resolution,
                                        (state['image']['width'], state['image']['height']))
    stats['smooth_roads'] = {**smoothed, 'after': roads.graph_stats(graph)}
    # 低分边被删后可能又出现断头路；路口已定稿，这里不再合并，否则分岔处的节点又被并回去。
    # 弱重复只在拓扑阶段判断：路线和平滑已改变折线，沿线概率支持不再能代表这条边
    return graph, stats, dropped, lambda: through_network(0., check_duplicates=False)


def clean_road_graph(state: PreprocessState, *, config: GlobalPreprocessConfig, reviewer: BranchReviewer | None = None,
                     semantic_segmenter=None) -> dict:
    build = _topology_from_masks if config.road_topo_enabled else _legacy_topology
    graph, stats, dropped, after_score = build(state, config, semantic_segmenter)
    def stage(name: str, operation: Callable[[], list[dict]]) -> None:
        before = roads.graph_stats(graph)
        dropped.extend(operation())
        stats[name] = {'before': before, 'after': roads.graph_stats(graph)}
    for *_, edge in graph.edges(data=True):
        edge['edge_score'] = roads.score_edge(edge, graph, config)
    stage('constraints', lambda: _constraint_filter(graph, state, config))
    if not config.road_topo_enabled:   # 掩码拓扑已经判断过路段取舍，评分只用于块预算排序
        stage('score', lambda: roads.drop_edges(graph,
            {e['edge_id'] for e in roads.edge_records(graph) if e['edge_score'] < config.min_edge_score}, 'low_edge_score'))
        stage('through_network_after_score', after_score)
    if config.keep_main_trunk_only:
        keep = roads.main_trunk_edges(graph, config.min_attached_edge_length_m)
        stage('main_trunk', lambda: roads.drop_edges(graph,
            {e['edge_id'] for e in roads.edge_records(graph)}-keep, 'outside_main_trunk'))
    decisions, review_path = [], None
    if config.branch_review_enabled:
        selected = roads.edge_records(graph)
        review_path = relative_path(layout.figure(state['output_dir'], 'branch_review'))
        render_global(state['image'], review_path, config, edges=selected)
        engine = reviewer or (RuleBasedReviewer(config.review_top_n_edges) if config.reviewer_type == 'rule_based' else PassThroughReviewer())
        decisions = validated_decisions(selected, engine.review(state['image_path'], review_path, selected))
        reasons = {d['edge_id']: d['reason'] for d in decisions}
        stage('branch_review', lambda: roads.drop_edges(graph, {d['edge_id'] for d in decisions if not d['keep']}, 'branch_review'))
        for edge in dropped:
            if edge['reason'] == 'branch_review':
                edge['review_reason'] = reasons[edge['edge_id']]
    target = layout.state_dir(state['output_dir'])/'clean_road_graph.json'
    data = {**roads.serialize_graph(graph), 'dropped_edges': dropped, 'filter_stats': stats,
            'review_decisions': decisions, 'branch_review_path': review_path}
    write_json(target, data)
    overview = relative_path(layout.figure(state['output_dir'], 'road_cleaning'))
    render_global(state['image'], overview, config, edges=data['edges'], dropped=dropped)
    for name, values in stats.items():
        logging.getLogger(f"map_preprocess.{state['task_id']}").info('过滤 %s: %s', name, values)
    return {'clean_road_graph_path': relative_path(target), 'clean_graph_overview_path': overview}


def build_blocks_and_groups(state: PreprocessState, *, config: GlobalPreprocessConfig) -> dict:
    cleaned = read_json(state['clean_road_graph_path'])
    edges, dropped = cleaned['edges'], list(cleaned['dropped_edges'])
    before = {'edges': len(edges), 'total_length_m': sum(e['length_m'] for e in edges)}
    removed = set()
    if not config.dedup_enabled and config.max_blocks_total is not None:
        counts = {e['edge_id']: len(block_positions(e['length_m'], config, warn=False)) for e in edges}
        total = sum(counts.values())
        for edge in sorted(edges, key=lambda e: (e['edge_score'], e['edge_id'])):
            if total <= config.max_blocks_total:
                break
            removed.add(edge['edge_id'])
            total -= counts[edge['edge_id']]
            dropped.append({**edge, 'reason': 'max_blocks_total'})
        if total > config.max_blocks_total:
            warnings.warn('块预算仍未满足', RuntimeWarning)
        if removed:
            warnings.warn(f'块预算触发：丢弃 {len(removed)} 条完整边', RuntimeWarning)
    edges = [e for e in edges if e['edge_id'] not in removed]

    def build(selected_edges: list[dict]) -> tuple[list[dict], list[dict], dict | None]:
        blocks, coverage = [], []
        for edge in selected_edges:
            selected, metrics = blocks_for_edge(edge, state['image'], state['tiles'],
                state['resolution_m_per_px'], config, warn=not config.dedup_enabled)
            edge['block_ids'] = [b['block_id'] for b in selected]
            blocks.extend(selected)
            coverage.append(metrics)
        if config.dedup_enabled:
            initial_blocks[:] = blocks
            return cross_edge_dedup(selected_edges, blocks, state['image'], state['tiles'],
                                    state['resolution_m_per_px'], config)
        connect_blocks(selected_edges, blocks)
        return blocks, coverage, None

    initial_blocks: list[dict] = []
    blocks, coverage, dedup_stats = build(edges)
    if config.dedup_enabled and config.max_blocks_total is not None:
        for edge in sorted(edges, key=lambda e: (e['edge_score'], e['edge_id'])):
            if len(blocks) <= config.max_blocks_total:
                break
            removed.add(edge['edge_id'])
            dropped.append({**edge, 'reason': 'max_blocks_total'})
            edges = [e for e in edges if e['edge_id'] not in removed]
            # 假设预算仍按整边丢弃；重选以避免已丢道路参与覆盖目标或成为块归属边。
            blocks, coverage, dedup_stats = build(edges)
        if removed:
            warnings.warn(f'块预算触发：丢弃 {len(removed)} 条完整边', RuntimeWarning)
    used_nodes = {e[k] for e in edges for k in ('u', 'v')}
    nodes = [n for n in cleaned['nodes'] if n['node_id'] in used_nodes]
    for metrics in coverage:
        logging.getLogger(f"map_preprocess.{state['task_id']}").info('边覆盖/重叠 %s', metrics)
    if dedup_stats is not None:
        logging.getLogger(f"map_preprocess.{state['task_id']}").info('cross_edge_dedup %s', dedup_stats)
    groups = group_blocks(edges, blocks, config)
    stats = {**cleaned['filter_stats'], 'block_budget': {'before': before,
        'after': {'edges': len(edges), 'total_length_m': sum(e['length_m'] for e in edges)}}}
    records = layout.state_dir(state['output_dir'])
    manifest_path = relative_path(records/'manifest.json')
    files = {'manifest': manifest_path, 'image': state['image_path'],
             'raw_graph': state['raw_road_graph_path'], 'clean_graph': state['clean_road_graph_path'],
             'clean_graph_overview': state['clean_graph_overview_path'], 'cache': state['road_graph_cache']['path'],
             'branch_review': cleaned['branch_review_path'], 'config': relative_path(records/'config.json'),
             'log': relative_path(records/'run.log'), 'state': relative_path(records/'state.json'),
             'state_dir': relative_path(records), 'visualization_dir': relative_path(layout.vis_dir(state['output_dir']))}
    raw = read_json(state['raw_road_graph_path'])
    files.update(raw.get('metadata', {}).get('files', {}))
    manifest = {'format_version': 2, 'workflow': 'global_blocks', 'task_id': state['task_id'],
        'image': state['image'], 'resolution_m_per_px': state['resolution_m_per_px'],
        'resolution_source': state['resolution_source'], 'input_sha256': state['input_sha256'],
        'tiles': state['tiles'], 'nodes': nodes, 'edges': edges, 'blocks': blocks, 'groups': groups,
        'dropped_edges': dropped, 'filter_stats': stats, 'coverage': coverage,
        'review_decisions': cleaned['review_decisions'], 'config': config.to_dict(), 'files': files,
        'extensions': {name: state.get(name, {'status': 'skipped'})
                       for name in ('semantic_features', 'elevation_features', 'vlm_candidate_detection',
                                    'block_features')}}
    if dedup_stats is not None:
        manifest['dedup_stats'] = dedup_stats
    debug_paths = {}
    if config.debug:
        for identifier in config.debug_block_ids:
            if identifier not in {b['block_id'] for b in blocks}:
                raise ValueError(f'未知 debug_block_id: {identifier}')
            target = layout.figure(state['output_dir'], 'debug_blocks')/f'{identifier}.png'
            target.parent.mkdir(exist_ok=True)
            Image.fromarray(crop_block(manifest, identifier)).save(target)
            debug_paths[identifier] = relative_path(target)
    manifest['files']['debug_blocks'] = debug_paths
    # 布块对照：去重前沿每条边独立布块（第一次找）与最终保留块。
    method = dedup_stats['method'] if dedup_stats else 'none'
    total_m = sum(e['length_m'] for e in edges)
    manifest['files']['road_topology'] = relative_path(layout.figure(state['output_dir'], 'road_topology'))
    render_road_topology(state['image'], manifest['files']['road_topology'], config, edges=edges, nodes=nodes,
                         title=f'road topology: {len(edges)} edges, {len(nodes)} nodes, {total_m:.0f} m')
    layouts = {'blocks_final': (blocks, f'final blocks ({method}): {len(blocks)}')}
    if config.dedup_enabled:
        layouts['blocks_initial'] = (initial_blocks, f'initial per-edge blocks (before dedup): {len(initial_blocks)}')
    for name, (selected, title) in layouts.items():
        manifest['files'][name] = relative_path(layout.figure(state['output_dir'], name))
        render_block_layout(state['image'], manifest['files'][name], config, edges=edges, blocks=selected, title=title)
    write_json(manifest_path, manifest)
    return {'manifest_path': manifest_path}


def refresh_global_manifest(state: PreprocessState) -> dict:
    manifest = read_json(state['manifest_path'])
    if state.get('pipeline_version') is not None:
        manifest['pipeline_version'] = state['pipeline_version']
    for name in ('semantic_features', 'elevation_features', 'vlm_candidate_detection', 'block_features'):
        if name in state:
            manifest['extensions'][name] = state[name]
    # 本节点自身的快照在返回后才写入；node_state_path 预先给出它的编号。
    own = layout.node_state_path(state['output_dir'], 'refresh_global_manifest')
    paths = sorted({*own.parent.glob('[0-9][0-9]_*.json'), own})
    manifest['files']['node_states'] = {path.stem: relative_path(path) for path in paths}
    # 最终结论：没有退化 / 有退化 / 不完整，下游只看这里就不会把“没跑完”误当成“没退化”
    from ..utils.final_summary import build_final_summary
    final = build_final_summary(state)
    manifest['final_summary'] = final
    write_json(state['manifest_path'], manifest)
    logging.getLogger(f"map_preprocess.{state.get('task_id')}").info('最终结论（%s）：%s', final['conclusion'], final['message'])
    return {'final_summary': final}
