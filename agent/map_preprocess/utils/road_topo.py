#!/usr/bin/env python3
"""融合道路提取 mask 和语义分割 mask，生成干净、平直、圆滑的道路拓扑图。

LangGraph 的 Node 3（clean_road_graph）通过 run_arrays 和 to_pipeline_graph 调用本模块。
也可以单独运行（在项目根目录）：
    python -m agent.map_preprocess.utils.road_topo --road road.png --sem sem.png --gsd 0.5 --out out_dir [--config pipeline.yaml] [--geotiff source.tif]

输入都是单通道、同尺寸的图：
    道路 mask：0/255（或 0/1）二值图；也接受 0-255 概率图，按 road_thresh 二值化
    语义 mask：像素值为类别 ID，类别表在 pipeline.yaml 的 road_topo.classes 里

输出（out_dir 下）：
    graph.pkl       networkx.MultiGraph；节点属性 pos=(x, y)、type=junction/endpoint；
                    边属性 pts（Nx2 有序点列，x, y）、length_m、width_m、source=core/filled、filled_m（其中补断口的长度）
    network.geojson 节点 Point、边 LineString；给了 --geotiff 时是 WGS84 经纬度，否则是像素坐标
    stats.json      每一步的统计
    vis/*.png       每个处理阶段一张可视化图
    overlay.png     给了 --geotiff 时，最终路网叠在影像上的图
    summary.png     初始骨架图和最终图的对比，灰色虚线是删掉的分支，绿色是补上的连接
    stages.png      全部阶段的拼图
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import networkx as nx
import numpy as np
import sknw
import yaml
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, substring
from skimage.morphology import skeletonize

DEFAULT_CONFIG = Path(__file__).resolve().parents[1]/'pipeline.yaml'


# ---------- 配置与输入 ----------

def _merge(base: dict, extra: dict) -> None:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def _namespace(value):
    return SimpleNamespace(**{k: _namespace(v) for k, v in value.items()}) if isinstance(value, dict) else value


def _plain(value):
    return {k: _plain(v) for k, v in vars(value).items()} if isinstance(value, SimpleNamespace) else value


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> SimpleNamespace:
    with open(path or DEFAULT_CONFIG, encoding='utf-8') as handle:
        values = yaml.safe_load(handle)
    values = values.get('road_topo', values)   # 流程配置里是 road_topo 一节
    if overrides:
        _merge(values, overrides)
    return _namespace(values)


def load_single_channel(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f'无法读取 {path}')
    if image.ndim != 2:
        raise ValueError(f'{path} 必须是单通道图，实际形状 {image.shape}')
    return image


def road_probability(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """道路 mask → (0~1 概率, 是否已是二值图)。"""
    values = np.unique(image[::max(1, image.shape[0]//2000), ::max(1, image.shape[1]//2000)])
    if np.issubdtype(image.dtype, np.floating):
        return image.astype(np.float32), False
    binary = values.size <= 2 and set(values.tolist()) <= {0, 1, 255}
    if binary:
        return (image > 0).astype(np.float32), True
    return image.astype(np.float32)/float(np.iinfo(image.dtype).max), False


# ---------- 1. mask 预处理 ----------

def _ellipse(metres: float, gsd: float) -> np.ndarray:
    size = max(1, round(metres/gsd)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def remove_small_components(mask: np.ndarray, min_area_m2: float, gsd: float) -> tuple[np.ndarray, int]:
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    small = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < min_area_m2/gsd**2)+1
    return (mask*(~np.isin(labels, small))).astype(np.uint8), int(small.size)


def preprocess_masks(road: np.ndarray, semantic: np.ndarray, cfg: SimpleNamespace, gsd: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """道路概率二值化得到 R，语义道路类得到 S；两者分别闭运算并删除小连通域。"""
    probability, binary = road_probability(road)
    R = ((probability >= cfg.preprocess.road_thresh) & (semantic != cfg.classes.nodata_id)).astype(np.uint8)
    S = (semantic == cfg.classes.road_id).astype(np.uint8)
    info = {'input_binary': binary}
    if cfg.steps.preprocess:
        kernel = _ellipse(cfg.preprocess.close_m, gsd)
        R = cv2.morphologyEx(R, cv2.MORPH_CLOSE, kernel)
        S = cv2.morphologyEx(S, cv2.MORPH_CLOSE, kernel)
        R, info['R_small_removed'] = remove_small_components(R, cfg.preprocess.min_area_m2, gsd)
        S, info['S_small_removed'] = remove_small_components(S, cfg.preprocess.min_area_m2, gsd)
    return R, S, info


# ---------- 2. 融合 ----------

def fuse_masks(R: np.ndarray, S: np.ndarray, cfg: SimpleNamespace, gsd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """core = R∩S；s_only 丢弃；r_only 的连通块只有桥接 core 或细长地延长主干时才并入。

    返回 (融合 mask, core, 未并入的 r_only, 统计)。
    """
    core = (R & S).astype(np.uint8)
    r_only = (R & (1-S)).astype(np.uint8)
    info = {'core_px': int(core.sum()), 'r_only_px': int(r_only.sum()), 's_only_discarded_px': int((S & (1-R)).sum()),
            'bridge_blocks': 0, 'extension_blocks': 0, 'rejected_blocks': 0}
    if not cfg.steps.fuse:
        return core.copy(), core, r_only, info
    _, core_labels = cv2.connectedComponents(core, connectivity=8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(r_only, connectivity=8)
    height, width = R.shape
    accepted = np.zeros(count, bool)
    kernel = np.ones((3, 3), np.uint8)
    for index in range(1, count):
        left, top, box_w, box_h, _ = stats[index]
        x0, y0, x1, y1 = max(left-1, 0), max(top-1, 0), min(left+box_w+1, width), min(top+box_h+1, height)
        block = (labels[y0:y1, x0:x1] == index).astype(np.uint8)
        touched = np.unique(core_labels[y0:y1, x0:x1][cv2.dilate(block, kernel) > 0])
        touched = touched[touched > 0]
        if touched.size == 0:
            info['rejected_blocks'] += 1
        elif touched.size >= 2:
            accepted[index] = True
            info['bridge_blocks'] += 1
        elif _is_elongated(block, cfg, gsd):
            accepted[index] = True
            info['extension_blocks'] += 1
        else:
            info['rejected_blocks'] += 1
    merged = accepted[labels]
    fused = (core | merged).astype(np.uint8)
    return fused, core, (r_only & ~merged).astype(np.uint8), info


def _is_elongated(block: np.ndarray, cfg: SimpleNamespace, gsd: float) -> bool:
    rows, columns = np.nonzero(block)
    (_, (side_a, side_b), _) = cv2.minAreaRect(np.column_stack([columns, rows]).astype(np.float32))
    if max(side_a, side_b)/max(1., min(side_a, side_b)) > cfg.fusion.elongation_ratio:
        return True
    return float(skeletonize(block > 0).sum())*gsd > cfg.fusion.elongation_min_skeleton_m


def _line_pixels(start: np.ndarray, end: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    count = max(2, int(np.ceil(np.linalg.norm(end-start)))+1)
    x = np.clip(np.round(np.linspace(start[0], end[0], count)).astype(int), 0, shape[1]-1)
    y = np.clip(np.round(np.linspace(start[1], end[1], count)).astype(int), 0, shape[0]-1)
    return x, y


def _hits_impassable(semantic: np.ndarray, start: np.ndarray, end: np.ndarray, cfg: SimpleNamespace) -> bool:
    x, y = _line_pixels(np.asarray(start, float), np.asarray(end, float), semantic.shape)
    return bool(np.isin(semantic[y, x], cfg.classes.impassable_ids).any())


def _skeleton_ends(skeleton: np.ndarray, tail_px: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """骨架端点及其向外的单位方向（端点减去末端一段骨架的均值）。"""
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8),
                             borderType=cv2.BORDER_CONSTANT)
    ends = []
    radius = int(math.ceil(tail_px))
    for y, x in zip(*np.where(skeleton & (neighbors == 1))):
        top, left = max(y-radius, 0), max(x-radius, 0)
        rows, columns = np.nonzero(skeleton[top:y+radius+1, left:x+radius+1])
        near = np.column_stack([columns+left, rows+top]).astype(float)
        point = np.array([x, y], float)
        near = near[np.linalg.norm(near-point, axis=1) <= tail_px]
        vector = point-near.mean(axis=0)
        if np.linalg.norm(vector) > 1e-6:
            ends.append((point, vector/np.linalg.norm(vector)))
    return ends


def bridge_occlusions(mask: np.ndarray, semantic: np.ndarray, cfg: SimpleNamespace, gsd: float) -> tuple[np.ndarray, list[dict]]:
    """遮挡补路：两个路端相对、缝隙主要是可穿越类、完全不碰不可穿越类时，按路宽补上这段路。"""
    settings = cfg.occlusion
    skeleton = skeletonize(mask > 0)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ends = _skeleton_ends(skeleton, settings.tail_m/gsd)
    if len(ends) < 2:
        return mask, []
    limit = math.cos(math.radians(settings.max_angle_deg))
    positions = np.array([e[0] for e in ends])
    pairs = sorted((float(np.linalg.norm(positions[i]-positions[j])), i, j)
                   for i, j in cKDTree(positions).query_pairs(settings.max_gap_m/gsd))
    bridged = mask.copy()
    used: set[int] = set()
    records: list[dict] = []
    for length, i, j in pairs:
        if i in used or j in used or length < 1:
            continue
        (p, p_dir), (q, q_dir) = ends[i], ends[j]
        direction = (q-p)/length
        if float(np.dot(p_dir, direction)) < limit or float(np.dot(q_dir, -direction)) < limit:
            continue
        x, y = _line_pixels(p, q, mask.shape)
        classes = semantic[y, x]
        outside = mask[y, x] == 0    # 只看缝隙里的像元
        if not outside.any() or np.isin(classes, cfg.classes.impassable_ids).any():
            continue
        if float(np.isin(classes[outside], cfg.classes.traversable_ids).mean()) < settings.min_traversable:
            continue
        width = max(settings.min_width_m/gsd, float(distance[int(p[1]), int(p[0])]+distance[int(q[1]), int(q[0])]))
        cv2.line(bridged, tuple(int(v) for v in p), tuple(int(v) for v in q), 1, max(1, round(width)))
        used.update((i, j))
        records.append({'from_xy': p.tolist(), 'to_xy': q.tolist(), 'gap_m': round(length*gsd, 2)})
    # 补路笔宽会蹭到建筑，建筑上的像元一律抹掉
    if records:
        bridged[(bridged > 0) & (mask == 0) & np.isin(semantic, cfg.classes.impassable_ids)] = 0
    return bridged, records


def fill_holes(mask: np.ndarray, max_area_m2: float, gsd: float) -> tuple[np.ndarray, int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1-mask, connectivity=4)
    height, width = mask.shape
    holes = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] < max_area_m2/gsd**2
             and stats[i, cv2.CC_STAT_LEFT] > 0 and stats[i, cv2.CC_STAT_TOP] > 0
             and stats[i, cv2.CC_STAT_LEFT]+stats[i, cv2.CC_STAT_WIDTH] < width
             and stats[i, cv2.CC_STAT_TOP]+stats[i, cv2.CC_STAT_HEIGHT] < height]
    filled = mask.copy()
    if holes:
        filled[np.isin(labels, holes)] = 1
    return filled, len(holes)


def smooth_mask(mask: np.ndarray, cfg: SimpleNamespace, gsd: float) -> np.ndarray:
    kernel = _ellipse(cfg.cleanup.smooth_m, gsd)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


def mark_area_regions(mask: np.ndarray, semantic: np.ndarray, cfg: SimpleNamespace, gsd: float) -> tuple[np.ndarray, int]:
    """面状区域标记：局部宽度超过 max_road_width，或语义是停车场类。返回 (区域标签图, 区域数)。"""
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    half = cfg.area.max_road_width_m/2/gsd
    seed = (distance > half).astype(np.uint8)
    region = np.zeros_like(mask)
    if seed.any():
        # 与原先边长 2·ceil(half)+1 的椭圆核膨胀等价（+0.3 像元最贴近 OpenCV 椭圆核的离散边缘，差异约为核面积的 0.1%）。
        # 精确距离变换为 O(像元数)；大核膨胀在 0.05 m 大图上要十几分钟（河西 30 m 宽度对应 601 像元的核）。
        reach = cv2.distanceTransform(1-seed, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        region = (reach <= max(1, math.ceil(half))+.3).astype(np.uint8) & mask
    if cfg.classes.parking_ids:
        region |= ((mask > 0) & np.isin(semantic, cfg.classes.parking_ids)).astype(np.uint8)
    count, labels = cv2.connectedComponents(region.astype(np.uint8), connectivity=8)
    return labels, count-1


# ---------- 3. 骨架化 ----------

def polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.


def skeleton_graph(mask: np.ndarray, gsd: float) -> tuple[nx.MultiGraph, np.ndarray]:
    skeleton = skeletonize(mask > 0)
    source = sknw.build_sknw(skeleton.astype(np.uint16), multi=True)
    graph = nx.MultiGraph()
    for node, data in source.nodes(data=True):
        row, column = np.asarray(data['o'], float)
        graph.add_node(int(node), pos=(float(column), float(row)))
    for a, b, data in source.edges(data=True):
        points = np.asarray(data['pts'], float)[:, ::-1]
        if len(points) < 2:   # 相邻路口之间没有中间像元
            points = np.array([graph.nodes[int(a)]['pos'], graph.nodes[int(b)]['pos']], float)
        graph.add_edge(int(a), int(b), pts=points, length_m=polyline_length(points)*gsd)
    return graph, skeleton


def path_from(graph: nx.MultiGraph, start: int, data: dict) -> np.ndarray:
    """边的点列，方向从 start 出发。"""
    points, origin = data['pts'], np.asarray(graph.nodes[start]['pos'])
    return points if np.linalg.norm(points[0]-origin) <= np.linalg.norm(points[-1]-origin) else points[::-1]


def _sample(points: np.ndarray, image: np.ndarray) -> np.ndarray:
    xs = np.clip(points[:, 0].astype(int), 0, image.shape[1]-1)
    ys = np.clip(points[:, 1].astype(int), 0, image.shape[0]-1)
    return image[ys, xs]


def edge_width(data: dict, ctx: SimpleNamespace) -> float:
    """边的平均路宽（米）：边上各点到路面边界距离的两倍取均值。"""
    if '_fixed' in data:
        return data['_fixed'][0]
    if '_width' not in data:
        values = _sample(data['pts'], ctx.distance)
        positive = values[values > 0]
        data['_width'] = float(2*positive.mean()*ctx.gsd) if positive.size else 0.
    return data['_width']


def edge_source(data: dict, ctx: SimpleNamespace) -> str:
    """core 或 filled：边上多数点落在 core 上才算 core；第 6 步补的直线一律是 filled。"""
    if data.get('bridge'):
        return 'filled'
    if '_fixed' in data:
        return data['_fixed'][1]
    if '_core' not in data:
        data['_core'] = float(_sample(data['pts'], ctx.core_near).mean())
    return 'core' if data['_core'] >= .5 else 'filled'


# ---------- 4. 剔除零碎分支 ----------

def _drop_edge(graph: nx.MultiGraph, a: int, b: int, key: int, ctx: SimpleNamespace) -> None:
    ctx.removed.append(graph[a][b][key]['pts'].copy())
    graph.remove_edge(a, b, key)


def _clear_isolates(graph: nx.MultiGraph) -> None:
    graph.remove_nodes_from(list(nx.isolates(graph)))


def remove_degree_two(graph: nx.MultiGraph, gsd: float) -> int:
    """合并度为 2 的节点，把两侧的点列按方向拼接。"""
    removed = 0
    changed = True
    while changed:
        changed = False
        for node in list(graph.nodes):
            if node not in graph or graph.degree(node) != 2 or graph.has_edge(node, node):
                continue
            (_, first, first_data), (_, second, second_data) = list(graph.edges(node, data=True))
            left, right = path_from(graph, node, first_data), path_from(graph, node, second_data)
            points = np.vstack([left[::-1], right[1:]])
            filled = sum(d.get('filled_m', d['length_m'] if d.get('bridge') else 0.) for d in (first_data, second_data))
            graph.remove_node(node)
            graph.add_edge(first, second, pts=points, length_m=polyline_length(points)*gsd, **({'filled_m': filled} if filled else {}))
            removed += 1
            changed = True
    return removed


def prune_spurs(graph: nx.MultiGraph, ctx: SimpleNamespace) -> int:
    """毛刺：一端度为 1，边长小于 max(spur_min_m, spur_width_factor × 所接主路宽度)。"""
    settings = ctx.cfg.prune
    spurs = []
    for a, b, key, data in graph.edges(keys=True, data=True):
        tips = [n for n in (a, b) if graph.degree(n) == 1]
        if a == b or len(tips) != 1:
            continue
        hub = b if tips[0] == a else a
        main = max((edge_width(other, ctx) for _, _, other in graph.edges(hub, data=True) if other is not data), default=0.)
        if data['length_m'] < max(settings.spur_min_m, settings.spur_width_factor*main):
            spurs.append((a, b, key))
    for edge in spurs:
        _drop_edge(graph, *edge, ctx)
    return len(spurs)


def prune_low_confidence(graph: nx.MultiGraph, ctx: SimpleNamespace) -> int:
    """低置信分支：一端度为 1、source=filled、长度小于 filled_min_m。"""
    weak = [(a, b, k) for a, b, k, d in graph.edges(keys=True, data=True)
            if a != b and (graph.degree(a) == 1 or graph.degree(b) == 1)
            and d['length_m'] < ctx.cfg.prune.filled_min_m and edge_source(d, ctx) == 'filled']
    for edge in weak:
        _drop_edge(graph, *edge, ctx)
    return len(weak)


def prune_area_branches(graph: nx.MultiGraph, ctx: SimpleNamespace) -> int:
    """面状区域内只保留穿过区域的主干（各出入口之间的最短路），删掉内部的网状分支。"""
    removed = 0
    for region in range(1, ctx.region_count+1):
        inside = [(a, b, k) for a, b, k, d in graph.edges(keys=True, data=True)
                  if float((_sample(d['pts'], ctx.regions) == region).mean()) >= .5]
        if not inside:
            continue
        inner = set(inside)
        members = {n for a, b, _ in inside for n in (a, b)}
        gates = [n for n in members if any((a, b, k) not in inner and (b, a, k) not in inner for a, b, k in graph.edges(n, keys=True))]
        simple = nx.Graph()
        for a, b, key in inside:
            length = graph[a][b][key]['length_m']
            if a != b and (not simple.has_edge(a, b) or length < simple[a][b]['weight']):
                simple.add_edge(a, b, weight=length, key=key)
        keep: set[tuple] = set()
        for index, start in enumerate(gates):
            for end in gates[index+1:]:
                if start in simple and end in simple and nx.has_path(simple, start, end):
                    path = nx.shortest_path(simple, start, end, weight='weight')
                    keep.update((min(a, b), max(a, b), simple[a][b]['key']) for a, b in zip(path, path[1:]))
        for a, b, key in inside:
            if (min(a, b), max(a, b), key) not in keep:
                _drop_edge(graph, a, b, key, ctx)
                removed += 1
    return removed


def prune_small_cycles(graph: nx.MultiGraph, ctx: SimpleNamespace) -> int:
    """小环：周长小于 min_cycle_m 的环，删掉环上最短的边。按边长升序检查，当前边就是它所在小环的最短边。"""
    limit = ctx.cfg.prune.min_cycle_m
    simple = nx.Graph()
    for a, b, data in graph.edges(data=True):
        if a != b and (not simple.has_edge(a, b) or data['length_m'] < simple[a][b]['weight']):
            simple.add_edge(a, b, weight=data['length_m'])
    removed = 0
    for length, a, b, key in sorted((d['length_m'], a, b, k) for a, b, k, d in graph.edges(keys=True, data=True)):
        if length >= limit:
            break
        if a == b:
            _drop_edge(graph, a, b, key, ctx)
            removed += 1
            continue
        parallel = [d['length_m'] for k, d in graph[a][b].items() if k != key]
        if parallel:
            if length+min(parallel) < limit:
                _drop_edge(graph, a, b, key, ctx)
                simple[a][b]['weight'] = min(parallel)
                removed += 1
            continue
        weight = simple[a][b]['weight']
        simple.remove_edge(a, b)
        other = nx.single_source_dijkstra_path_length(simple, a, cutoff=limit-length, weight='weight').get(b)
        if other is not None and length+other < limit:
            _drop_edge(graph, a, b, key, ctx)
            removed += 1
        else:
            simple.add_edge(a, b, weight=weight)
    return removed


def prune_isolated(graph: nx.MultiGraph, ctx: SimpleNamespace) -> int:
    """孤立碎片：总长度小于 min_component_m 的连通分量。"""
    removed = 0
    for nodes in list(nx.connected_components(graph)):
        if sum(d['length_m'] for *_, d in graph.subgraph(nodes).edges(data=True)) < ctx.cfg.prune.min_component_m:
            for a, b, key in list(graph.subgraph(nodes).edges(keys=True)):
                _drop_edge(graph, a, b, key, ctx)
            graph.remove_nodes_from(nodes)
            removed += 1
    return removed


def prune_branches(graph: nx.MultiGraph, ctx: SimpleNamespace) -> dict:
    """第 4 步：迭代执行各剔除规则，直到图不再变化（最多 rounds 轮）。"""
    steps = ctx.cfg.steps
    rules = [('spurs', steps.prune_spurs, prune_spurs), ('low_confidence', steps.prune_low_confidence, prune_low_confidence),
             ('area_branches', steps.prune_area and ctx.region_count > 0, prune_area_branches),
             ('small_cycles', steps.prune_cycles, prune_small_cycles), ('isolated_components', steps.prune_isolated, prune_isolated)]
    totals = {name: 0 for name, _, _ in rules}
    totals['degree2_merged'] = remove_degree_two(graph, ctx.gsd)
    rounds = 0
    for rounds in range(1, ctx.cfg.prune.rounds+1):
        changed = 0
        line = {}
        for name, enabled, rule in rules:
            count = rule(graph, ctx) if enabled else 0
            _clear_isolates(graph)
            line[name] = count
            totals[name] += count
            changed += count
        merged = remove_degree_two(graph, ctx.gsd)
        totals['degree2_merged'] += merged
        ctx.log(f'    第 {rounds} 轮 {line} 合并度 2 节点 {merged}')
        if changed == 0 and merged == 0:
            break
    totals['rounds'] = rounds
    return totals


# ---------- 5. 路口合并与规整 ----------

def _contract(graph: nx.MultiGraph, members: list[int], center: tuple[float, float], ctx: SimpleNamespace) -> int:
    """把 members 并成位于 center 的新节点，相连的边接到新节点。"""
    new = max(graph.nodes)+1
    graph.add_node(new, pos=tuple(float(v) for v in center))
    inside = set(members)
    for a, b, key, data in list(graph.edges(members, keys=True, data=True)):
        start, end = (a, b) if a in inside else (b, a)
        if a in inside and b in inside:
            if data['length_m'] >= ctx.cfg.junction.self_loop_min_m:
                points = np.vstack([center, path_from(graph, start, data), center])
                graph.add_edge(new, new, pts=points, length_m=polyline_length(points)*ctx.gsd)
            graph.remove_edge(a, b, key)
            continue
        points = np.vstack([center, path_from(graph, start, data)])
        graph.remove_edge(a, b, key)
        graph.add_edge(new, end, pts=points, length_m=polyline_length(points)*ctx.gsd)
        ctx.modified += 1
    graph.remove_nodes_from(members)
    return new


def merge_close_nodes(graph: nx.MultiGraph, ctx: SimpleNamespace) -> dict:
    """距离小于 merge_m 的节点聚类合并，坐标取均值；隔着不可穿越类的两个节点不合并。"""
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        return {'merged_nodes': 0, 'clusters': 0}
    positions = np.array([graph.nodes[n]['pos'] for n in nodes], float)
    links = nx.Graph()
    links.add_nodes_from(nodes)
    for i, j in cKDTree(positions).query_pairs(ctx.cfg.junction.merge_m/ctx.gsd):
        if not _hits_impassable(ctx.semantic, positions[i], positions[j], ctx.cfg):
            links.add_edge(nodes[i], nodes[j])
    clusters = [sorted(c) for c in nx.connected_components(links) if len(c) > 1]
    for members in clusters:
        _contract(graph, members, tuple(np.mean([graph.nodes[n]['pos'] for n in members], axis=0).tolist()), ctx)
    return {'merged_nodes': sum(map(len, clusters)), 'clusters': len(clusters), 'degree2_merged': remove_degree_two(graph, ctx.gsd)}


def _loop_shape(ring: np.ndarray, gsd: float) -> tuple[float, float]:
    """把闭合环近似成长方形，返回 (宽, 长)，单位米。"""
    polygon = Polygon(ring)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    area = polygon.area*gsd**2
    half = polyline_length(np.vstack([ring, ring[:1]]))*gsd/2
    width = (half-math.sqrt(max(half*half-4*area, 0.)))/2
    return width, half-width


def _resample_count(points: np.ndarray, count: int) -> np.ndarray:
    arcs = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    samples = np.linspace(0., arcs[-1], count)
    return np.column_stack([np.interp(samples, arcs, points[:, 0]), np.interp(samples, arcs, points[:, 1])])


def merge_narrow_loops(graph: nx.MultiGraph, ctx: SimpleNamespace) -> dict:
    """窄长环（隔离带把一条路分成两条线）：两条平行边合并成中线；多条边围成的窄长环删掉最窄的那条边。"""
    settings, gsd = ctx.cfg.loop, ctx.gsd
    info = {'parallel_merged': 0, 'loop_edges_removed': 0}
    def narrow(ring: np.ndarray) -> bool:
        width, length = _loop_shape(ring, gsd)
        return width < settings.narrow_width_m and length > settings.min_elongation*max(width, 1e-6)
    for _ in range(200):
        pair = None
        for a, b in {(min(a, b), max(a, b)) for a, b in graph.edges() if a != b}:
            if graph.number_of_edges(a, b) < 2:
                continue
            first, second = sorted(graph[a][b], key=lambda k: graph[a][b][k]['length_m'])[:2]
            one, two = path_from(graph, a, graph[a][b][first]), path_from(graph, a, graph[a][b][second])
            if narrow(np.vstack([one, two[::-1]])):
                pair = (a, b, first, second, one, two)
                break
        if pair is None:
            break
        a, b, first, second, one, two = pair
        data_one, data_two = graph[a][b][first], graph[a][b][second]
        fixed = (edge_width(data_one, ctx)+edge_width(data_two, ctx),
                 'core' if 'core' in (edge_source(data_one, ctx), edge_source(data_two, ctx)) else 'filled')
        if data_two['length_m'] <= settings.parallel_length_ratio*data_one['length_m']:
            count = max(len(one), len(two), 2)
            middle = (_resample_count(one, count)+_resample_count(two, count))/2
            middle[0], middle[-1] = graph.nodes[a]['pos'], graph.nodes[b]['pos']
            ctx.removed.append(one.copy())
        else:
            middle = one
        ctx.removed.append(two.copy())
        graph.remove_edge(a, b, first)
        graph.remove_edge(a, b, second)
        graph.add_edge(a, b, pts=middle, length_m=polyline_length(middle)*gsd, _fixed=fixed)
        info['parallel_merged'] += 1
    for _ in range(200):
        simple = nx.Graph()
        for a, b, key, data in graph.edges(keys=True, data=True):
            if a != b and (not simple.has_edge(a, b) or data['length_m'] < simple[a][b]['weight']):
                simple.add_edge(a, b, weight=data['length_m'], key=key)
        target = None
        for cycle in nx.minimum_cycle_basis(simple, weight='weight'):
            ordered = nx.find_cycle(simple.subgraph(cycle))
            ring = np.vstack([path_from(graph, a, graph[a][b][simple[a][b]['key']]) for a, b in ordered])
            if narrow(ring):
                target = min(((a, b, simple[a][b]['key']) for a, b in ordered),
                             key=lambda e: (edge_width(graph[e[0]][e[1]][e[2]], ctx), -graph[e[0]][e[1]][e[2]]['length_m']))
                break
        if target is None:
            break
        _drop_edge(graph, *target, ctx)
        remove_degree_two(graph, gsd)
        info['loop_edges_removed'] += 1
    info['degree2_merged'] = remove_degree_two(graph, gsd)
    return info


def _arm_fit(graph: nx.MultiGraph, node: int, data: dict, ctx: SimpleNamespace) -> SimpleNamespace:
    """取边上离路口 [skip_m, skip_m+fit_span_m] 的点拟合方向；direction 指向远离路口的一侧。"""
    settings = ctx.cfg.junction
    points = path_from(graph, node, data)
    arcs = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    low = min(settings.skip_m/ctx.gsd, arcs[-1]*.3)
    window = points[(arcs >= low) & (arcs <= low+settings.fit_span_m/ctx.gsd)]
    if len(window) < 2:
        window = points
    center = window.mean(axis=0)
    direction = np.linalg.svd(window-center, full_matrices=False)[2][0]
    if float(np.dot(direction, window[-1]-window[0])) < 0:
        direction = -direction
    return SimpleNamespace(points=points, arcs=arcs, low=low, window=window, center=center, direction=direction)


def _deviation(arm_a: SimpleNamespace, arm_b: SimpleNamespace) -> float:
    """两条边偏离共线的角度：0° 表示方向正好相反。"""
    return math.degrees(math.acos(float(np.clip(np.dot(arm_a.direction, -arm_b.direction), -1., 1.))))


def _pair_arms(arms: list[SimpleNamespace], limit: float) -> list[tuple[int, int]]:
    """度为 4 时尝试分成两对共线的边；否则返回偏差最小的一对；都不共线时返回空表。"""
    count = len(arms)
    deviations = {(i, j): _deviation(arms[i], arms[j]) for i in range(count) for j in range(i+1, count)}
    if count == 4:
        pairings = [p for p in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))) if all(deviations[e] < limit for e in p)]
        if pairings:
            return list(min(pairings, key=lambda p: sum(deviations[e] for e in p)))
    best = min(deviations, key=deviations.get)
    return [best] if deviations[best] < limit else []


def contract_double_nodes(graph: nx.MultiGraph, ctx: SimpleNamespace) -> dict:
    """两个度为 3 的节点之间的短边：四条外侧边能分成两对共线时，当作一个度为 4 的路口。"""
    settings = ctx.cfg.junction
    contracted = rejected = 0
    limit = settings.double_node_m
    for a, b, key, data in sorted(graph.edges(keys=True, data=True), key=lambda e: e[3]['length_m']):
        if data['length_m'] >= limit:
            break
        if a == b or a not in graph or b not in graph or not graph.has_edge(a, b, key):
            continue
        if graph.degree(a) != 3 or graph.degree(b) != 3 or graph.number_of_edges(a, b) != 1 or graph.has_edge(a, a) or graph.has_edge(b, b):
            continue
        outer = [_arm_fit(graph, a, d, ctx) for _, n, d in graph.edges(a, data=True) if n != b]
        outer += [_arm_fit(graph, b, d, ctx) for _, n, d in graph.edges(b, data=True) if n != a]
        if len(_pair_arms(outer, settings.collinear_deg)) == 2:
            center = tuple(np.mean([graph.nodes[a]['pos'], graph.nodes[b]['pos']], axis=0).tolist())
            _contract(graph, [a, b], center, ctx)
            contracted += 1
        else:
            rejected += 1
    return {'double_nodes': contracted, 'kept_apart': rejected}


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    return center, np.linalg.svd(points-center, full_matrices=False)[2][0]


def _intersect(p: np.ndarray, u: np.ndarray, q: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, float] | None:
    """直线 p+t·u 与 q+s·v 的交点；返回 (交点, t)，近似平行时返回 None。"""
    denominator = u[0]*v[1]-u[1]*v[0]
    if abs(denominator) < 1e-6:
        return None
    t = float(((q[0]-p[0])*v[1]-(q[1]-p[1])*v[0])/denominator)
    return p+t*u, t


def _cut(arm: SimpleNamespace) -> np.ndarray:
    """跳过路口附近 low 以内的点，返回其余点列。"""
    return arm.points[int(np.searchsorted(arm.arcs, arm.low)):]


def _tail_on_line(arm: SimpleNamespace, c: np.ndarray, u: np.ndarray, ctx: SimpleNamespace, keep: np.ndarray | None = None) -> np.ndarray | None:
    """主路一侧保留的点列：从跳过 skip_m 之后第一个回到拟合直线上的点开始，切掉路口的弯折。

    keep 是额外的布尔条件；拟合范围内没有点回到直线上时，退回第一个满足条件的点，都没有时返回 None。
    """
    settings = ctx.cfg.junction
    normal = np.array([-u[1], u[0]])
    near_line = np.abs(np.dot(arm.points-c, normal)) <= settings.on_line_m/ctx.gsd
    allowed = arm.arcs >= arm.low
    if keep is not None:
        allowed &= keep
    within = allowed & near_line & (arm.arcs <= arm.low+settings.fit_span_m/ctx.gsd)
    indices = np.flatnonzero(within) if within.any() else np.flatnonzero(allowed)
    return arm.points[int(indices[0]):] if indices.size else None


def _rewrite(data: dict, points: np.ndarray, gsd: float) -> None:
    data['pts'] = points
    data['length_m'] = polyline_length(points)*gsd
    data.pop('_width', None)
    data.pop('_core', None)


def _regularize_cross(graph: nx.MultiGraph, node: int, incident: list, arms: list, pairs: list, ctx: SimpleNamespace) -> bool:
    (i, j), (k, m) = pairs
    p, u = _fit_line(np.vstack([arms[i].window, arms[j].window]))
    q, v = _fit_line(np.vstack([arms[k].window, arms[m].window]))
    crossing = _intersect(p, u, q, v)
    if crossing is None:
        return False
    point = crossing[0]
    if np.linalg.norm(point-np.asarray(graph.nodes[node]['pos']))*ctx.gsd > ctx.cfg.junction.extend_max_m:
        return False
    lines = {i: (p, u), j: (p, u), k: (q, v), m: (q, v)}
    tails = [_tail_on_line(arm, *lines[index], ctx) for index, arm in enumerate(arms)]
    if any(tail is None or len(tail) < 1 for tail in tails):
        return False
    graph.nodes[node]['pos'] = tuple(float(v) for v in point)
    for (_, _, _, data), tail in zip(incident, tails):
        _rewrite(data, np.vstack([point, tail]), ctx.gsd)
        ctx.modified += 1
    return True


def _branch_target(arm: SimpleNamespace, c: np.ndarray, u: np.ndarray, ctx: SimpleNamespace) -> np.ndarray | None:
    """支路沿自身方向延长与主路求交；不相交或延长太远时取垂足；穿过不可穿越类时放弃。"""
    settings = ctx.cfg.junction
    start = _cut(arm)[0]
    foot = c+float(np.dot(start-c, u))*u
    crossing = _intersect(arm.center, arm.direction, c, u)
    candidates = []
    if crossing is not None and crossing[1] < 0 and np.linalg.norm(crossing[0]-start)*ctx.gsd <= settings.extend_max_m:
        candidates.append(crossing[0])
    candidates.append(foot)
    for point in candidates:
        if not _hits_impassable(ctx.semantic, start, point, ctx.cfg):
            return point
    return None


def _regularize_main(graph: nx.MultiGraph, node: int, incident: list, arms: list, pair: tuple[int, int], ctx: SimpleNamespace) -> int:
    """主路拉直，支路接到主路上；交点分开时插入新节点把主路拆成多段。返回新增节点数，放弃时返回 -1。"""
    settings = ctx.cfg.junction
    i, j = pair
    c, u = _fit_line(np.vstack([arms[i].window, arms[j].window]))
    origin = np.asarray(graph.nodes[node]['pos'], float)
    branches = [n for n in range(len(arms)) if n not in pair]
    places = []
    for branch in branches:
        target = _branch_target(arms[branch], c, u, ctx)
        if target is None:
            return -1
        places.append((float(np.dot(target-c, u)), branch))
    places.sort()
    clusters: list[list[tuple[float, int]]] = [[places[0]]]
    for place in places[1:]:
        if (place[0]-clusters[-1][-1][0])*ctx.gsd < settings.merge_m:
            clusters[-1].append(place)
        else:
            clusters[-1:] = [clusters[-1], [place]]
    centers = [c+float(np.mean([s for s, _ in cluster]))*u for cluster in clusters]
    if any(np.linalg.norm(point-origin)*ctx.gsd > settings.extend_max_m for point in centers):
        return -1
    scale = [float(np.dot(point-c, u)) for point in centers]
    plus, minus = (i, j) if float(np.dot(arms[i].direction, u)) > 0 else (j, i)
    if float(np.dot(arms[plus].direction, u)) <= 0 or float(np.dot(arms[minus].direction, u)) >= 0:
        return -1
    def main_tail(arm: SimpleNamespace, limit: float, sign: float) -> np.ndarray | None:
        """主路一侧的点列必须在交点外侧，防止主路折返。"""
        return _tail_on_line(arm, c, u, ctx, keep=sign*(np.dot(arm.points-c, u)-limit) > 1.)
    tail_plus, tail_minus = main_tail(arms[plus], scale[-1], 1.), main_tail(arms[minus], scale[0], -1.)
    if tail_plus is None or tail_minus is None:
        return -1
    nearest = int(np.argmin([np.linalg.norm(point-origin) for point in centers]))
    ids = []
    for index, point in enumerate(centers):
        if index == nearest:
            graph.nodes[node]['pos'] = tuple(float(v) for v in point)
            ids.append(node)
        else:
            new = max(graph.nodes)+1
            graph.add_node(new, pos=tuple(float(v) for v in point))
            ids.append(new)
    def attach(number: int, target: int, point: np.ndarray, tail: np.ndarray) -> None:
        _, far, key, data = incident[number]
        points = np.vstack([point, tail])
        if target == node:
            _rewrite(data, points, ctx.gsd)
        else:
            graph.remove_edge(node, far, key)
            graph.add_edge(target, far, pts=points, length_m=polyline_length(points)*ctx.gsd)
        ctx.modified += 1
    attach(plus, ids[-1], centers[-1], tail_plus)
    attach(minus, ids[0], centers[0], tail_minus)
    for index, cluster in enumerate(clusters):
        for _, branch in cluster:
            attach(branch, ids[index], centers[index], _cut(arms[branch]))
    for index in range(len(ids)-1):
        points = np.vstack([centers[index], centers[index+1]])
        graph.add_edge(ids[index], ids[index+1], pts=points, length_m=polyline_length(points)*ctx.gsd)
        ctx.added.append(points.copy())
    return len(ids)-1


def regularize_junctions(graph: nx.MultiGraph, ctx: SimpleNamespace) -> dict:
    """规整路口：共线的两条边算主路并拉直，支路接到主路上；找不到共线的边时是真正的 Y 形，保持不动。"""
    info = {'crossroads': 0, 't_junctions': 0, 'kept_y': 0, 'inserted_nodes': 0, 'skipped': 0}
    for node in list(graph.nodes):
        if node not in graph or graph.degree(node) < 3 or graph.has_edge(node, node):
            continue
        incident = list(graph.edges(node, keys=True, data=True))
        arms = [_arm_fit(graph, node, data, ctx) for *_, data in incident]
        pairs = _pair_arms(arms, ctx.cfg.junction.collinear_deg)
        if not pairs:
            info['kept_y'] += 1
        elif len(pairs) == 2:
            if _regularize_cross(graph, node, incident, arms, pairs, ctx):
                info['crossroads'] += 1
            else:
                info['skipped'] += 1
        else:
            inserted = _regularize_main(graph, node, incident, arms, pairs[0], ctx)
            if inserted < 0:
                info['skipped'] += 1
            else:
                info['t_junctions'] += 1
                info['inserted_nodes'] += inserted
    return info


# ---------- 6. 补断口 ----------

def _tail_direction(points: np.ndarray, tail_px: float) -> np.ndarray:
    """路端（points[-1]）向外的单位方向：用末端约 tail_px 的点拟合直线。"""
    arcs = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points[::-1], axis=0), axis=1))]
    tail = points[::-1][arcs <= tail_px]
    if len(tail) < 2:
        tail = points[-2:][::-1]
    center = tail.mean(axis=0)
    direction = np.linalg.svd(tail-center, full_matrices=False)[2][0]
    return direction if float(np.dot(direction, tail[0]-tail[-1])) >= 0 else -direction


def bridge_gaps(graph: nx.MultiGraph, ctx: SimpleNamespace) -> dict:
    """两个度为 1 的端点距离和方向满足阈值、连线不穿过不可穿越类时，用直线连上。

    max_m 以内用宽松的方向阈值；方向几乎正对时放宽到 long_max_m，方向用更长的路段拟合。
    """
    settings = ctx.cfg.gap
    tips = [n for n in graph.nodes if graph.degree(n) == 1]
    if len(tips) < 2:
        return {'added_edges': 0, 'detail': []}
    ends = {}
    for node in tips:
        _, _, data = next(iter(graph.edges(node, data=True)))
        path = path_from(graph, node, data)[::-1]
        ends[node] = (np.asarray(graph.nodes[node]['pos'], float), _tail_direction(path, settings.tail_m/ctx.gsd), _tail_direction(path, settings.long_tail_m/ctx.gsd))
    positions = np.array([ends[n][0] for n in tips])
    limit = math.cos(math.radians(settings.max_angle_deg))
    long_limit = math.cos(math.radians(settings.long_max_angle_deg))
    pairs = sorted((float(np.linalg.norm(positions[i]-positions[j])), tips[i], tips[j])
                   for i, j in cKDTree(positions).query_pairs(settings.long_max_m/ctx.gsd))
    used: set[int] = set()
    detail = []
    for length, a, b in pairs:
        if a in used or b in used or length < 1:
            continue
        (p, p_dir, p_long), (q, q_dir, q_long) = ends[a], ends[b]
        direction = (q-p)/length
        if length*ctx.gsd < settings.max_m:
            aligned = float(np.dot(p_dir, direction)) >= limit and float(np.dot(q_dir, -direction)) >= limit
        else:
            aligned = float(np.dot(p_long, direction)) >= long_limit and float(np.dot(q_long, -direction)) >= long_limit
        if not aligned or _hits_impassable(ctx.semantic, p, q, ctx.cfg):
            continue
        points = np.vstack([p, q])
        graph.add_edge(a, b, pts=points, length_m=length*ctx.gsd, bridge=True)
        ctx.added.append(points.copy())
        used.update((a, b))
        detail.append({'u': a, 'v': b, 'gap_m': round(length*ctx.gsd, 2)})
    return {'added_edges': len(detail), 'detail': detail}


def connect_ends_to_roads(graph: nx.MultiGraph, ctx: SimpleNamespace) -> dict:
    """路端接主路：度为 1 的端点沿自身方向在 max_m 内碰到别的边、连线不穿过不可穿越类时接上，在落点处拆分那条边。"""
    settings, gsd = ctx.cfg.gap, ctx.gsd
    reach, snap = settings.max_m/gsd, ctx.cfg.junction.merge_m/gsd
    limit = math.cos(math.radians(settings.max_angle_deg))
    detail = []
    for node in [n for n in graph.nodes if graph.degree(n) == 1]:
        if node not in graph or graph.degree(node) != 1:
            continue
        _, other, own = next(iter(graph.edges(node, data=True)))
        p = np.asarray(graph.nodes[node]['pos'], float)
        direction = _tail_direction(path_from(graph, node, own)[::-1], settings.tail_m/gsd)
        ray, origin = LineString([p, p+direction*reach]), Point(p)
        best = None
        for a, b, key, data in graph.edges(keys=True, data=True):
            if data is own or len(data['pts']) < 2:
                continue
            line = LineString(data['pts'])
            if line.distance(origin) > reach:
                continue
            hit = ray.intersection(line)
            q = np.asarray(nearest_points(origin, line if hit.is_empty else hit)[1].coords[0], float)
            distance = float(np.linalg.norm(q-p))
            if distance < 1 or (hit.is_empty and float(np.dot((q-p)/distance, direction)) < limit):
                continue
            if best is None or distance < best[0]:
                best = (distance, a, b, key, q, line)
        if best is None:
            continue
        distance, a, b, key, q, line = best
        data = graph[a][b][key]
        along = line.project(Point(q))
        first = a if np.linalg.norm(data['pts'][0]-np.asarray(graph.nodes[a]['pos'])) <= np.linalg.norm(data['pts'][0]-np.asarray(graph.nodes[b]['pos'])) else b
        last = b if first == a else a
        target = first if along < snap else last if line.length-along < snap else None
        if a == b and target is not None:
            target = a
        end = np.asarray(graph.nodes[target]['pos'], float) if target is not None else q
        if target == node or _hits_impassable(ctx.semantic, p, end, ctx.cfg):
            continue
        gap_m = float(np.linalg.norm(end-p))*gsd
        if other in (a, b):   # 接上后会和自己的路口围成很小的环
            back = along if other == first else line.length-along
            if own['length_m']+gap_m+back*gsd < ctx.cfg.prune.min_cycle_m:
                continue
        if target is None:
            target = max(graph.nodes)+1
            graph.add_node(target, pos=(float(q[0]), float(q[1])))
            keep = {k: v for k, v in data.items() if k in ('bridge', '_fixed')}
            for part, far in ((substring(line, 0, along), first), (substring(line, along, line.length), last)):
                points = np.asarray(part.coords, float)
                graph.add_edge(target, far, pts=points, length_m=polyline_length(points)*gsd, **keep)
            graph.remove_edge(a, b, key)
            ctx.modified += 1
        points = np.vstack([p, end])
        graph.add_edge(node, target, pts=points, length_m=gap_m, bridge=True)
        ctx.added.append(points.copy())
        detail.append({'u': node, 'v': target, 'gap_m': round(gap_m, 2)})
    return {'added_edges': len(detail), 'detail': detail}


# ---------- 7. 平直与圆滑 ----------

def _douglas_peucker(points: np.ndarray, tolerance: float) -> np.ndarray:
    if len(points) < 3 or np.array_equal(points[0], points[-1]):
        return points.copy()
    simplified = np.asarray(LineString(points).simplify(tolerance, preserve_topology=False).coords, float)
    simplified[0], simplified[-1] = points[0], points[-1]
    return simplified


def _chord_deviation(points: np.ndarray) -> float:
    chord = points[-1]-points[0]
    length = float(np.linalg.norm(chord))
    if length < 1e-6:
        return float(np.linalg.norm(points-points[0], axis=1).max())
    offset = points-points[0]
    return float(np.abs(offset[:, 0]*chord[1]-offset[:, 1]*chord[0]).max()/length)


def _chaikin(points: np.ndarray, iterations: int) -> np.ndarray:
    current = points
    for _ in range(iterations):
        q = .75*current[:-1]+.25*current[1:]
        r = .25*current[:-1]+.75*current[1:]
        current = np.vstack([current[0], np.column_stack([q, r]).reshape(-1, 2), current[-1]])
    return current


def _resample(points: np.ndarray, step: float) -> np.ndarray:
    arcs = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    count = max(2, int(round(arcs[-1]/step))+1)
    samples = np.linspace(0., arcs[-1], count)
    return np.column_stack([np.interp(samples, arcs, points[:, 0]), np.interp(samples, arcs, points[:, 1])])


def straighten_and_smooth(graph: nx.MultiGraph, ctx: SimpleNamespace) -> dict:
    """端点不动：近似直线的边换成直线，其余用 Chaikin 圆滑并重采样；路口 skip_m 内保持规整结果，偏差过大时退回简化结果。"""
    settings, gsd = ctx.cfg.smooth, ctx.gsd
    tolerance = settings.simplify_m/gsd
    skip = ctx.cfg.junction.skip_m/gsd
    info = {'straightened': 0, 'smoothed': 0, 'reverted': 0, 'simplified_only': 0}
    for a, b, data in graph.edges(data=True):
        points = data['pts']
        data['_ref'] = points   # 平滑前的几何，最后一步用它限制累计偏差
        if len(points) < 3 or a == b:
            info['simplified_only'] += 1
            _rewrite_keep(data, _douglas_peucker(points, tolerance), gsd)
            continue
        if _chord_deviation(points)*gsd < settings.straight_tol_m:
            info['straightened'] += 1
            _rewrite_keep(data, np.vstack([points[0], points[-1]]), gsd)
            continue
        start = a if np.linalg.norm(points[0]-np.asarray(graph.nodes[a]['pos'])) <= np.linalg.norm(points[0]-np.asarray(graph.nodes[b]['pos'])) else b
        end = b if start == a else a
        arcs = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        head = int(np.searchsorted(arcs, skip, side='right'))-1 if graph.degree(start) >= 3 else 0
        tail = int(np.searchsorted(arcs, arcs[-1]-skip, side='left')) if graph.degree(end) >= 3 else len(points)-1
        fallback = _douglas_peucker(points, tolerance)
        if tail-head < 2:
            info['simplified_only'] += 1
            _rewrite_keep(data, fallback, gsd)
            continue
        curve = _resample(_chaikin(_douglas_peucker(points[head:tail+1], tolerance), settings.chaikin_iterations), settings.resample_m/gsd)
        smooth = np.vstack([_douglas_peucker(points[:head+1], tolerance)[:-1], curve, _douglas_peucker(points[tail:], tolerance)[1:]])
        if LineString(smooth).hausdorff_distance(LineString(points))*gsd > settings.max_deviation_m:
            info['reverted'] += 1
            _rewrite_keep(data, fallback, gsd)
        else:
            info['smoothed'] += 1
            _rewrite_keep(data, smooth, gsd)
    return info


def _rewrite_keep(data: dict, points: np.ndarray, gsd: float) -> None:
    """改几何但保留已算好的路宽和来源。"""
    data['pts'] = points
    data['length_m'] = polyline_length(points)*gsd


def _fillet(base: np.ndarray, reach: float, step: float, min_turn_deg: float) -> tuple[np.ndarray, int, float]:
    """折线的每个拐角用二次贝塞尔弧倒圆，弧的两端各占 reach（不超过相邻线段的一半）。返回 (点列, 弧数, 剩余直线长度)。"""
    segments = np.diff(base, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    out, arcs, straight = [base[0]], 0, float(lengths.sum())
    for index in range(1, len(base)-1):
        before, after = segments[index-1]/lengths[index-1], segments[index]/lengths[index]
        turn = math.degrees(math.acos(float(np.clip(np.dot(before, after), -1., 1.))))
        if turn < min_turn_deg:
            out.append(base[index])
            continue
        cut = min(reach, .5*lengths[index-1], .5*lengths[index])
        start, end = base[index]-before*cut, base[index]+after*cut
        t = np.linspace(0., 1., max(3, int(math.ceil(2*cut/step))+1))[:, None]
        out.extend((1-t)**2*start+2*t*(1-t)*base[index]+t**2*end)
        arcs += 1
        straight -= 2*cut
    out.append(base[-1])
    points = np.asarray(out, float)
    distinct = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]   # 相邻两段弧首尾相接时会有重复点
    distinct[-1] = True
    return points[distinct], arcs, straight


def refine_curvature(graph: nx.MultiGraph, ctx: SimpleNamespace) -> dict:
    """曲率与直线优化：每条边拟合成若干直线段，拐角用圆滑弧线过渡；与平滑前路径的偏差超限时缩小圆角，仍超限则不改。"""
    settings, gsd = ctx.cfg.refine, ctx.gsd
    info = {'refined': 0, 'already_straight': 0, 'kept': 0, 'arcs': 0}
    straight_px = total_px = 0.
    for a, b, data in graph.edges(data=True):
        points = data['pts']
        reference = LineString(data.get('_ref', points)) if len(data.get('_ref', points)) >= 2 else None
        length = polyline_length(points)
        total_px += length
        if len(points) < 3 or a == b or reference is None:
            info['already_straight'] += 1
            straight_px += length if len(points) == 2 else 0.
            continue
        base = _douglas_peucker(points, settings.straight_tol_m/gsd)
        for reach in (settings.fillet_m, settings.fillet_m/2, settings.fillet_m/4):
            refined, arcs, straight = _fillet(base, reach/gsd, settings.resample_m/gsd, settings.min_turn_deg)
            if LineString(refined).hausdorff_distance(reference)*gsd <= settings.max_deviation_m:
                _rewrite_keep(data, refined, gsd)
                info['refined'] += 1
                info['arcs'] += arcs
                straight_px += straight
                break
        else:
            info['kept'] += 1
    info['straight_share'] = round(float(straight_px/total_px), 3) if total_px else 0.
    return info


# ---------- 8. 统计与可视化 ----------

def graph_stats(graph: nx.MultiGraph) -> dict:
    degrees = [d for _, d in graph.degree]
    return {'nodes': graph.number_of_nodes(), 'edges': graph.number_of_edges(), 'deg1': degrees.count(1), 'deg3plus': sum(d >= 3 for d in degrees),
            'components': nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
            'length_km': round(sum(d['length_m'] for *_, d in graph.edges(data=True))/1000, 3)}


def _dashed(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], on: float = 7., off: float = 5.) -> None:
    arcs = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    for begin in np.arange(0., arcs[-1], on+off):
        samples = np.linspace(begin, min(begin+on, arcs[-1]), 4)
        dash = np.column_stack([np.interp(samples, arcs, points[:, 0]), np.interp(samples, arcs, points[:, 1])]).astype(np.int32)
        cv2.polylines(image, [dash], False, color, 1, cv2.LINE_AA)


class Reporter:
    GREY, GREEN, YELLOW = (150, 150, 150), (60, 220, 60), (0, 220, 255)

    def __init__(self, out: Path, shape: tuple[int, int], cfg: SimpleNamespace, log=print):
        self.log = log
        self.out, self.directory = out, out/'vis'
        self.scale = min(1., cfg.vis.max_px/max(shape))
        self.records: list[dict] = []
        self.all_removed: list[np.ndarray] = []
        self.all_added: list[np.ndarray] = []
        self.raw: np.ndarray | None = None
        self.clock = time.time()
        self.directory.mkdir(parents=True, exist_ok=True)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        if self.scale >= 1:
            return image
        return cv2.resize(image, (max(1, round(image.shape[1]*self.scale)), max(1, round(image.shape[0]*self.scale))), interpolation=cv2.INTER_AREA)

    def _title(self, image: np.ndarray, title: str) -> np.ndarray:
        canvas = image.copy()
        cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def mask(self, name: str, mask: np.ndarray, title: str, extra: dict | None = None, layers: tuple = ()) -> None:
        pixels = int((mask > 0).sum())
        seconds, self.clock = round(time.time()-self.clock, 1), time.time()
        self.records.append({'step': name, 'road_pixels': pixels, 'seconds': seconds, **(extra or {})})
        shown = {k: v for k, v in (extra or {}).items() if k != 'detail'}
        self.log(f'[{name}] {seconds} s 道路像元 {pixels} {shown}')
        image = cv2.cvtColor(self._resize((mask > 0).astype(np.uint8)*150), cv2.COLOR_GRAY2BGR)
        for layer, color in layers:
            if layer.any():
                image[self._resize(layer.astype(np.uint8)*255) > 60] = color
        cv2.imwrite(str(self.directory/f'{name}.png'), self._title(image, title))

    def _render(self, graph: nx.MultiGraph, background: np.ndarray, ctx: SimpleNamespace, removed: list, added: list) -> np.ndarray:
        image = cv2.cvtColor(self._resize((background > 0).astype(np.uint8)*60), cv2.COLOR_GRAY2BGR)
        for points in removed:
            _dashed(image, points*self.scale, self.GREY)
        for *_, data in graph.edges(data=True):
            color = self.GREEN if edge_source(data, ctx) == 'filled' else self.YELLOW
            cv2.polylines(image, [(data['pts']*self.scale).astype(np.int32)], False, color, 1, cv2.LINE_AA)
        for points in added:
            cv2.polylines(image, [(points*self.scale).astype(np.int32)], False, self.GREEN, 2, cv2.LINE_AA)
        radius = max(3, round(4*self.scale**.5))
        for node, position in graph.nodes(data='pos'):
            center = tuple(int(v*self.scale) for v in position)
            degree = graph.degree(node)
            if degree >= 3:
                cv2.circle(image, center, radius, (0, 0, 255), -1, cv2.LINE_AA)
            elif degree == 1:
                cv2.circle(image, center, radius, (255, 80, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(image, center, radius-1, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def graph(self, name: str, graph: nx.MultiGraph, background: np.ndarray, title: str, ctx: SimpleNamespace, extra: dict | None = None) -> None:
        stats = graph_stats(graph)
        change = {'removed': len(ctx.removed), 'added': len(ctx.added), 'modified': ctx.modified}
        seconds, self.clock = round(time.time()-self.clock, 1), time.time()
        self.records.append({'step': name, 'seconds': seconds, **stats, **change, **(extra or {})})
        shown = {k: v for k, v in (extra or {}).items() if k != 'detail'}
        self.log(f"[{name}] {seconds} s 节点 {stats['nodes']} 边 {stats['edges']} 度1 {stats['deg1']} 度≥3 {stats['deg3plus']} 连通分量 {stats['components']} "
              f"总长 {stats['length_km']} km | 删除 {change['removed']} 新增 {change['added']} 修改 {change['modified']} {shown}")
        image = self._render(graph, background, ctx, ctx.removed, ctx.added)
        cv2.imwrite(str(self.directory/f'{name}.png'), self._title(image, f"{title}  n={stats['nodes']} e={stats['edges']}"))
        if self.raw is None:
            self.raw = self._title(image, f"before: skeleton n={stats['nodes']} e={stats['edges']}")
        self.all_removed += ctx.removed
        self.all_added += ctx.added
        ctx.removed, ctx.added, ctx.modified = [], [], 0

    def overlay(self, graph: nx.MultiGraph, geotiff: str, ctx: SimpleNamespace, shape: tuple[int, int]) -> None:
        """最终路网叠在影像上：黄色是 core，绿色是 filled，红点路口，蓝点路端。"""
        import rasterio
        size = (max(1, round(shape[0]*self.scale)), max(1, round(shape[1]*self.scale)))
        with rasterio.open(geotiff) as source:
            if source.count < 3:
                return
            image = source.read([1, 2, 3], out_shape=(3, *size), resampling=rasterio.enums.Resampling.average)
        if image.dtype != np.uint8:
            image = cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image = np.ascontiguousarray(image.transpose(1, 2, 0)[:, :, ::-1])
        for *_, data in graph.edges(data=True):
            points = [(data['pts']*self.scale).astype(np.int32)]
            cv2.polylines(image, points, False, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.polylines(image, points, False, self.GREEN if edge_source(data, ctx) == 'filled' else self.YELLOW, 2, cv2.LINE_AA)
        for node, position in graph.nodes(data='pos'):
            center = tuple(int(v*self.scale) for v in position)
            cv2.circle(image, center, 5, (0, 0, 255) if graph.degree(node) >= 3 else (255, 80, 0), -1, cv2.LINE_AA)
        cv2.imwrite(str(self.out/'overlay.png'), image)

    def summary(self, graph: nx.MultiGraph, background: np.ndarray, ctx: SimpleNamespace) -> None:
        stats = graph_stats(graph)
        final = self._title(self._render(graph, background, ctx, self.all_removed, self.all_added),
                            f"after: n={stats['nodes']} e={stats['edges']} (grey dashed = removed, green = filled)")
        cv2.imwrite(str(self.out/'summary.png'), np.hstack([self.raw, np.full((self.raw.shape[0], 6, 3), 255, np.uint8), final]))
        tiles = [cv2.imread(str(p)) for p in sorted(self.directory.glob('*.png'))]
        width = 640
        tiles = [cv2.resize(t, (width, round(t.shape[0]*width/t.shape[1])), interpolation=cv2.INTER_AREA) for t in tiles]
        blank = np.zeros_like(tiles[0])
        rows = [np.hstack((tiles[i:i+4]+[blank]*4)[:4]) for i in range(0, len(tiles), 4)]
        cv2.imwrite(str(self.out/'stages.png'), np.vstack(rows))


# ---------- 输出 ----------

def _coordinate_transform(geotiff: str | None, shape: tuple[int, int]):
    """返回 (x, y 点列 → 输出坐标) 的函数和坐标说明；没有地理参考时输出像素坐标。"""
    if geotiff is None:
        return (lambda points: np.asarray(points, float)), {'type': 'pixel'}
    import rasterio
    from rasterio.warp import transform as warp
    with rasterio.open(geotiff) as source:
        if (source.height, source.width) != shape:
            raise ValueError(f'GeoTIFF 尺寸 {(source.height, source.width)} 与 mask {shape} 不一致')
        if source.crs is None:
            raise ValueError(f'{geotiff} 没有坐标系')
        transform, crs = source.transform, source.crs
    def convert(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, float)
        xs, ys = rasterio.transform.xy(transform, values[:, 1], values[:, 0], offset='center')
        lon, lat = warp(crs, 'EPSG:4326', list(np.atleast_1d(xs)), list(np.atleast_1d(ys)))
        return np.column_stack([lon, lat])
    return convert, {'type': 'geographic', 'crs': 'EPSG:4326', 'source_crs': str(crs)}


def finalize_attributes(graph: nx.MultiGraph, ctx: SimpleNamespace) -> None:
    """写入节点 type 和边的 width_m、source，并清掉计算缓存。"""
    for node in graph.nodes:
        graph.nodes[node]['type'] = 'junction' if graph.degree(node) >= 3 else 'endpoint'
    for *_, data in graph.edges(data=True):
        data['width_m'] = round(edge_width(data, ctx), 2)
        data['source'] = edge_source(data, ctx)
        data['filled_m'] = round(float(data['length_m'] if data.get('bridge') else data.get('filled_m', 0.)), 2)
        for key in ('_width', '_core', '_fixed', '_ref', 'bridge'):
            data.pop(key, None)


def write_outputs(graph: nx.MultiGraph, out: Path, geotiff: str | None, shape: tuple[int, int]) -> dict:
    convert, coordinate = _coordinate_transform(geotiff, shape)
    with open(out/'graph.pkl', 'wb') as handle:
        pickle.dump(graph, handle)
    features = []
    for node, data in graph.nodes(data=True):
        x, y = convert(np.array([data['pos']], float))[0]
        features.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [float(x), float(y)]},
                         'properties': {'kind': 'node', 'id': int(node), 'type': data['type'], 'degree': graph.degree(node), 'pos': [float(v) for v in data['pos']]}})
    for index, (a, b, data) in enumerate(graph.edges(data=True)):
        features.append({'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': convert(data['pts']).tolist()},
                         'properties': {'kind': 'edge', 'id': index, 'u': int(a), 'v': int(b), 'length_m': round(float(data['length_m']), 3),
                                        'width_m': data['width_m'], 'source': data['source'], 'filled_m': data['filled_m']}})
    with open(out/'network.geojson', 'w', encoding='utf-8') as handle:
        json.dump({'type': 'FeatureCollection', 'coordinates_kind': coordinate, 'features': features}, handle, ensure_ascii=False)
    return coordinate


# ---------- 主流程 ----------

def to_pipeline_graph(graph: nx.MultiGraph, resolution: float, image_size: tuple[int, int]) -> nx.MultiGraph:
    """转成 LangGraph 下游使用的格式：节点 ID 是字符串，节点 xy 和边 polyline_xy 用像素边缘坐标（像素中心 +0.5）。"""
    limit = np.array(image_size, float)
    def place(points: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(points, float)+.5, 0., limit)
    result = nx.MultiGraph()
    for node, data in graph.nodes(data=True):
        result.add_node(str(node), xy=place(data['pos']).tolist(), node_type=data['type'])
    records = []
    for a, b, data in graph.edges(data=True):
        origin = data['pts'][0]
        near_a = np.linalg.norm(origin-np.asarray(graph.nodes[a]['pos'])) <= np.linalg.norm(origin-np.asarray(graph.nodes[b]['pos']))
        u, v = (a, b) if near_a else (b, a)
        points = place(data['pts'])
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
        points = points[keep]
        line = points.tolist()
        line[0], line[-1] = list(result.nodes[str(u)]['xy']), list(result.nodes[str(v)]['xy'])
        records.append({'u': str(u), 'v': str(v), 'polyline_xy': line, 'length_m': polyline_length(points)*resolution, 'confidence': None,
                        'source': 'samroad', 'semantic_fraction': 0., 'road_topo_source': data['source'],
                        'width_m': data['width_m'], 'filled_m': data['filled_m']})
    for index, edge in enumerate(sorted(records, key=lambda e: (e['polyline_xy'], e['u'], e['v']))):
        edge['edge_id'] = str(index)
        result.add_edge(edge['u'], edge['v'], key=edge['edge_id'], **edge)
    return result


def run(road_path: str | Path, semantic_path: str | Path, gsd: float, out: str | Path, cfg: SimpleNamespace | None = None,
        geotiff: str | None = None) -> tuple[nx.MultiGraph, list[dict]]:
    road, semantic = load_single_channel(road_path), load_single_channel(semantic_path)
    graph, records, _ = run_arrays(road, semantic, gsd, out, cfg, geotiff=geotiff)
    return graph, records


def run_arrays(road: np.ndarray, semantic: np.ndarray, gsd: float, out: str | Path, cfg: SimpleNamespace | None = None,
               geotiff: str | None = None, vis_out: str | Path | None = None, log=print) -> tuple[nx.MultiGraph, list[dict], list[np.ndarray]]:
    """完整流程。out 放数据（graph.pkl、network.geojson、stats.json），vis_out 放图片，默认同 out。返回 (图, 每步统计, 删掉的分支点列)。"""
    cfg = cfg or load_config()
    steps = cfg.steps
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    vis_out = Path(vis_out) if vis_out is not None else out
    if semantic.shape != road.shape:
        raise ValueError(f'语义 mask 尺寸 {semantic.shape} 与道路 mask {road.shape} 不一致')
    report = Reporter(vis_out, road.shape, cfg, log)

    R, S, info = preprocess_masks(road, semantic, cfg, gsd)
    report.mask('01_road_R', R, '1 road extraction R', info)
    report.mask('02_semantic_S', S, '2 semantic road S')
    mask, core, rejected, info = fuse_masks(R, S, cfg, gsd)
    report.mask('03_fused', mask, '3 fused (green=r_only kept, red=r_only rejected)', info,
                layers=(((mask > 0) & (core == 0), (60, 220, 60)), (rejected > 0, (0, 0, 255))))
    bridges: list[dict] = []
    before = mask.copy()
    if steps.occlusion:
        mask, bridges = bridge_occlusions(mask, semantic, cfg, gsd)
    report.mask('04_occlusion', mask, '4 occlusion bridges (green)', {'bridges': len(bridges), 'detail': bridges}, layers=(((mask > 0) & (before == 0), (60, 220, 60)),))
    holes = 0
    if steps.fill_holes:
        mask, holes = fill_holes(mask, cfg.cleanup.hole_m2, gsd)
    if steps.smooth_mask:
        mask = smooth_mask(mask, cfg, gsd)
    report.mask('05_cleaned', mask, '5 holes filled, boundary smoothed', {'filled_holes': holes})
    regions, region_count = np.zeros_like(mask, np.int32), 0
    if steps.area_regions:
        regions, region_count = mark_area_regions(mask, semantic, cfg, gsd)
    report.mask('06_area_regions', mask, '6 area regions (green)', {'regions': region_count}, layers=((regions > 0, (60, 220, 60)),))

    tolerance = cfg.fusion.source_tolerance_m/gsd
    core_near = cv2.distanceTransform((core == 0).astype(np.uint8), cv2.DIST_L2, 3) <= tolerance
    ctx = SimpleNamespace(cfg=cfg, gsd=gsd, semantic=semantic, core_near=core_near, distance=cv2.distanceTransform(mask, cv2.DIST_L2, 5),
                          regions=regions, region_count=region_count, removed=[], added=[], modified=0, log=log)
    graph, _ = skeleton_graph(mask, gsd)
    report.graph('07_skeleton_graph', graph, mask, '7 skeleton graph', ctx)
    if any((steps.prune_spurs, steps.prune_low_confidence, steps.prune_area, steps.prune_cycles, steps.prune_isolated)):
        report.graph('08_pruned', graph, mask, '8 fragments pruned', ctx, prune_branches(graph, ctx))
    if steps.merge_junctions:
        report.graph('09a_merged_nodes', graph, mask, '9a close nodes merged', ctx, merge_close_nodes(graph, ctx))
    if steps.narrow_loops:
        report.graph('09a2_narrow_loops', graph, mask, '9a2 narrow loops merged', ctx, merge_narrow_loops(graph, ctx))
    if steps.double_nodes:
        report.graph('09b_double_nodes', graph, mask, '9b double nodes contracted', ctx, contract_double_nodes(graph, ctx))
    if steps.regularize_junctions:
        report.graph('09c_regularized', graph, mask, '9c junctions regularized', ctx, regularize_junctions(graph, ctx))
    if steps.bridge_gaps:
        report.graph('10_gaps', graph, mask, '10 gaps bridged (green)', ctx, bridge_gaps(graph, ctx))
    if steps.connect_ends:
        report.graph('10b_end_to_road', graph, mask, '10b road ends joined to roads (green)', ctx, connect_ends_to_roads(graph, ctx))
    if steps.bridge_gaps or steps.connect_ends:
        report.graph('10c_merge_old_ends', graph, mask, '10c old end nodes merged', ctx, {'degree2_merged': remove_degree_two(graph, gsd)})
    if steps.straighten_smooth:
        report.graph('11_smoothed', graph, mask, '11 straightened and smoothed', ctx, straighten_and_smooth(graph, ctx))
    if steps.refine_curvature:
        report.graph('12_refined', graph, mask, '12 straight lines and rounded curves', ctx, refine_curvature(graph, ctx))
    report.graph('13_final', graph, mask, '13 final', ctx)
    report.summary(graph, mask, ctx)
    if geotiff is not None:
        report.overlay(graph, geotiff, ctx, road.shape)

    finalize_attributes(graph, ctx)
    coordinate = write_outputs(graph, out, geotiff, road.shape)
    with open(out/'stats.json', 'w', encoding='utf-8') as handle:
        json.dump({'config': _plain(cfg), 'gsd_m_per_px': gsd, 'coordinates': coordinate, 'steps': report.records}, handle, ensure_ascii=False, indent=1, default=str)
    return graph, report.records, report.all_removed


def main() -> None:
    parser = argparse.ArgumentParser(description='道路 mask + 语义 mask → 道路拓扑图')
    parser.add_argument('--road', required=True, help='道路提取 mask：单通道 0/255 二值图或 0-255 概率图')
    parser.add_argument('--sem', required=True, help='语义 mask：单通道，像素值为类别 ID')
    parser.add_argument('--gsd', type=float, required=True, help='米/像素')
    parser.add_argument('--out', required=True, help='输出目录')
    parser.add_argument('--config', help='YAML 配置，默认用 agent/map_preprocess/pipeline.yaml 的 road_topo 一节')
    parser.add_argument('--geotiff', help='与 mask 同尺寸的 GeoTIFF；给了就输出 WGS84 经纬度')
    arguments = parser.parse_args()
    run(arguments.road, arguments.sem, arguments.gsd, arguments.out, load_config(arguments.config), geotiff=arguments.geotiff)


if __name__ == '__main__':
    main()
