"""道路拓扑优化：并行去重、近距连通、短路删除、贯通路网、曲率平滑。

都只在 degree-2 压缩后的 MultiGraph 上工作，连接关系仍以模型图为准：只有悬空路端（度 1）会被接到别的边上，
两条边的几何交叉不建路口。折线方向始终与 u→v 一致，节点坐标等于折线首尾。
"""
from __future__ import annotations

import math
from typing import Callable

import networkx as nx
import numpy as np
import shapely
from shapely.geometry import LineString

from .global_road_graph import drop_edges, edge_records, length_px, merge_close_junctions, recompress

_SAMPLE_STEP_M = 2.        # 判并行的沿边采样间隔
_TANGENT_WINDOW_M = 5.     # 局部方向取采样点前后该距离的弦，抗折线抖动
_FOOT_MARGIN_PX = 1e-6     # 垂足落在对方端点上不算并行（路口前后接续的两条边不是并行路）
_REACH_FACTOR = 2.         # 判并行时，整条边都须在对方 _REACH_FACTOR×距离 内
_FOLD_DEG = 120.           # 单个顶点转角超过该值视为折返毛刺（真实路口转弯约 90°，掉头弯由多个顶点组成）
_FOLD_ARM_SIGMAS = 3.      # 折返毛刺的短臂上限（sigma 的倍数），真实毛刺臂长约 15 m
_CUSP_DEG = 30.            # 平滑后（间距约 sigma/5）单个顶点仍超过该值视为尖点
_SMOOTH_PASSES = 3
_THROUGH_MAX_DEFLECTION_DEG = 60.  # 路口处两条边的接续偏折不超过该值才算同一条路穿过路口
_TANGENT_REACH_SIGMAS = 2.         # 路端方向取离节点该距离（sigma 的倍数）的点，抗抖动
_CORNER_DEG = 45.                  # 前后各 _CORNER_WINDOW_M 内走向变化超过该值视为真实拐角，保留原样、不做圆滑
_CORNER_WINDOW_M = 6.
_DETOUR_FACTOR = 10.       # 路端沿路网绕到目标点不足 _DETOUR_FACTOR×距离 时不连（已经连着，再连只造小环）
_THROUGH_PASSES = 30       # 贯通路网的迭代上限；每轮至少删一条边或合并一个路口，实际几轮就稳定
_CLOSE_NODE_SPREAD = 2.    # 过近节点合并组的直径上限（距离阈值的倍数），防止一长串路口被压成一点
_CYCLE_THROUGH_DEG = 35.   # 边端与路口另一条边的接续偏折不超过该值，算这条路直行穿过路口
_CYCLE_THROUGH_REACH_M = 10.  # 判直行时路端方向取离节点该距离的点，抗抖动
_SHARED_TOL_M = 4.         # 从同一节点出发的两条边相距不超过该值算贴在一起走
_SHARED_STEP_M = 1.        # 判贴合的沿边采样间隔


def _find_edge(graph: nx.MultiGraph, identifier: str) -> tuple[str, str, object, dict]:
    return next(item for item in graph.edges(keys=True, data=True) if item[3]['edge_id'] == identifier)


def _node_on_edge(graph: nx.MultiGraph, identifier: str, along_px: float, snap_px: float, resolution: float) -> str:
    """取边上弧长 along_px 处的节点：离边端不足 snap_px 时直接用端点，否则在该处打断边。"""
    a, b, key, edge = _find_edge(graph, identifier)
    shape = LineString(edge['polyline_xy'])
    if along_px <= snap_px:
        return edge['u']
    if shape.length-along_px <= snap_px:
        return edge['v']
    point = shape.interpolate(along_px)
    joint = f'r{identifier}_{round(along_px)}'
    while joint in graph:
        joint += 'x'
    graph.add_node(joint, xy=[point.x, point.y])
    line = np.asarray(edge['polyline_xy'], float)
    arcs = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    head = [*line[arcs < along_px].tolist(), [point.x, point.y]]
    tail = [[point.x, point.y], *line[arcs > along_px].tolist()]
    graph.remove_edge(a, b, key)
    for suffix, start, end, part in (('a', edge['u'], joint, head), ('b', joint, edge['v'], tail)):
        graph.add_edge(start, end, key=identifier+suffix, **{**edge, 'edge_id': identifier+suffix, 'u': start, 'v': end,
            'polyline_xy': part, 'length_m': length_px(part)*resolution, 'split_of': edge.get('split_of', identifier)})
    return joint


def _reattach(graph: nx.MultiGraph, node: str, joint: str, resolution: float) -> list[dict]:
    """把 node 上的边改接到 joint（折线向 joint 延长）并删除 node。改接后首尾相同的边没有意义，返回它们。"""
    target = list(graph.nodes[joint]['xy'])
    collapsed = []
    for a, b, key, edge in list(graph.edges(node, keys=True, data=True)):
        graph.remove_edge(a, b, key)
        line = [list(p) for p in edge['polyline_xy']]
        start, end = edge['u'], edge['v']
        if start == node:
            line.insert(0, target)
            start = joint
        if end == node:
            line.append(target)
            end = joint
        if start == end and edge['u'] != edge['v']:
            collapsed.append(edge)
            continue
        graph.add_edge(start, end, key=key, **{**edge, 'u': start, 'v': end, 'polyline_xy': line,
                                               'length_m': length_px(line)*resolution})
    graph.remove_node(node)
    return collapsed


def _sampled(line: LineString, arcs: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    """弧长 arcs 处的坐标与单位切向。"""
    def at(values):
        return shapely.get_coordinates(shapely.line_interpolate_point(line, values))
    tangent = at(np.minimum(arcs+window, line.length))-at(np.maximum(arcs-window, 0.))
    return at(arcs), tangent/np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)


def _parallel_cover(line: LineString, other: LineString, step: float, window: float,
                    distance: float, cos_limit: float) -> tuple[float, float]:
    """line 上与 other 并行的长度比例，以及 line 到 other 的最大距离。"""
    count = max(2, math.ceil(line.length/step))
    points, tangent = _sampled(line, (np.arange(count)+.5)*line.length/count, window)
    feet = shapely.line_locate_point(other, shapely.points(points))
    near, other_tangent = _sampled(other, feet, window)
    gaps = np.linalg.norm(points-near, axis=1)
    parallel = ((gaps <= distance) & (feet > _FOOT_MARGIN_PX) & (feet < other.length-_FOOT_MARGIN_PX)
                & (np.abs(np.sum(tangent*other_tangent, axis=1)) >= cos_limit))
    return float(parallel.mean()), float(gaps.max())


def _covered_pairs(graph: nx.MultiGraph, distance: float, min_overlap: float, cos_limit: float,
                   resolution: float) -> list[tuple[dict, dict]]:
    """(被覆盖边, 覆盖它的边)。互相覆盖时只保留低分边作为被覆盖方。"""
    records = [e for e in edge_records(graph) if e['u'] != e['v']]
    lines = [LineString(e['polyline_xy']) for e in records]
    tree = shapely.STRtree(lines)
    step, window = _SAMPLE_STEP_M/resolution, _TANGENT_WINDOW_M/resolution
    covered: dict[tuple[int, int], bool] = {}
    for i, j in zip(*tree.query(lines, predicate='dwithin', distance=distance)):
        if i != j:
            overlap, reach = _parallel_cover(lines[i], lines[j], step, window, distance, cos_limit)
            covered[int(i), int(j)] = overlap >= min_overlap and reach <= _REACH_FACTOR*distance
    rank = lambda e: (e['edge_score'], e['length_m'], e['edge_id'])
    return [(records[i], records[j]) for (i, j), hit in covered.items()
            if hit and not (covered.get((j, i)) and rank(records[i]) > rank(records[j]))]


def merge_parallel_roads(graph: nx.MultiGraph, distance_m: float, max_angle_deg: float, min_overlap: float,
                         resolution: float) -> list[dict]:
    """两条路并行太近时去掉一条（原地）。A 上至少 min_overlap 比例的长度与 B 相距 <= distance_m 且局部同向、
    整条 A 不离开 B 两倍距离时，A 视为 B 的重复；互相覆盖时删低分边。

    被删边端点上还挂着别的边时，这些边改接到保留边上，所以不会把路网断开。每次删一条再重新判断，
    保证判断总基于当前的边。"""
    dropped: list[dict] = []
    if distance_m <= 0:
        return dropped
    distance = distance_m/resolution
    cos_limit = math.cos(math.radians(max_angle_deg))
    while True:
        pairs = _covered_pairs(graph, distance, min_overlap, cos_limit, resolution)
        if not pairs:
            break
        loser, keeper = min(pairs, key=lambda p: (p[0]['edge_score'], p[0]['length_m'], p[0]['edge_id']))
        a, b, key, _ = _find_edge(graph, loser['edge_id'])
        graph.remove_edge(a, b, key)
        dropped.append({**loser, 'reason': 'parallel_duplicate', 'kept_edge_id': keeper['edge_id']})
        for node in (a, b):
            if not graph.degree(node):
                graph.remove_node(node)
                continue
            part = _nearest_part(graph, node, keeper.get('split_of', keeper['edge_id']))
            if node in (part['u'], part['v']):
                continue
            along = LineString(part['polyline_xy']).project(shapely.Point(graph.nodes[node]['xy']))
            joint = _node_on_edge(graph, part['edge_id'], along, distance, resolution)
            dropped.extend({**e, 'reason': 'parallel_duplicate', 'kept_edge_id': keeper['edge_id']}
                           for e in _reattach(graph, node, joint, resolution))
    if dropped:
        recompress(graph, resolution)
    return dropped


def _nearest_part(graph: nx.MultiGraph, node: str, identifier: str) -> dict:
    """identifier 这条边可能已被改接打断成几段（split_of 记录来源），返回其中离 node 最近的一段。"""
    parts = [e for *_, e in graph.edges(data=True) if e.get('split_of', e['edge_id']) == identifier]
    point = shapely.Point(graph.nodes[node]['xy'])
    return min(parts, key=lambda e: (LineString(e['polyline_xy']).distance(point), e['edge_id']))


def connect_near_roads(graph: nx.MultiGraph, distance_m: float, resolution: float) -> dict:
    """悬空路端离别的路 <= distance_m 就接上（原地）：路端折线延长到目标边，投影点不在端点时打断目标边。
    只处理度 1 的路端；沿路网本来就离目标很近的不连。"""
    stats = {'connected': 0, 'gaps_m': []}
    if distance_m <= 0:
        return stats
    distance = distance_m/resolution
    for node in sorted(n for n in graph.nodes if graph.degree(n) == 1):
        if node not in graph or graph.degree(node) != 1:
            continue
        tip = shapely.Point(graph.nodes[node]['xy'])
        reachable = nx.single_source_dijkstra_path_length(graph, node, cutoff=_DETOUR_FACTOR*distance_m,
                                                          weight=lambda _a, _b, d: min(e['length_m'] for e in d.values()))
        best = None
        for edge in edge_records(graph):
            if node in (edge['u'], edge['v']):
                continue
            shape = LineString(edge['polyline_xy'])
            gap = shape.distance(tip)
            if gap > distance:
                continue
            along = shape.project(tip)
            detour = min(reachable.get(edge['u'], math.inf)+along*resolution,
                         reachable.get(edge['v'], math.inf)+(shape.length-along)*resolution)
            if detour > _DETOUR_FACTOR*distance_m and (best is None or (gap, edge['edge_id']) < best[:2]):
                best = (gap, edge['edge_id'], along)
        if best is None:
            continue
        gap, identifier, along = best
        joint = _node_on_edge(graph, identifier, along, distance, resolution)
        _reattach(graph, node, joint, resolution)
        stats['connected'] += 1
        stats['gaps_m'].append(round(gap*resolution, 2))
    if stats['connected']:
        recompress(graph, resolution)
    return stats


def _loop_reach_m(edge: dict, resolution: float) -> float:
    points = np.asarray(edge['polyline_xy'], float)
    return float(np.linalg.norm(points-points[0], axis=1).max())*resolution


def remove_short_roads(graph: nx.MultiGraph, max_length_m: float, max_loop_reach_m: float, resolution: float) -> list[dict]:
    """删除短路和小环（原地）：一端悬空且短于 max_length_m 的边；挂在路口上、离开路口的最远距离不到
    max_loop_reach_m 的自环（整圈都在路口附近，多是语义骨架绕院子的小圈）。两端都接路口的短边是路网的一部分，
    不删，过近路口由 merge_close_junctions 处理。删除后两侧并成新边再判断，直到不变。阈值为 0 关闭对应项。"""
    dropped: list[dict] = []
    while True:
        identifiers = {e['edge_id'] for a, b, e in graph.edges(data=True)
                       if (_loop_reach_m(e, resolution) < max_loop_reach_m if a == b else
                           e['length_m'] < max_length_m and (graph.degree(a) == 1 or graph.degree(b) == 1))}
        if not identifiers:
            break
        dropped.extend(drop_edges(graph, identifiers, 'short_road'))
        recompress(graph, resolution)
    return dropped


def _through_ends(edge: dict, graph: nx.MultiGraph, reach: float) -> int:
    """边的两端有几端能接上别的边继续直行（0、1、2）。"""
    limit, count, line = math.cos(math.radians(_CYCLE_THROUGH_DEG)), 0, LineString(edge['polyline_xy'])
    for node, from_start in ((edge['u'], True), (edge['v'], False)):
        own = _outward_direction(line, from_start, reach)
        for a, b, other in graph.edges(node, data=True):
            if a != b and other['edge_id'] != edge['edge_id'] and -float(np.dot(
                    own, _outward_direction(LineString(other['polyline_xy']), other['u'] == node, reach))) >= limit:
                count += 1
                break
    return count


def _weakest(edges: list[dict], graph: nx.MultiGraph, resolution: float) -> dict:
    """环上最该去掉的边：两端能继续直行的端数最少（主路穿过路口时不被低置信度的短段打断），其次置信度最低（缺失按 1），再次最短。"""
    reach = _CYCLE_THROUGH_REACH_M/resolution
    return min(edges, key=lambda e: (_through_ends(e, graph, reach), 1. if e['confidence'] is None else e['confidence'],
                                     e['length_m'], e['edge_id']))


def _small_cycle_edges(graph: nx.MultiGraph, max_length_m: float) -> list[dict] | None:
    """周长最短且小于 max_length_m 的环上的全部边（自环、两点重边、多点环）；没有则 None。"""
    cycles = [[e] for a, b, e in graph.edges(data=True) if a == b]
    simple = nx.Graph()
    for a, b, edge in graph.edges(data=True):
        if a == b:
            continue
        if simple.has_edge(a, b):
            current = simple[a][b]['edge']
            cycles.append([current, edge])
            edge = min(current, edge, key=lambda e: e['length_m'])
        simple.add_edge(a, b, weight=edge['length_m'], edge=edge)
    for nodes in nx.minimum_cycle_basis(simple, weight='weight'):
        cycles.append([data['edge'] for *_, data in simple.subgraph(nodes).edges(data=True)])
    size = lambda edges: sum(e['length_m'] for e in edges)
    small = [edges for edges in cycles if size(edges) < max_length_m]
    return min(small, key=lambda edges: (size(edges), sorted(e['edge_id'] for e in edges))) if small else None


def remove_small_cycles(graph: nx.MultiGraph, max_length_m: float, resolution: float) -> list[dict]:
    """打开小环（原地）：每次在周长最短的小环上去掉最弱的一条边，直到没有小环。环上的边删掉一条不会断开路网。"""
    dropped: list[dict] = []
    while max_length_m > 0 and (edges := _small_cycle_edges(graph, max_length_m)) is not None:
        dropped.extend(drop_edges(graph, {_weakest(edges, graph, resolution)['edge_id']}, 'small_cycle'))
        recompress(graph, resolution)
    return dropped


def road_support(polyline: list, probability: np.ndarray, image_size: tuple[float, float], threshold: float,
                 resolution: float) -> float:
    """折线上模型道路概率 >= threshold 的长度比例（沿线等距采样，概率图双线性插值）。"""
    line = LineString(polyline)
    count = max(2, math.ceil(line.length*resolution/_SAMPLE_STEP_M))
    points = shapely.get_coordinates(shapely.line_interpolate_point(line, (np.arange(count)+.5)*line.length/count))
    height, width = probability.shape[:2]
    x = np.clip(points[:, 0]*width/image_size[0], 0, width-1.001)
    y = np.clip(points[:, 1]*height/image_size[1], 0, height-1.001)
    column, row = x.astype(int), y.astype(int)
    right, down = x-column, y-row
    values = (probability[row, column]*(1-right)*(1-down)+probability[row, column+1]*right*(1-down)
              +probability[row+1, column]*(1-right)*down+probability[row+1, column+1]*right*down)
    return float((values >= threshold).mean())


def remove_self_loops(graph: nx.MultiGraph, resolution: float) -> list[dict]:
    """删除自环（原地）：从路口出发又回到同一路口的路没有出口，不论多长都不算贯通。"""
    identifiers = {e['edge_id'] for a, b, e in graph.edges(data=True) if a == b}
    if not identifiers:
        return []
    dropped = drop_edges(graph, identifiers, 'self_loop')
    recompress(graph, resolution)
    return dropped


def remove_weak_duplicates(graph: nx.MultiGraph, support: Callable[[list], float] | None, min_support: float,
                           resolution: float) -> list[dict]:
    """两条边首尾相同时，去掉沿线概率支持低于 min_support 的边（原地），最强的一条始终保留，路网不会断开。

    清洗用的 confidence 是连接置信度，假路也可能很高，沿线概率才能区分路面上的线和压在房屋上的线。"""
    if support is None or min_support <= 0:
        return []
    groups: dict[frozenset, list[dict]] = {}
    for a, b, edge in graph.edges(data=True):
        if a != b:
            groups.setdefault(frozenset((a, b)), []).append(edge)
    scores: dict[str, float] = {}
    weak: set[str] = set()
    for edges in groups.values():
        if len(edges) < 2:
            continue
        scores.update({e['edge_id']: support(e['polyline_xy']) for e in edges})
        ranked = sorted(edges, key=lambda e: (scores[e['edge_id']], e['edge_id']))
        weak.update(e['edge_id'] for e in ranked[:-1] if scores[e['edge_id']] < min_support)
    if not weak:
        return []
    dropped = [{**record, 'road_support': scores[record['edge_id']]} for record in drop_edges(graph, weak, 'weak_duplicate')]
    recompress(graph, resolution)
    return dropped


def _shared_length(first: LineString, second: LineString, tolerance: float, step: float) -> float:
    """从同一点出发的两条折线贴在一起走的弧长。"""
    limit = min(first.length, second.length)
    arcs = np.minimum(np.arange(1, math.ceil(limit/step)+1)*step, limit)
    apart = ((shapely.distance(shapely.line_interpolate_point(first, arcs), second) > tolerance)
             | (shapely.distance(shapely.line_interpolate_point(second, arcs), first) > tolerance))
    index = int(np.argmax(apart)) if apart.any() else len(arcs)
    return float(arcs[index-1]) if index else 0.


def _most_shared_pair(graph: nx.MultiGraph, tolerance: float, step: float, minimum: float,
                      skip: Callable[[list], bool] | None = None) -> tuple | None:
    """贴合最长的一对边 (节点, 边甲, 边乙, 贴合弧长)，都不到 minimum 时返回 None。"""
    best = None
    for node in graph.nodes:
        ends = [(e, LineString(e['polyline_xy'] if e['u'] == node else e['polyline_xy'][::-1]))
                for a, b, e in graph.edges(node, data=True) if a != b]
        for index, (first, first_line) in enumerate(ends):
            for second, second_line in ends[index+1:]:
                shared = _shared_length(first_line, second_line, tolerance, step)
                if shared >= minimum and (best is None or shared > best[3]):
                    fork = first_line.interpolate(max(shared-tolerance, 0.))
                    if skip is None or not skip([fork.x, fork.y]):
                        best = (node, first, second, shared)
    return best


def split_shared_stretches(graph: nx.MultiGraph, min_length_m: float, resolution: float,
                           skip: Callable[[list], bool] | None = None) -> dict:
    """支路贴着干路走到路口时，把路口还原到真实分岔处（原地）。

    路口合并后，支路会沿干路走到合并节点，两条边在这一段重合。在分岔处打断干路建节点，支路从这里接入，重合段只留一条。
    支路整条都贴在干路上时，把它另一端的节点移到干路上。分岔位置满足 skip 时不处理。min_length_m 为 0 时关闭。"""
    if min_length_m <= 0:
        return {'splits': 0, 'shared_m': []}
    tolerance, step = _SHARED_TOL_M/resolution, _SHARED_STEP_M/resolution
    minimum = min_length_m/resolution+tolerance  # 贴合弧长比实际重合多一个容差，门槛按实际重合长度
    lengths: list[float] = []
    for _ in range(graph.number_of_edges()*2):
        found = _most_shared_pair(graph, tolerance, step, minimum, skip)
        if found is None:
            break
        node, first, second, shared = found
        trunk, branch = sorted((first, second), key=lambda e: (1. if e['confidence'] is None else e['confidence'],
                                                               e['length_m'], e['edge_id']), reverse=True)
        # 贴合的终点比真实分岔靠后约一个容差（两线要分开这么远才算离开），节点放回分岔处；支路整条贴合时节点就在它的路端
        whole = shared >= LineString(branch['polyline_xy']).length-step
        fork = shared if whole else max(shared-tolerance, step)
        lengths.append(fork*resolution)
        a, b, key, _ = _find_edge(graph, branch['edge_id'])
        graph.remove_edge(a, b, key)
        points = np.asarray(branch['polyline_xy'] if branch['u'] == node else branch['polyline_xy'][::-1], float)
        far = branch['v'] if branch['u'] == node else branch['u']
        along = fork if trunk['u'] == node else LineString(trunk['polyline_xy']).length-fork
        joint = _node_on_edge(graph, trunk['edge_id'], along, step, resolution)
        if far == joint:
            continue
        arcs = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if whole:  # 支路整条贴在干路上：另一端的路改接到干路
            if graph.has_node(far):
                _reattach(graph, far, joint, resolution)
            continue
        rest = points[arcs > fork+step].tolist()
        line = [list(graph.nodes[joint]['xy']), *rest]
        identifier = f"{branch['edge_id']}c"
        graph.add_edge(joint, far, key=identifier, **{**branch, 'edge_id': identifier, 'u': joint, 'v': far, 'polyline_xy': line,
                                                      'length_m': length_px(line)*resolution})
    if lengths:
        recompress(graph, resolution)
    return {'splits': len(lengths), 'shared_m': [round(value, 1) for value in lengths]}


def split_crowded_junctions(graph: nx.MultiGraph, gap_m: float, resolution: float, max_roads: int = 4) -> int:
    """限制一个节点接的路数（原地）：超过 max_roads 的路口拆成相距 gap_m 的两个节点。

    先找最接近一条直线的一对路作为干路，两个节点放在干路上；其余支路按朝向分到两侧，两节点间连一条短边，干路仍能直行穿过。
    拆开后仍超过上限时继续拆。gap_m 为 0 时关闭。返回拆分次数。"""
    if gap_m <= 0:
        return 0
    gap, reach, count = gap_m/resolution, _CYCLE_THROUGH_REACH_M/resolution, 0
    for _ in range(graph.number_of_nodes()*4):
        crowded = [n for n in graph.nodes if sum(1 for a, b in graph.edges(n) if a != b) > max_roads]
        if not crowded:
            break
        node = max(crowded, key=lambda n: (graph.degree(n), str(n)))
        center = np.asarray(graph.nodes[node]['xy'], float)
        arms = []
        for a, b, key, edge in graph.edges(node, keys=True, data=True):
            if a != b:
                line = LineString(edge['polyline_xy'] if edge['u'] == node else edge['polyline_xy'][::-1])
                arms.append({'edge': edge, 'key': key, 'other': b if a == node else a, 'direction': _outward_direction(line, True, reach),
                             'line': np.asarray(line.coords, float)})
        first, second = min(((i, j) for i in range(len(arms)) for j in range(i+1, len(arms))),
                            key=lambda p: (float(np.dot(arms[p[0]]['direction'], arms[p[1]]['direction'])), p))
        axis = arms[second]['direction']-arms[first]['direction']
        axis = axis/max(float(np.linalg.norm(axis)), 1e-12)
        rest = sorted((i for i in range(len(arms)) if i not in (first, second)),
                      key=lambda i: (float(np.dot(arms[i]['direction'], axis)), i))
        groups = {f'{node}_a': (center-axis*gap/2, [first, *rest[:len(rest)//2]]),
                  f'{node}_b': (center+axis*gap/2, [second, *rest[len(rest)//2:]])}
        for name, (position, _) in groups.items():
            graph.add_node(name, xy=position.tolist())
        for name, (position, members) in groups.items():
            for index in members:
                arm = arms[index]
                graph.remove_edge(node, arm['other'], arm['key'])
                kept = arm['line'][np.linalg.norm(arm['line']-center, axis=1) > gap]
                outward = [position.tolist(), *(kept.tolist() if len(kept) else [arm['line'][-1].tolist()])]
                edge = arm['edge']
                identifier = f"{edge['edge_id']}s"
                end = arm['other']
                line, u, v = (outward, name, end) if edge['u'] == node else (outward[::-1], end, name)
                graph.add_edge(u, v, key=identifier, **{**edge, 'edge_id': identifier, 'u': u, 'v': v, 'polyline_xy': line,
                                                        'length_m': length_px(line)*resolution})
        (a_name, (a_position, _)), (b_name, (b_position, _)) = groups.items()
        bridge = [a_position.tolist(), b_position.tolist()]
        identifier = f'{node}_link'
        graph.add_edge(a_name, b_name, key=identifier, edge_id=identifier, u=a_name, v=b_name, polyline_xy=bridge, length_m=length_px(bridge)*resolution,
                       confidence=1., source='samroad', semantic_fraction=0.)
        graph.remove_node(node)
        count += 1
    if count:
        recompress(graph, resolution)
    return count


def _near_border(xy: list[float], margin_px: float, image_size: tuple[float, float]) -> bool:
    return min(xy[0], xy[1], image_size[0]-xy[0], image_size[1]-xy[1]) <= margin_px


def remove_dead_ends(graph: nx.MultiGraph, border_margin_m: float, image_size: tuple[float, float],
                     resolution: float) -> list[dict]:
    """逐层删除断头路（原地）：边的端点度为 1 且不贴图边时删除。贴图边的路端算通向图外，是有出口的路。"""
    dropped: list[dict] = []
    margin = border_margin_m/resolution
    while True:
        tips = {n for n, degree in graph.degree if degree == 1 and not _near_border(graph.nodes[n]['xy'], margin, image_size)}
        identifiers = {e['edge_id'] for a, b, e in graph.edges(data=True) if a in tips or b in tips}
        if not identifiers:
            break
        dropped.extend(drop_edges(graph, identifiers, 'dead_end_road'))
    if dropped:
        recompress(graph, resolution)
    return dropped


def remove_short_border_stubs(graph: nx.MultiGraph, max_length_m: float, resolution: float) -> list[dict]:
    """删除贴图边路端到路口的短边（原地）：路端节点离路口太近，这段路也放不下一个块。路口还有别的路，不会断网。"""
    identifiers = {e['edge_id'] for a, b, e in graph.edges(data=True) if a != b and e['length_m'] < max_length_m
                   and sorted((min(graph.degree(a), 3), min(graph.degree(b), 3))) == [1, 3]}
    if not identifiers:
        return []
    dropped = drop_edges(graph, identifiers, 'short_border_stub')
    recompress(graph, resolution)
    return dropped


def keep_largest_component(graph: nx.MultiGraph) -> list[dict]:
    """只保留总长最大的连通分量（原地），其余路与主路网不连通，直接去掉。"""
    components = [edge_records(graph.subgraph(nodes)) for nodes in nx.connected_components(graph)]
    if len(components) < 2:
        return []
    largest = max(components, key=lambda edges: (sum(e['length_m'] for e in edges), min(e['edge_id'] for e in edges)))
    keep = {e['edge_id'] for e in largest}
    return drop_edges(graph, {e['edge_id'] for edges in components for e in edges}-keep, 'disconnected_road')


def enforce_through_network(graph: nx.MultiGraph, *, close_node_m: float, small_cycle_m: float, border_margin_m: float,
                            image_size: tuple[float, float], resolution: float,
                            road_support_fn: Callable[[list], float] | None = None,
                            duplicate_min_support: float = 0.) -> list[dict]:
    """只保留贯通路网（原地）：每条路都要有进有出，路口之间不过近，没有自环和小环，全图只有一个连通分量。

    先删断头路，避免只因断头支路才成为路口的节点参与合并。自环和首尾相同的弱重复边常由路口合并产生，放在合并之后。
    合并、开环、删路会互相产生新情况，循环到不再变化。"""
    dropped: list[dict] = []
    for _ in range(_THROUGH_PASSES):
        changes = [*remove_dead_ends(graph, border_margin_m, image_size, resolution),
                   *remove_short_border_stubs(graph, close_node_m, resolution),
                   *merge_close_junctions(graph, close_node_m, resolution, _CLOSE_NODE_SPREAD),
                   *remove_self_loops(graph, resolution),
                   *remove_weak_duplicates(graph, road_support_fn, duplicate_min_support, resolution),
                   *remove_small_cycles(graph, small_cycle_m, resolution),
                   *keep_largest_component(graph)]
        if not changes:
            break
        dropped.extend(changes)
    return dropped


def _turns_deg(points: np.ndarray) -> np.ndarray:
    v = np.diff(points, axis=0)
    return np.degrees(np.abs(np.arctan2(v[:-1, 0]*v[1:, 1]-v[:-1, 1]*v[1:, 0], np.sum(v[:-1]*v[1:], axis=1))))


def _remove_folds(points: np.ndarray, max_arm: float) -> np.ndarray:
    """去掉折返顶点（毛刺）：曲线在这里原地掉头，平滑抹不掉。每次删转角最大的内部顶点，直到没有折返。
    删一个顶点最多让线移动较短一臂的长度，短臂超过 max_arm 的是真实形状不是毛刺，不删。"""
    while len(points) > 2:
        arms = np.linalg.norm(np.diff(points, axis=0), axis=1)
        turns = np.where(np.minimum(arms[:-1], arms[1:]) <= max_arm, _turns_deg(points), 0.)
        worst = int(np.argmax(turns))
        if turns[worst] <= _FOLD_DEG:
            break
        points = np.delete(points, worst+1, axis=0)
    return points


def _smooth_samples(points: np.ndarray, step: float, sigma: float) -> np.ndarray:
    """等间距采样点沿弧长高斯平滑。两端按点对称延拓，首尾不动，端点切向也不被拉偏。"""
    radius = min(math.ceil(3*sigma/step), len(points)-1)
    kernel = np.exp(-.5*(np.arange(-radius, radius+1)*step/sigma)**2)
    padded = np.vstack([2*points[0]-points[radius:0:-1], points, 2*points[-1]-points[-2:-radius-2:-1]])
    return np.column_stack([np.convolve(padded[:, axis], kernel/kernel.sum(), mode='valid') for axis in (0, 1)])


def _gaussian_pass(line: LineString, sigma_px: float) -> np.ndarray:
    """等弧长重采样后沿弧长高斯平滑。"""
    sigma = min(sigma_px, line.length/4)
    count = max(3, math.ceil(line.length/(sigma/5))+1)
    points = shapely.get_coordinates(shapely.line_interpolate_point(line, np.linspace(0, line.length, count)))
    return _smooth_samples(points, line.length/(count-1), sigma)


def smooth_polyline(polyline: list, sigma_px: float, bounds: tuple[float, float] | None = None) -> list[list[float]]:
    """折线弧度化：去掉小于 sigma/6 的抖动和折返毛刺后做高斯平滑。sigma 不超过边长的 1/4，避免短边被抹平。
    急弯一遍平滑后可能还有尖点，再平滑一遍（只会更圆，不删顶点，否则 V 形会被一路拉直）。"""
    simplified = shapely.get_coordinates(LineString(polyline).simplify(sigma_px/6))
    # 自环首尾相同，删顶点可能把整圈缩没
    closed = np.array_equal(simplified[0], simplified[-1])
    smooth = simplified if closed else _remove_folds(simplified, _FOLD_ARM_SIGMAS*sigma_px)
    for _ in range(_SMOOTH_PASSES):
        smooth = _gaussian_pass(LineString(smooth), sigma_px)
        if _turns_deg(smooth).max() <= _CUSP_DEG:
            break
    smooth[0], smooth[-1] = polyline[0], polyline[-1]
    if bounds is not None:
        smooth = np.clip(smooth, 0, bounds)
    return smooth.tolist()


def max_turn_deg(polyline: list) -> float:
    """折线内部顶点的最大转角（度）；相邻线段夹角越大折得越厉害。"""
    points = np.asarray(polyline, float)
    return float(_turns_deg(points).max()) if len(points) >= 3 else 0.


def _cleaned_edge(polyline: list, sigma_px: float) -> np.ndarray:
    """去掉小于 sigma/6 的抖动和折返毛刺，保留首尾。"""
    return _remove_folds(shapely.get_coordinates(LineString(polyline).simplify(sigma_px/6)), _FOLD_ARM_SIGMAS*sigma_px)


def _outward_direction(line: LineString, from_start: bool, reach: float) -> np.ndarray:
    """路端离开节点的单位方向。"""
    origin, target = (0., min(reach, line.length)) if from_start else (line.length, max(line.length-reach, 0.))
    vector = shapely.get_coordinates(line.interpolate(target))[0]-shapely.get_coordinates(line.interpolate(origin))[0]
    return vector/max(float(np.linalg.norm(vector)), 1e-12)


def _pair_through_ends(lines: dict[str, LineString], ends: dict[str, list[tuple[str, int]]], reach: float) -> dict:
    """每个节点上把最顺的两个路端配成一对（同一条路穿过路口）；偏折超过上限的路端不配对，按支路结束于节点。"""
    links: dict[tuple[str, int], tuple[str, int]] = {}
    limit = math.cos(math.radians(_THROUGH_MAX_DEFLECTION_DEG))
    for items in ends.values():
        directions = {end: _outward_direction(lines[end[0]], end[1] == 0, reach) for end in items}
        pairs = sorted(((float(np.dot(directions[a], -directions[b])), a, b)
                        for i, a in enumerate(items) for b in items[i+1:]), key=lambda p: (-p[0], p[1], p[2]))
        for straightness, a, b in pairs:
            if straightness >= limit and a not in links and b not in links and a[0] != b[0]:
                links[a], links[b] = b, a
    return links


def _build_strokes(identifiers: list[str], links: dict) -> list[list[tuple[str, bool]]]:
    """把配对的边连成整条路线：[(edge_id, 是否沿 u→v), ...]。先从有自由端的边走，剩下的是闭合路线。"""
    used: set[str] = set()
    strokes = []
    def walk(start: str, forward: bool) -> None:
        stroke, edge, along = [], start, forward
        while edge not in used:
            used.add(edge)
            stroke.append((edge, along))
            following = links.get((edge, 1 if along else 0))
            if following is None:
                break
            edge, along = following[0], following[1] == 0
        strokes.append(stroke)
    for edge in identifiers:
        if edge not in used and (edge, 0) not in links:
            walk(edge, True)
        elif edge not in used and (edge, 1) not in links:
            walk(edge, False)
    for edge in identifiers:
        if edge not in used:
            walk(edge, True)
    return strokes


def _find_corners(points: np.ndarray, step: float, window_px: float) -> list[int]:
    """等间距采样点上的真实拐角下标：前后各 window_px 的走向变化超过 _CORNER_DEG，连续命中的取转得最厉害的点。"""
    span = max(1, round(window_px/step))
    if len(points) <= 2*span:
        return []
    before, after = points[span:-span]-points[:-2*span], points[2*span:]-points[span:-span]
    angles = np.degrees(np.abs(np.arctan2(before[:, 0]*after[:, 1]-before[:, 1]*after[:, 0], np.sum(before*after, axis=1))))
    hits = np.flatnonzero(angles >= _CORNER_DEG)
    if not len(hits):
        return []
    return [int(group[np.argmax(angles[group])])+span for group in np.split(hits, np.flatnonzero(np.diff(hits) > 1)+1)]


def _smooth_between_corners(points: np.ndarray, corners: list[int], step: float, sigma: float) -> np.ndarray:
    """拐角之间的路段各自平滑（首尾不动），拐角保持原样。"""
    result = points.copy()
    for start, end in zip([0, *corners], [*corners, len(points)-1]):
        segment = points[start:end+1]
        if len(segment) < 3:
            continue
        local = min(sigma, (end-start)*step/4)
        for _ in range(_SMOOTH_PASSES):
            segment = _smooth_samples(segment, step, local)
            if _turns_deg(segment).max() <= _CUSP_DEG:
                break
        result[start:end+1] = segment
    return result


def _smooth_stroke(stroke: list[tuple[str, bool]], lines: dict[str, LineString], sigma_px: float,
                   corner_window_px: float) -> tuple[dict[str, np.ndarray], int]:
    """整条路线平滑后拆回各边（点序为 u→v），并返回保留的拐角数。真实拐角保持原样；路线穿过的节点保持原位，
    偏移沿弧长余弦衰减，所以路段内切向连续。"""
    total = sum(lines[edge].length for edge, _ in stroke)
    sigma = min(sigma_px, total/4)
    target = sigma/5
    pieces, cuts = [], [0]
    for edge, forward in stroke:
        line = lines[edge]
        count = max(1, math.ceil(line.length/target))
        arcs = np.linspace(0, line.length, count+1) if forward else np.linspace(line.length, 0, count+1)
        pieces.append(shapely.get_coordinates(shapely.line_interpolate_point(line, arcs))[0 if not pieces else 1:])
        cuts.append(cuts[-1]+count)
    points = np.vstack(pieces)
    step = total/max(len(points)-1, 1)
    pins = points[cuts[1:-1]].copy()
    corners = _find_corners(points, step, corner_window_px)
    smooth = _smooth_between_corners(points, corners, step, sigma)
    reach = math.ceil(3*sigma/step)
    fixed = sorted({*cuts, *corners})
    for position, index in enumerate(cuts[1:-1]):
        slot = fixed.index(index)
        radius = min(reach, (index-fixed[slot-1])//2, (fixed[slot+1]-index)//2)
        offset = pins[position]-smooth[index]
        if radius < 1:
            smooth[index] = pins[position]
            continue
        window = .5*(1+np.cos(np.pi*np.arange(-radius, radius+1)/(radius+1)))
        smooth[index-radius:index+radius+1] += window[:, None]*offset
    return ({edge: smooth[cuts[i]:cuts[i+1]+1] if forward else smooth[cuts[i]:cuts[i+1]+1][::-1]
             for i, (edge, forward) in enumerate(stroke)}, len(corners))


def smooth_roads(graph: nx.MultiGraph, sigma_m: float, resolution: float, bounds: tuple[float, float]) -> dict:
    """所有边弧度化（原地）。同一条路穿过路口时连成整条路线一起平滑，切向在节点处连续；节点不动。
    真实拐角（短距离内走向急转）保持原样，真正的支路（T 形路口的支线、偏折过大的接续）在节点处转折。
    返回平滑前后的最大转角和偏移。"""
    stats = {'edges': 0, 'through_links': 0, 'strokes': 0, 'corners': 0, 'max_turn_before_deg': 0., 'max_turn_after_deg': 0., 'max_offset_m': 0.}
    if sigma_m <= 0:
        return stats
    sigma = sigma_m/resolution
    records = {e['edge_id']: e for *_, e in graph.edges(data=True)}
    lines, ends, results = {}, {}, {}
    for identifier, edge in records.items():
        if edge['u'] == edge['v']:
            results[identifier] = np.asarray(smooth_polyline(edge['polyline_xy'], sigma, bounds))
            continue
        lines[identifier] = LineString(_cleaned_edge(edge['polyline_xy'], sigma))
        for side, node in enumerate((edge['u'], edge['v'])):
            ends.setdefault(node, []).append((identifier, side))
    links = _pair_through_ends(lines, ends, _TANGENT_REACH_SIGMAS*sigma)
    strokes = _build_strokes(sorted(lines), links)
    for stroke in strokes:
        parts, corners = _smooth_stroke(stroke, lines, sigma, _CORNER_WINDOW_M/resolution)
        results.update(parts)
        stats['corners'] += corners
    for identifier, edge in records.items():
        before = edge['polyline_xy']
        points = np.clip(results[identifier], 0, bounds)
        points[0], points[-1] = before[0], before[-1]
        after = points.tolist()
        stats['edges'] += 1
        stats['max_turn_before_deg'] = max(stats['max_turn_before_deg'], max_turn_deg(before))
        stats['max_turn_after_deg'] = max(stats['max_turn_after_deg'], max_turn_deg(after))
        stats['max_offset_m'] = max(stats['max_offset_m'],
                                    LineString(after).hausdorff_distance(LineString(before))*resolution)
        edge['polyline_xy'], edge['length_m'] = after, length_px(after)*resolution
    stats['through_links'], stats['strokes'] = len(links)//2, len(strokes)
    return {k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()}
