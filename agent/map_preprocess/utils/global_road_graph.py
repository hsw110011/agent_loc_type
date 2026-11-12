"""全局 MultiGraph 清洗工具。模型连接是唯一拓扑依据，几何交叉不建节点。"""
from __future__ import annotations

import math
from typing import Any
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point

_ROUTE_FOLD_DEG = 120.  # 路端接簇内路线时转角超过该值视为往回折


def length_px(polyline: list) -> float:
    return sum(math.dist(a, b) for a, b in zip(polyline, polyline[1:]))


def raw_to_graph(raw: dict) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    for vertex in raw['vertices']:
        identifier, xy = str(vertex['node_id']), vertex['xy']
        if identifier in graph or len(xy) != 2 or not np.isfinite(xy).all():
            raise ValueError('原始图节点 ID 重复或坐标非法')
        graph.add_node(identifier, xy=list(map(float, xy)))
    for edge in raw['edges']:
        a, b = str(edge['u']), str(edge['v'])
        if a not in graph or b not in graph:
            raise ValueError('原始边引用不存在节点')
        line = edge.get('polyline_xy') or [graph.nodes[a]['xy'], graph.nodes[b]['xy']]
        values = np.asarray(line, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError('非法形状折线')
        if not np.allclose(values[0], graph.nodes[a]['xy']) or not np.allclose(values[-1], graph.nodes[b]['xy']):
            raise ValueError('折线首尾必须对应 u/v 节点')
        confidence = edge.get('confidence')
        if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (float, int))
                                       or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError('confidence 必须在 [0,1] 或 null')
        if length_px(line) <= 0:
            raise ValueError('边长度必须为正')
        source = edge.get('source', 'samroad')
        if source not in {'samroad', 'semantic'}:
            raise ValueError('原始边 source 必须为 samroad 或 semantic')
        graph.add_edge(a, b, u=a, v=b, polyline_xy=values.tolist(), confidence=confidence, source=source)
    return graph


def compress_degree_two(original: nx.MultiGraph, resolution: float) -> nx.MultiGraph:
    anchors = {n for n, d in original.degree if d != 2 or len(set(original.neighbors(n))-{n}) < 2}
    for component in nx.connected_components(original):
        if not anchors.intersection(component):
            anchors.add(min(component, key=lambda n: (original.nodes[n]['xy'], n)))
    graph = nx.MultiGraph()
    for node in sorted(anchors):
        graph.add_node(node, **original.nodes[node])
    visited, records = set(), []
    def token(a, b, k):
        return (tuple(sorted((a, b))), k)
    def oriented(edge, start):
        return edge['polyline_xy'] if edge['u'] == start else list(reversed(edge['polyline_xy']))
    for start in sorted(anchors):
        for _, neighbor, key, data in sorted(original.edges(start, keys=True, data=True), key=lambda e: (e[1], e[2])):
            if token(start, neighbor, key) in visited:
                continue
            line, current, previous = [], neighbor, start
            confidences, weights, semantic_px = [], [], 0.
            while True:
                visited.add(token(previous, current, key))
                part = oriented(data, previous)
                line.extend(part if not line else part[1:])
                if data['confidence'] is not None:
                    confidences.append(data['confidence'])
                    weights.append(length_px(part))
                # 已压缩过的边（再次压缩时）带 semantic_fraction；原始边按 source 计
                fraction = data.get('semantic_fraction', 1. if data.get('source', 'samroad') == 'semantic' else 0.)
                semantic_px += length_px(part)*fraction
                if current in anchors:
                    break
                following = next((e for e in original.edges(current, keys=True, data=True)
                                  if token(e[0], e[1], e[2]) not in visited), None)
                if following is None:
                    raise ValueError('degree-2 链无法闭合')
                previous, current, key, data = following
            # 假设边方向取完整折线字典序较小者，稳定且不表示交通方向。
            forward, backward = tuple(map(tuple, line)), tuple(map(tuple, reversed(line)))
            u, v = start, current
            if backward < forward:
                line, u, v = list(reversed(line)), v, u
            # 来源：samroad（SAM-Road 提取）/ semantic（语义补全）/ mixed（压缩时两者首尾相连）
            fraction = semantic_px/max(length_px(line), 1e-12)
            source = 'semantic' if fraction >= .99 else 'samroad' if fraction <= .01 else 'mixed'
            records.append({'u': u, 'v': v, 'polyline_xy': line, 'length_m': length_px(line)*resolution,
                            'confidence': float(np.average(confidences, weights=weights)) if weights else None,
                            'source': source, 'semantic_fraction': round(float(fraction), 4)})
    for index, edge in enumerate(sorted(records, key=lambda e: (e['polyline_xy'], e['u'], e['v'],
                                                               -1 if e['confidence'] is None else e['confidence']))):
        edge['edge_id'] = str(index)
        graph.add_edge(edge['u'], edge['v'], key=edge['edge_id'], **edge)
    return graph


def recompress(graph: nx.MultiGraph, resolution: float) -> None:
    """拓扑变动后重新 degree-2 压缩并重新编号（原地）。"""
    compressed = compress_degree_two(graph, resolution)
    graph.clear()
    graph.add_nodes_from(compressed.nodes(data=True))
    graph.add_edges_from(compressed.edges(keys=True, data=True))


def _attach_route(line: list, route: list) -> list:
    """line 末端在合并簇成员上，接上到落点的路线。路线若往回折（这条路本来就从落点旁边经过），
    就在路上离落点最近处直接接过去，不走回头路。"""
    arrive, leave = np.subtract(line[-1], line[-2]), np.subtract(route[1], route[0]) if len(route) > 1 else None
    if leave is None:
        return line
    turn = abs(math.degrees(math.atan2(arrive[0]*leave[1]-arrive[1]*leave[0], float(np.dot(arrive, leave)))))
    if turn <= _ROUTE_FOLD_DEG:
        return line[:-1]+route
    shape, target = LineString(line), route[-1]
    along = shape.project(Point(target))
    arcs = np.r_[0., np.cumsum(np.linalg.norm(np.diff(np.asarray(line, float), axis=0), axis=1))]
    head = [list(p) for p, arc in zip(line, arcs) if arc < along] or [list(line[0])]
    foot = [shape.interpolate(along).x, shape.interpolate(along).y]
    trimmed = [*head, *([foot] if along > 0 else []), list(target)]
    # 两端都在同一簇的边（合并后成自环）投影可能退化成一个点，此时保留原样，交给小环规则
    return trimmed if length_px(trimmed) > 0 else line[:-1]+route


def _junction_representative(graph: nx.MultiGraph, members: list[str]) -> str:
    def rank(node: str) -> tuple:
        model_edges = sum(e.get('source', 'samroad') != 'semantic' for *_, e in graph.edges(node, data=True))
        return graph.degree(node), model_edges, node
    return max(members, key=rank)


def _cluster_routes(graph: nx.MultiGraph, members: list[str], short_keys: set) -> tuple[str, dict[str, list]]:
    """合并簇的落点和各成员到落点的真实路线。

    簇内只走置信度最高的生成树（低置信的假连接不走），落点取树的加权中位点：路端沿路走的总距离最短，
    同样近时取度最高的路口。返回 (落点, {成员: 从成员到落点的折线})。"""
    member_set = set(members)
    links = nx.Graph()
    external = {node: 0 for node in members}
    for a, b, key, edge in graph.edges(members, keys=True, data=True):
        if (a, b, key) not in short_keys and (b, a, key) not in short_keys:
            for node in (a, b):
                if node in member_set:
                    external[node] += 1
            continue
        if a == b:
            continue
        confidence = 1. if edge['confidence'] is None else edge['confidence']
        if not links.has_edge(a, b) or (confidence, -edge['length_m']) > (links[a][b]['confidence'], -links[a][b]['length_m']):
            links.add_edge(a, b, confidence=confidence, length_m=edge['length_m'], edge=edge)
    tree = nx.maximum_spanning_tree(links, weight='confidence')
    distances = dict(nx.all_pairs_dijkstra_path_length(tree, weight='length_m'))
    cost = {root: sum(external[node]*distances[root][node] for node in members) for root in members}
    best = min(cost.values())
    root = _junction_representative(graph, [node for node in members if cost[node] <= best+1e-9])
    routes = {}
    for node in members:
        line: list = [list(graph.nodes[node]['xy'])]
        path = nx.shortest_path(tree, node, root, weight='length_m')
        for a, b in zip(path, path[1:]):
            edge = tree[a][b]['edge']
            part = edge['polyline_xy'] if edge['u'] == a else list(reversed(edge['polyline_xy']))
            line.extend(list(p) for p in part[1:])
        routes[node] = line
    return root, routes


def merge_close_junctions(graph: nx.MultiGraph, distance_m: float, resolution: float, max_spread: float = 1.) -> list[dict]:
    """合并过近的路口（原地修改）：两端都是路口（度 >= 3）且短于 distance_m 的边，两端路口合并为一个节点，
    该短边删除，其余边沿簇内真实路线走到新节点（不拉动路端，拐角按原样保留）；反复直到没有过近路口，最后重新 degree-2 压缩。
    合并后的节点沿用簇内一个原始路口的坐标，不取均值：原始路口都在路上，均值可能落到路外。max_spread 是合并组直径上限相对 distance_m 的倍数。返回被删除的短边（reason=close_junction_merge）。"""
    dropped: list[dict] = []
    if distance_m <= 0:
        return dropped
    for _ in range(20):
        short = [(a, b, k, d) for a, b, k, d in graph.edges(keys=True, data=True)
                 if a != b and d['length_m'] < distance_m and graph.degree(a) >= 3 and graph.degree(b) >= 3]
        if not short:
            break
        # 由短到长合并；合并后组内任意两路口相距须 <= distance_m（不把一串相邻路口压成一个点）
        group = {n: {n} for n in graph.nodes}
        limit_px = max_spread*distance_m/resolution
        merged_keys = set()
        for a, b, k, data in sorted(short, key=lambda e: (e[3]['length_m'], e[3]['edge_id'])):
            if group[a] is group[b]:
                merged_keys.add((a, b, k))
                continue
            union = group[a] | group[b]
            # 直径按原始路口计算（已合并节点记录其来源坐标），多轮合并也不会累积漂移
            points = np.asarray([xy for n in union for xy in graph.nodes[n].get('merged_from_xy', [graph.nodes[n]['xy']])], float)
            spread = np.max(np.linalg.norm(points[:, None]-points[None], axis=2))
            if spread > limit_px:
                continue
            for n in union:
                group[n] = union
            merged_keys.add((a, b, k))
        clusters: dict[str, list[str]] = {}
        for members in {id(g): g for g in group.values()}.values():
            if len(members) > 1:
                clusters[min(members)] = sorted(members)
        if not clusters:
            break
        short_keys = merged_keys
        for root, members in clusters.items():
            member_set = set(members)
            landing, routes = _cluster_routes(graph, members, short_keys)
            center = list(graph.nodes[landing]['xy'])
            identifier = f'j{root}'
            graph.add_node(identifier, xy=center, merged_from_xy=[xy for n in members
                           for xy in graph.nodes[n].get('merged_from_xy', [graph.nodes[n]['xy']])])
            for a, b, k, data in list(graph.edges(members, keys=True, data=True)):
                if (a, b, k) in short_keys or (b, a, k) in short_keys:
                    dropped.append({**data, 'reason': 'close_junction_merge'})
                    continue
                inside_a, inside_b = a in member_set, b in member_set
                line = [list(p) for p in data['polyline_xy']]
                start, end = data['u'], data['v']
                original_px = length_px(line)
                if start in member_set:
                    line = _attach_route(line[::-1], routes[start])[::-1]
                    start = identifier
                if end in member_set:
                    line = _attach_route(line, routes[end])
                    end = identifier
                if inside_a and inside_b and original_px < distance_m/resolution:
                    dropped.append({**data, 'reason': 'close_junction_merge'})
                    continue
                graph.add_edge(start, end, **{**data, 'u': start, 'v': end, 'polyline_xy': line,
                                              'length_m': length_px(line)*resolution})
            graph.remove_nodes_from(members)
    if not dropped:
        return dropped
    # 合并后部分节点度数变为 2：重新压缩并重新编号
    recompress(graph, resolution)
    return dropped


def edge_records(graph: nx.MultiGraph) -> list[dict]:
    return sorted((dict(data) for *_, data in graph.edges(data=True)), key=lambda e: e['edge_id'])


def graph_stats(graph: nx.MultiGraph) -> dict:
    return {'edges': graph.number_of_edges(), 'total_length_m': sum(e['length_m'] for e in edge_records(graph))}


def drop_edges(graph: nx.MultiGraph, identifiers: set[str], reason: str) -> list[dict]:
    dropped = []
    for a, b, key, edge in list(graph.edges(keys=True, data=True)):
        if edge['edge_id'] in identifiers:
            dropped.append({**edge, 'reason': reason})
            graph.remove_edge(a, b, key)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return dropped


def remove_small_components(graph: nx.MultiGraph, minimum: float) -> list[dict]:
    ids = set()
    for nodes in nx.connected_components(graph):
        edges = edge_records(graph.subgraph(nodes))
        if sum(e['length_m'] for e in edges) < minimum:
            ids.update(e['edge_id'] for e in edges)
    return drop_edges(graph, ids, 'short_component')


def prune_short_spurs(graph: nx.MultiGraph, minimum: float) -> list[dict]:
    dropped = []
    while True:
        ids = {e['edge_id'] for a, b, e in graph.edges(data=True)
               if a != b and e['length_m'] < minimum and (graph.degree(a) == 1 or graph.degree(b) == 1)}
        if not ids:
            return dropped
        dropped.extend(drop_edges(graph, ids, 'short_spur'))


def score_edge(edge: dict, graph: nx.MultiGraph, config) -> float:
    """长度饱和归一化、缺失置信度按 1、弯曲度倒数、两端 degree/4 饱和。"""
    line = edge['polyline_xy']
    values = {'length': min(1., edge['length_m']/config.score_length_scale_m),
              'confidence': 1. if edge['confidence'] is None else edge['confidence'],
              'straightness': min(1., math.dist(line[0], line[-1])/max(length_px(line), 1e-12)),
              'connectivity': min(1., (graph.degree(edge['u'])+graph.degree(edge['v']))/8)}
    return sum(values[k]*w for k, w in config.edge_score_weights.items())/sum(config.edge_score_weights.values())


def main_trunk_edges(graph: nx.MultiGraph, attached_minimum: float) -> set[str]:
    if not graph.number_of_edges():
        return set()
    # 假设“主干”为最长总长度分量的最大生成树之加权直径；环边和单跳连接边按阈值附加。
    component = max(nx.connected_components(graph), key=lambda ns: (
        sum(e['length_m'] for e in edge_records(graph.subgraph(ns))), tuple(sorted(ns))))
    tree = nx.maximum_spanning_tree(graph.subgraph(component), weight='length_m')
    def farthest(start):
        distances, paths = nx.single_source_dijkstra(tree, start, weight='length_m')
        end = max(distances, key=lambda n: (distances[n], n))
        return end, paths[end]
    end, _ = farthest(min(component))
    _, path = farthest(end)
    selected = set()
    for a, b in zip(path, path[1:]):
        edge = max(tree[a][b].values(), key=lambda e: e['length_m'])
        selected.add(edge['edge_id'])
    for edge in edge_records(graph):
        if edge['u'] in path or edge['v'] in path:
            if edge['length_m'] >= attached_minimum:
                selected.add(edge['edge_id'])
    return selected


def serialize_graph(graph: nx.MultiGraph) -> dict[str, Any]:
    return {'nodes': [{'node_id': n, **data} for n, data in sorted(graph.nodes(data=True))],
            'edges': edge_records(graph)}
