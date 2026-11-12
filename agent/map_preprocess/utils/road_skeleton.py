"""用模型道路概率的骨架补 MaGRoad 路网漏掉的路：主路网不变，只增加路网里没有、概率强、两端能接入路网的路段。

路面掩码以模型概率为准，语义 road 类只填概率旁边的断口（树荫下），语义图不单独决定哪里有路。
"""
from __future__ import annotations

from collections import deque

import cv2
import networkx as nx
import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from .global_road_graph import edge_records, length_px, recompress
from .road_refine import _node_on_edge

_GRID_M = 1.
_PROB = .3                  # 概率掩码阈值
_GAP_FILL_M = 12.           # 语义 road 类只填概率带这个距离内的断口
_CLOSE_M = 5.               # 闭运算连通小断口
_HOLE_M2 = 500.             # 小于该面积的孔（分隔带、路口内小块）填掉
_COMPONENT_M2 = 400.        # 小于该面积的连通块去掉
_SPUR_M = 30.               # 骨架短刺
_COVER_M = 18.              # 离现有路网超过该距离才算漏掉的路（与并行去重距离一致，不把双线补回来）
_REACH_M = 25.              # 补上的路两端离路网不超过该距离才接入，否则仍是断头路，不补
_NODE_SNAP_M = 8.           # 接入点离已有路口小于该距离时直接用路口
_MIN_SUPPORT = .7           # 沿线概率达到 _PROB 的长度比例
_MAX_BUILDING = .3          # 沿线语义为建筑的比例上限，防止绕屋顶的假路
_BUILDING = 1
_ROAD = 0
_PASSES = 4                 # 新接上的路能让别的候选也接上


def _thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen 细化。"""
    image = mask.astype(np.uint8).copy()
    def removable(current: np.ndarray, first: bool) -> np.ndarray:
        p = np.pad(current, 1)
        p2, p3, p4, p5, p6, p7, p8, p9 = (p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2])
        count = p2+p3+p4+p5+p6+p7+p8+p9
        ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
        transitions = sum(((ring[i] == 0) & (ring[i+1] == 1)).astype(np.uint8) for i in range(8))
        first_rule = (p2*p4*p6 == 0) if first else (p2*p4*p8 == 0)
        second_rule = (p4*p6*p8 == 0) if first else (p2*p6*p8 == 0)
        return (current == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & first_rule & second_rule
    while True:
        changed = False
        for first in (True, False):
            mark = removable(image, first)
            if mark.any():
                image[mark], changed = 0, True
        if not changed:
            return image > 0


def _road_mask(probability: np.ndarray, road: np.ndarray) -> np.ndarray:
    strong = probability >= _PROB
    kernel = lambda size: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*int(size)+1,)*2)
    near = cv2.dilate(strong.astype(np.uint8), kernel(_GAP_FILL_M)) > 0
    mask = cv2.morphologyEx((strong | (road & near)).astype(np.uint8), cv2.MORPH_CLOSE, kernel(_CLOSE_M))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1-mask, connectivity=4)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < _HOLE_M2/_GRID_M**2 and stats[index, cv2.CC_STAT_LEFT] > 0 and stats[index, cv2.CC_STAT_TOP] > 0:
            mask[labels == index] = 1
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < _COMPONENT_M2/_GRID_M**2:
            mask[labels == index] = 0
    return mask > 0


def _farthest(points: set, source: tuple) -> tuple[tuple, dict]:
    parent, queue, last = {source: None}, deque([source]), source
    while queue:
        last = queue.popleft()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor = (last[0]+dx, last[1]+dy)
                if neighbor in points and neighbor not in parent:
                    parent[neighbor] = last
                    queue.append(neighbor)
    return last, parent


def skeleton_lines(mask: np.ndarray) -> list[list[list[float]]]:
    """掩码骨架上路口之间的折线（网格坐标），已去掉短刺。"""
    skeleton = _thin(mask).astype(np.uint8)
    p = np.pad(skeleton, 1)
    ring = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
    ring.append(ring[0])
    crossings = sum(((ring[i] == 0) & (ring[i+1] == 1)).astype(np.uint8) for i in range(8))  # 交叉数：楼梯台阶不算路口
    junction = (skeleton > 0) & (crossings >= 3)
    wide = cv2.dilate(junction.astype(np.uint8), np.ones((3, 3), np.uint8)) & skeleton  # 路口连同紧邻骨架挖掉，各臂才断开
    _, junction_labels = cv2.connectedComponents(wide, connectivity=8)
    centers = {}
    for label in range(1, int(junction_labels.max())+1):
        rows, columns = np.where((junction_labels == label) & junction)
        if not len(columns):
            rows, columns = np.where(junction_labels == label)
        centers[label] = [float(columns.mean())+.5, float(rows.mean())+.5]
    count, labels = cv2.connectedComponents(((skeleton > 0) & (wide == 0)).astype(np.uint8), connectivity=8)
    def touching(point: tuple) -> int | None:
        window = junction_labels[max(point[1]-2, 0):point[1]+3, max(point[0]-2, 0):point[0]+3]
        window = window[window > 0]
        return int(np.bincount(window).argmax()) if window.size else None
    pieces = []
    for index in range(1, count):
        rows, columns = np.where(labels == index)
        points = {(int(x), int(y)) for x, y in zip(columns, rows)}
        first, _ = _farthest(points, next(iter(points)))
        last, parent = _farthest(points, first)
        path = [last]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        start, end = touching(path[0]), touching(path[-1])
        if start is not None and start == end and len(path)*_GRID_M < _SPUR_M:
            continue
        pieces.append((start, end, [[x+.5, y+.5] for x, y in path]))
    while True:  # 去短刺：一端悬空且很短
        degree: dict[int, int] = {}
        for start, end, _ in pieces:
            for node in (start, end):
                if node is not None:
                    degree[node] = degree.get(node, 0)+1
        kept = [p for p in pieces if not (len(p[2])*_GRID_M < _SPUR_M and (p[0] is None) != (p[1] is None))]
        if len(kept) == len(pieces):
            break
        pieces = kept
    return [[*([centers[start]] if start is not None else []), *line, *([centers[end]] if end is not None else [])]
            for start, end, line in pieces if len(line) >= 2]


def _attach(graph: nx.MultiGraph, point: np.ndarray, resolution: float) -> str:
    """路网上离 point 最近的接入节点：靠近路口直接用路口，否则在边上打断。"""
    position = Point(point)
    nearest = min(edge_records(graph), key=lambda e: (LineString(e['polyline_xy']).distance(position), e['edge_id']))
    line = LineString(nearest['polyline_xy'])
    return _node_on_edge(graph, nearest['edge_id'], line.project(position), _NODE_SNAP_M/resolution, resolution)


def fill_from_skeleton(graph: nx.MultiGraph, probability: np.ndarray, labels: np.ndarray, image_size: tuple[float, float],
                       resolution: float, min_length_m: float) -> dict:
    """把骨架上路网漏掉的路段接入路网（原地）：离路网超过覆盖距离、概率强、不压建筑，且两端都能接入路网。"""
    result = {'added': 0, 'added_m': 0.}
    if min_length_m <= 0 or not graph.number_of_edges():
        return result
    width, height = max(1, round(image_size[0]*resolution/_GRID_M)), max(1, round(image_size[1]*resolution/_GRID_M))
    grid_probability = cv2.resize(probability.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    grid_labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)
    lines = skeleton_lines(_road_mask(grid_probability, grid_labels == _ROAD))
    scale = np.array([image_size[0]/width, image_size[1]/height])
    cover, reach = _COVER_M/resolution, _REACH_M/resolution
    def evidence(points: np.ndarray) -> bool:
        column = np.clip((points[:, 0]/scale[0]).astype(int), 0, width-1)
        row = np.clip((points[:, 1]/scale[1]).astype(int), 0, height-1)
        return float((grid_probability[row, column] >= _PROB).mean()) >= _MIN_SUPPORT and float((grid_labels[row, column] == _BUILDING).mean()) <= _MAX_BUILDING
    for _ in range(_PASSES):
        accepted = 0
        for line in sorted(lines, key=len, reverse=True):
            points = np.asarray(line, float)*scale
            network = unary_union([LineString(e['polyline_xy']) for e in edge_records(graph)])
            far = shapely.distance(shapely.points(points), network) > cover
            if not far.any():
                continue
            runs = np.flatnonzero(np.diff(np.r_[False, far, False]))
            for begin, end in zip(runs[::2], runs[1::2]):
                run = points[begin:end]
                if len(run) < 2 or LineString(run).length*resolution < min_length_m or not evidence(run):
                    continue
                ends = [Point(run[0]), Point(run[-1])]
                network = unary_union([LineString(e['polyline_xy']) for e in edge_records(graph)])
                if max(network.distance(point) for point in ends) > reach:
                    continue
                first = _attach(graph, run[0], resolution)
                last = _attach(graph, run[-1], resolution)
                if first == last:
                    continue
                polyline = [list(graph.nodes[first]['xy']), *LineString(run).simplify(.8/resolution).coords, list(graph.nodes[last]['xy'])]
                polyline = [list(map(float, xy)) for xy in polyline]
                identifier = f"fill{result['added']}"
                column = np.clip((run[:, 0]/scale[0]).astype(int), 0, width-1)
                row = np.clip((run[:, 1]/scale[1]).astype(int), 0, height-1)
                graph.add_edge(first, last, key=identifier, edge_id=identifier, u=first, v=last, polyline_xy=polyline,
                               length_m=length_px(polyline)*resolution, source='samroad', semantic_fraction=0.,
                               confidence=round(min(1., float(grid_probability[row, column].mean())/.5), 4))
                result['added'] += 1
                result['added_m'] += length_px(polyline)*resolution
                accepted += 1
        if not accepted:
            break
    if result['added']:
        recompress(graph, resolution)
    result['added_m'] = round(result['added_m'], 1)
    return result
