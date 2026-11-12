"""语义路线：拓扑定稿后，把路线收到语义道路（road 类）的中线上，连接关系不变。

每条边在原线两侧的走廊内求最小代价路径：道路类区域越靠中线代价越低；没有道路类的地方（树荫等）代价高，
并被拉回原线，所以不会漂移；建筑代价最高。路口节点先吸附到道路类中心，路端节点不动（贴图边的路端要保持贴边）。
结果只去掉栅格台阶，曲线的连续顺滑由后面的 smooth_roads 完成。
"""
from __future__ import annotations

import heapq
import math

import cv2
import networkx as nx
import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from .global_road_graph import drop_edges, length_px, recompress
from .road_refine import _reattach, _weakest

ROAD, BUILDING = 0, 1
_RING_COMPACT = .4          # 环的紧凑度（4πA/P²）低于该值算细长（双向分隔的两条线），只去一条边，否则收成一个路口
_WIDE_JUNCTION_MAX_M = 40.  # 宽路面上两个路口相距小于该值、周围几乎全是路面时算同一个路口
_WIDE_JUNCTION_RADIUS_M = 25.
_WIDE_JUNCTION_MIN_ROAD = .8
_PROB_STRONG = .3           # 模型道路概率达到该值算有概率证据，路线沿概率脊线走
_PROB_WEIGHT = 20.          # 离当地概率脊线越远代价越高，按当地最大概率归一
_SEMANTIC_BASE = 3.         # 没有概率证据的路段由语义补，基础代价高于概率脊线，路线优先走概率
_GRID_M = .5                # 寻路网格
_CLOSE_KERNEL_PX = 5        # 闭运算连通车辆、细小阴影造成的空洞（网格像元）
_CENTER_WEIGHT = 20.        # 道路类内离当地路中线越远代价越高，按当地半宽归一，宽路窄路中线的吸引力一样
_LOCAL_WIDTH_WINDOW_M = 30. # 当地半宽：该窗口内离路边的最大距离
_SNAP_MAX_M = 15.           # 宽路上路口吸附半径随半宽放大的上限
_COST_OFF_ROAD = 10.
_COST_BUILDING = 50.
_PULL_ON_ROAD = .02         # 每离原线 1 m 增加的代价：道路类内很小，只防止拐进支路
_PULL_OFF_ROAD = 2.         # 没有道路类时很大：沿原线走，不漂移
_PREFILTER_SIGMA_M = 2.5    # 去栅格台阶的高斯 sigma，真正的平滑交给 smooth_roads
_MAX_DETOUR = 1.35          # 新路线比原线长太多视为绕路，放弃
_NEIGHBORS = ((1, 0, 1.), (-1, 0, 1.), (0, 1, 1.), (0, -1, 1.),
              (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)))


def _least_cost_path(start: tuple[int, int], goal: tuple[int, int], allowed: np.ndarray, cost: np.ndarray) -> list[tuple[int, int]] | None:
    """8 邻域 Dijkstra，坐标 (x, y)。"""
    height, width = allowed.shape
    best, previous, queue = {start: 0.}, {}, [(0., start)]
    while queue:
        distance, current = heapq.heappop(queue)
        if current == goal:
            break
        if distance > best[current]:
            continue
        x, y = current
        for dx, dy, step in _NEIGHBORS:
            a, b = x+dx, y+dy
            if not (0 <= a < width and 0 <= b < height and allowed[b, a]):
                continue
            value = distance+step*(cost[y, x]+cost[b, a])/2
            if value < best.get((a, b), math.inf):
                best[(a, b)], previous[(a, b)] = value, current
                heapq.heappush(queue, (value, (a, b)))
    if goal not in previous:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    return path[::-1]


def _remove_stairs(points: np.ndarray, sigma_px: float) -> np.ndarray:
    """等距重采样后高斯平滑，首尾不动。"""
    line = LineString(points)
    count = max(4, int(line.length/(sigma_px/4))+1)
    samples = shapely.get_coordinates(shapely.line_interpolate_point(line, np.linspace(0, line.length, count)))
    step = line.length/(count-1)
    radius = min(int(3*sigma_px/step), count-1)
    if radius < 1:
        return samples
    kernel = np.exp(-.5*(np.arange(-radius, radius+1)*step/sigma_px)**2)
    padded = np.vstack([2*samples[0]-samples[radius:0:-1], samples, 2*samples[-1]-samples[-2:-radius-2:-1]])
    smooth = np.column_stack([np.convolve(padded[:, axis], kernel/kernel.sum(), mode='valid') for axis in (0, 1)])
    smooth[0], smooth[-1] = samples[0], samples[-1]
    return smooth


def _reanchor(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """形状不变，只把首尾移到新节点：端点的位移沿弧长线性分摊。"""
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    ratio = (arc/arc[-1])[:, None] if arc[-1] > 0 else np.linspace(0, 1, len(points))[:, None]
    return points+(1-ratio)*(start-points[0])+ratio*(end-points[-1])


class _Field:
    """语义标签降到寻路网格后的代价场；坐标换算在原图像素与网格之间。"""

    def __init__(self, labels: np.ndarray, image_size: tuple[float, float], resolution: float,
                 probability: np.ndarray | None = None):
        label_height, label_width = labels.shape
        label_resolution = resolution*image_size[0]/label_width
        factor = max(1, round(_GRID_M/label_resolution))
        small = labels[::factor, ::factor]
        self.cell = label_resolution*factor
        self.scale = np.array([label_width/image_size[0]/factor, label_height/image_size[1]/factor])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_CLOSE_KERNEL_PX, _CLOSE_KERNEL_PX))
        self.road = cv2.morphologyEx((small == ROAD).astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        self.building = small == BUILDING
        self.clearance = cv2.distanceTransform(self.road.astype(np.uint8), cv2.DIST_L2, 5)*self.cell
        self.cost = np.full(small.shape, _COST_OFF_ROAD, np.float32)
        self.cost[self.building] = _COST_BUILDING
        window = 2*round(_LOCAL_WIDTH_WINDOW_M/self.cell/2)+1
        self.halfwidth = cv2.dilate(self.clearance, cv2.getStructuringElement(cv2.MORPH_RECT, (window, window)))
        offcenter = 1-np.clip(self.clearance/np.maximum(self.halfwidth, 1e-3), 0, 1)
        self.cost[self.road] = (1+_CENTER_WEIGHT*offcenter**2)[self.road]
        self.prob, self.strong = None, np.zeros(small.shape, bool)
        if probability is not None:
            # 概率优先：有概率证据处走概率脊线，语义只补概率弱或断开的路段
            self.prob = cv2.resize(probability.astype(np.float32), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_LINEAR)
            local = cv2.dilate(self.prob, cv2.getStructuringElement(cv2.MORPH_RECT, (window, window)))
            self.strong = self.prob >= _PROB_STRONG
            self.cost = self.cost+_SEMANTIC_BASE
            ridge_gap = 1-np.clip(self.prob/np.maximum(local, 1e-3), 0, 1)
            self.cost[self.strong] = (1+_PROB_WEIGHT*ridge_gap**2)[self.strong]
        self.passable = self.road | self.strong

    def cell_of(self, xy) -> tuple[int, int]:
        x, y = np.asarray(xy, float)*self.scale
        return (min(max(int(x), 0), self.road.shape[1]-1), min(max(int(y), 0), self.road.shape[0]-1))

    def on_road(self, polyline: list, mask: np.ndarray | None = None) -> tuple[int, int]:
        line = LineString(polyline)
        count = max(2, math.ceil(line.length))
        points = shapely.get_coordinates(shapely.line_interpolate_point(line, (np.arange(count)+.5)*line.length/count))*self.scale
        column = np.clip(points[:, 0].astype(int), 0, self.road.shape[1]-1)
        row = np.clip(points[:, 1].astype(int), 0, self.road.shape[0]-1)
        return int((self.road if mask is None else mask)[row, column].sum()), count

    def center_near(self, xy: list[float], radius_m: float) -> list[float] | None:
        """半径内最靠道路类中心的位置（原图像素）；附近没有道路类返回 None。"""
        x, y = self.cell_of(xy)
        radius = int(radius_m/self.cell)
        left, top = max(0, x-radius), max(0, y-radius)
        window = self.clearance[top:y+radius+1, left:x+radius+1]
        probability = None if self.prob is None else self.prob[top:y+radius+1, left:x+radius+1]
        use_probability = probability is not None and probability.size and probability.max() >= _PROB_STRONG
        if window.size == 0 or (not use_probability and window.max() <= 0):
            return None
        rows, columns = np.mgrid[0:window.shape[0], 0:window.shape[1]]
        distance = np.hypot(columns+left-x, rows+top-y)*self.cell
        value = probability-.01*distance if use_probability else window-.25*distance
        score = np.where(distance <= radius_m, value, -math.inf)
        row, column = np.unravel_index(int(np.argmax(score)), score.shape)
        return ((np.array([column+left, row+top], float)+.5)/self.scale).tolist()


def _snap_junctions(graph: nx.MultiGraph, field: _Field, snap_m: float, close_px: float) -> int:
    """路口吸附到道路类中心；路端（度 1）不动；吸附后不得比原来更靠近过近阈值内的其它节点。"""
    moved = 0
    for node, degree in sorted(graph.degree, key=lambda item: str(item[0])):
        if degree < 2:
            continue
        current = graph.nodes[node]['xy']
        x, y = field.cell_of(current)
        target = field.center_near(current, min(_SNAP_MAX_M, max(snap_m, float(field.halfwidth[y, x]))))
        if target is None:
            continue
        others = [graph.nodes[n]['xy'] for n in graph.nodes if n != node]
        if any(math.dist(target, xy) < min(close_px, math.dist(current, xy)) for xy in others):
            continue
        graph.nodes[node]['xy'] = target
        moved += 1
    return moved


def _route_edge(edge: dict, start: np.ndarray, end: np.ndarray, field: _Field, corridor_m: float,
                resolution: float) -> np.ndarray | None:
    """走廊内的最小代价路径（原图像素）；找不到或绕路时返回 None。"""
    points = np.asarray(edge['polyline_xy'], float)*field.scale
    margin = corridor_m/field.cell+2
    left, top = np.floor(points.min(axis=0)-margin).astype(int).clip(0)
    right = min(int(math.ceil(points[:, 0].max()+margin)), field.road.shape[1])
    bottom = min(int(math.ceil(points[:, 1].max()+margin)), field.road.shape[0])
    origin = np.array([left, top])
    line_mask = np.zeros((bottom-top, right-left), np.uint8)
    cv2.polylines(line_mask, [(points-origin).round().astype(np.int32)], False, 255, 1)
    distance = cv2.distanceTransform(255-line_mask, cv2.DIST_L2, 5)*field.cell
    allowed = distance <= corridor_m
    begin, finish = tuple(np.array(field.cell_of(start))-origin), tuple(np.array(field.cell_of(end))-origin)
    for x, y in (begin, finish):
        if not (0 <= y < allowed.shape[0] and 0 <= x < allowed.shape[1]):
            return None
        allowed[y, x] = True
    begin, finish = (int(begin[0]), int(begin[1])), (int(finish[0]), int(finish[1]))
    if begin == finish:
        return None
    cost = field.cost[top:bottom, left:right]+np.where(field.passable[top:bottom, left:right], _PULL_ON_ROAD, _PULL_OFF_ROAD)*distance
    path = _least_cost_path(begin, finish, allowed, cost)
    if path is None or len(path) < 3:
        return None
    cells = np.array(path, float)+origin+.5
    route = _remove_stairs(cells/field.scale, _PREFILTER_SIGMA_M/resolution)
    if LineString(route).length > _MAX_DETOUR*max(length_px(edge['polyline_xy']), 1.):
        return None
    route[0], route[-1] = start, end
    return route


def route_along_semantic(graph: nx.MultiGraph, labels: np.ndarray, image_size: tuple[float, float], resolution: float,
                         corridor_m: float, node_snap_m: float, close_node_m: float,
                         probability: np.ndarray | None = None) -> dict:
    """所有边改走道路中线（原地）：有模型概率时优先沿概率脊线，概率弱或断开处用语义 road 类补。
    首尾相同的重复边各走各的原形，否则会被收到同一条中线上。"""
    field = _Field(labels, image_size, resolution, probability)
    before = [field.on_road(e['polyline_xy']) for *_, e in graph.edges(data=True)]
    moved = _snap_junctions(graph, field, node_snap_m, close_node_m/resolution)
    pairs: dict[frozenset, int] = {}
    for a, b, _ in graph.edges(data=True):
        pairs[frozenset((a, b))] = pairs.get(frozenset((a, b)), 0)+1
    routed = kept = 0
    for a, b, edge in graph.edges(data=True):
        start = np.asarray(graph.nodes[edge['u']]['xy'], float)
        end = np.asarray(graph.nodes[edge['v']]['xy'], float)
        route = None if a == b or pairs[frozenset((a, b))] > 1 else _route_edge(edge, start, end, field, corridor_m, resolution)
        if route is None:
            route = _reanchor(np.asarray(edge['polyline_xy'], float), start, end)
            kept += 1
        else:
            routed += 1
        edge['polyline_xy'], edge['length_m'] = route.tolist(), length_px(route.tolist())*resolution
    after = [field.on_road(e['polyline_xy']) for *_, e in graph.edges(data=True)]
    share = lambda items: round(sum(i[0] for i in items)/max(sum(i[1] for i in items), 1), 4)
    prob_after = [field.on_road(e['polyline_xy'], field.strong) for *_, e in graph.edges(data=True)]
    return {'edges_routed': routed, 'edges_kept': kept, 'nodes_moved': moved, 'on_road_before': share(before),
            'on_road_after': share(after), 'on_probability_after': share(prob_after) if probability is not None else None}


def _candidate_rings(graph: nx.MultiGraph, max_perimeter_m: float) -> list[list[dict]]:
    """周长不超过上限的环（最小环基和首尾相同的边对），由小到大。"""
    simple, rings = nx.Graph(), []
    for a, b, edge in graph.edges(data=True):
        if a == b:
            continue
        if simple.has_edge(a, b):
            rings.append([simple[a][b]['edge'], edge])
        else:
            simple.add_edge(a, b, weight=edge['length_m'], edge=edge)
    rings.extend([data['edge'] for *_, data in simple.subgraph(nodes).edges(data=True)]
                 for nodes in nx.minimum_cycle_basis(simple, weight='weight'))
    perimeter = lambda edges: sum(e['length_m'] for e in edges)
    return sorted((r for r in rings if perimeter(r) <= max_perimeter_m), key=lambda r: (perimeter(r), sorted(e['edge_id'] for e in r)))


def _ring_polygon(edges: list[dict]) -> Polygon | None:
    polygons = list(polygonize(unary_union([LineString(e['polyline_xy']) for e in edges])))
    return max(polygons, key=lambda p: p.area) if polygons else None


def _road_fraction(polygon: Polygon, labels: np.ndarray, image_size: tuple[float, float]) -> float:
    """环内部语义为道路的比例（无数据不计）。"""
    height, width = labels.shape[:2]
    points = np.clip((np.asarray(polygon.exterior.coords)*[width/image_size[0], height/image_size[1]]).astype(np.int32),
                     0, [width-1, height-1])
    low, high = points.min(0), points.max(0)+1
    mask = np.zeros((high[1]-low[1], high[0]-low[0]), np.uint8)
    cv2.fillPoly(mask, [points-low], 1)
    inside = labels[low[1]:high[1], low[0]:high[0]][mask > 0]
    inside = inside[inside != 255]
    return float((inside == ROAD).mean()) if inside.size else 0.


def _disk_road_fraction(center: list, labels: np.ndarray, image_size: tuple[float, float], radius_m: float, resolution: float) -> float:
    """以 center 为圆心、radius_m 为半径的圆内语义为道路的比例（无数据不计）。"""
    height, width = labels.shape[:2]
    scale_x, scale_y = width/image_size[0], height/image_size[1]
    x, y, r = int(center[0]*scale_x), int(center[1]*scale_y), max(1, round(radius_m/resolution*scale_x))
    top, left = max(y-r, 0), max(x-r, 0)
    patch = labels[top:y+r+1, left:x+r+1]
    rows, columns = np.ogrid[top-y:top-y+patch.shape[0], left-x:left-x+patch.shape[1]]
    inside = patch[rows*rows+columns*columns <= r*r]
    inside = inside[inside != 255]
    return float((inside == ROAD).mean()) if inside.size else 0.


def is_wide_road(point: list, labels: np.ndarray, image_size: tuple[float, float], resolution: float) -> bool:
    return _disk_road_fraction(point, labels, image_size, _WIDE_JUNCTION_RADIUS_M, resolution) >= _WIDE_JUNCTION_MIN_ROAD


def merge_wide_junctions(graph: nx.MultiGraph, labels: np.ndarray, image_size: tuple[float, float], resolution: float) -> list[dict]:
    """合并宽路面上成对的路口（原地）：两端都是路口的短边周围几乎全是路面时，两端收到短边中点。

    宽路口里同一个路口常被画成相隔十几到三十多米的两个节点。窄街道上的两个相邻路口周围有房屋绿地，路面占比低，不受影响。"""
    dropped: list[dict] = []
    while True:
        candidates = sorted((e for a, b, e in graph.edges(data=True) if a != b and e['length_m'] < _WIDE_JUNCTION_MAX_M
                             and graph.degree(a) >= 3 and graph.degree(b) >= 3), key=lambda e: (e['length_m'], e['edge_id']))
        edge = next((e for e in candidates if _disk_road_fraction(LineString(e['polyline_xy']).interpolate(.5, normalized=True).coords[0],
                                                                  labels, image_size, _WIDE_JUNCTION_RADIUS_M, resolution) >= _WIDE_JUNCTION_MIN_ROAD), None)
        if edge is None:
            return dropped
        center = LineString(edge['polyline_xy']).interpolate(.5, normalized=True)
        dropped.extend(drop_edges(graph, {edge['edge_id']}, 'wide_junction_merge'))
        joint = f"p{min(edge['u'], edge['v'])}"
        while joint in graph:
            joint += 'x'
        graph.add_node(joint, xy=[center.x, center.y])
        for node in (edge['u'], edge['v']):
            if graph.has_node(node):
                dropped.extend({**e, 'reason': 'wide_junction_merge'} for e in _reattach(graph, node, joint, resolution))
        recompress(graph, resolution)


def collapse_paved_rings(graph: nx.MultiGraph, labels: np.ndarray, image_size: tuple[float, float], min_road: float,
                         max_perimeter_m: float, resolution: float) -> list[dict]:
    """收掉内部几乎全是路面的环（原地）：宽路口里模型画出的网格、分隔带两侧的双线。

    真实街区中间是房屋和绿地，路面占比低，不受影响。紧凑的环收成一个路口节点（放在环的形心，外接的路接过来）；
    细长的环和两条边的环去掉一条边，优先保留能直行穿过路口的边。min_road 为 0 时关闭。"""
    dropped: list[dict] = []
    if min_road <= 0:
        return dropped
    for _ in range(graph.number_of_edges()):
        found = None
        for edges in _candidate_rings(graph, max_perimeter_m):
            polygon = _ring_polygon(edges)
            if polygon is not None and _road_fraction(polygon, labels, image_size) >= min_road:
                found = edges, polygon
                break
        if found is None:
            break
        edges, polygon = found
        if len(edges) == 2 or 4*math.pi*polygon.area/polygon.length**2 < _RING_COMPACT:
            dropped.extend(drop_edges(graph, {_weakest(edges, graph, resolution)['edge_id']}, 'paved_ring'))
        else:
            members = {e['u'] for e in edges} | {e['v'] for e in edges}
            dropped.extend(drop_edges(graph, {e['edge_id'] for e in edges}, 'paved_ring'))
            joint = f'p{min(members)}'
            while joint in graph:
                joint += 'x'
            graph.add_node(joint, xy=[polygon.centroid.x, polygon.centroid.y])
            for node in members:
                if graph.has_node(node):
                    dropped.extend({**e, 'reason': 'paved_ring'} for e in _reattach(graph, node, joint, resolution))
            if not graph.degree(joint):
                graph.remove_node(joint)
        recompress(graph, resolution)
    return dropped
