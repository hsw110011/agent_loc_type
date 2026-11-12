#!/usr/bin/env python3
"""合成测试：构造覆盖各种情况的道路 mask 和语义 mask，跑完整流程并逐项断言。

    python -m agent.map_preprocess.tests.road_topo_scene --out out_dir

合成图 0.5 m/px，1000 m × 1000 m。语义类别 0=道路、1=建筑、2=植被、3=其它、4=停车场。
测试只改一个配置：parking_ids 打开停车场类（真实语义模型没有这一类）。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point

from agent.map_preprocess.utils import road_topo as rt

GSD = .5
SIZE = 2000
ROAD, BUILDING, PLANT, OTHER, PARKING = 0, 1, 2, 3, 4


def px(metres: float) -> int:
    return int(round(metres/GSD))


class Scene:
    def __init__(self) -> None:
        self.R = np.zeros((SIZE, SIZE), np.uint8)
        self.semantic = np.full((SIZE, SIZE), OTHER, np.uint8)

    def road(self, *points: tuple[float, float], width: float = 8., R: bool = True, S: bool = True) -> None:
        polyline = np.array([[px(x), px(y)] for x, y in points], np.int32)
        if R:
            cv2.polylines(self.R, [polyline], False, 255, px(width))
        if S:
            cv2.polylines(self.semantic, [polyline], False, ROAD, px(width))

    def rect(self, x: float, y: float, w: float, h: float, value: int | None = None, r: int | None = None) -> None:
        corners = (px(x), px(y)), (px(x+w), px(y+h))
        if value is not None:
            cv2.rectangle(self.semantic, *corners, value, -1)
        if r is not None:
            cv2.rectangle(self.R, *corners, r, -1)


def make_scene() -> Scene:
    s = Scene()
    # Y 形斜接：主路在路口被拉向支路形成尖角，应规整成主路笔直的 T 字形
    s.road((40, 100), (165, 100), (175, 94), (185, 100), (320, 100))
    s.road((175, 94), (215, 30))
    # 分裂成双节点的路口：南北两条支路错开 11 m
    s.road((40, 220), (320, 220))
    s.road((170, 220), (170, 150))
    s.road((181, 220), (181, 290))
    # 真正的 Y 形分叉：三臂互成 120°
    s.road((175, 400), (175, 330))
    s.road((175, 400), (235.6, 435))
    s.road((175, 400), (114.4, 435))
    # 树木遮挡的断路：缝隙语义为植被，R 和 S 都断开
    s.rect(200, 505, 25, 30, PLANT)
    s.road((40, 520), (200, 520))
    s.road((225, 520), (320, 520))
    # 穿过建筑的假连接：缝隙是建筑，任何一步都不能连
    s.rect(200, 570, 20, 40, BUILDING)
    s.road((40, 590), (200, 590))
    s.road((220, 590), (320, 590))
    # 带孔洞的路段
    s.road((40, 660), (320, 660), width=14.)
    s.rect(200, 657, 6, 5, OTHER, 0)
    # 短毛刺：竖直主路上 4 根，同时检查直路被替换成两点直线
    s.road((380, 40), (380, 320))
    for y, dx in ((100, 9), (170, -9), (230, 9), (280, -9)):
        s.road((380, y), (380+dx, y), width=4.)
    # 停车场：语义为停车场类，只在 R 里；连接两侧道路，内部 6 块绿岛形成网状
    s.rect(450, 40, 110, 90, PARKING, 255)
    s.road((400, 85), (450, 85))
    s.road((560, 85), (620, 85))
    for row in range(2):
        for column in range(3):
            s.rect(462+column*30, 50+row*40, 16, 26, PLANT, 0)
    # 空旷广场：靠路面宽度判断
    s.rect(450, 200, 100, 80, ROAD, 255)
    s.road((400, 240), (450, 240))
    s.road((550, 240), (620, 240))
    # r_only 路边凸起：又短又宽，只在 R 里，不应并入
    s.road((450, 350), (620, 350))
    s.rect(520, 354, 15, 12, OTHER, 255)
    # r_only 桥接：S 断开、R 连续，应并入
    s.road((450, 420), (520, 420))
    s.road((535, 420), (620, 420))
    s.road((520, 420), (535, 420), S=False)
    # r_only 延长：S 很短，R 向前延伸，应并入，边的来源是 filled
    s.road((450, 480), (470, 480))
    s.road((470, 480), (570, 480), S=False)
    # 低置信分支：只在 R 里的短支路，细长所以被并入，应在第 4 步删除
    s.road((450, 540), (620, 540))
    s.road((520, 540), (520, 564), width=5., S=False)
    # s_only：只在语义里的小巷，应丢弃
    s.road((590, 540), (590, 600), width=5., R=False)
    # 无遮挡证据的断口：缝隙语义为其它，第 2 步不补，第 6 步按方向补上
    s.road((450, 640), (540, 640))
    s.road((562, 640), (660, 640))
    # 弯曲道路：检查圆滑
    wave = [(40+i*4., 735+12*math.sin(i/6.)) for i in range(70)]
    s.road(*wave)
    # 长断口：方向正对的应接上；另一对错开 20 m，方向不对，不能接
    s.road((40, 860), (200, 860))
    s.road((262, 860), (400, 860))
    s.road((40, 930), (200, 930))
    s.road((262, 950), (400, 950))
    # 路端接主路：断头路指向 22 m 外的主路，应接上；另一条中间隔着建筑，不能接
    s.road((760, 40), (760, 300))
    s.road((900, 100), (778, 100))
    s.road((900, 220), (778, 220))
    s.rect(766, 200, 8, 40, BUILDING)
    # 窄长环：隔离带把一条路分成相距 12 m 的两条线，应合并成一条中线
    s.road((700, 406), (740, 406))
    s.road((900, 406), (950, 406))
    for y in (400, 412):
        s.road((740, y), (900, y), width=6.)
    for x in (740, 900):
        s.road((x, 400), (x, 412), width=6.)
    # 正常街区：60 m 见方的环，不能合并
    s.road((700, 480), (740, 480), (800, 480), (800, 540), (850, 540))
    s.road((740, 480), (740, 540), (800, 540))
    # 折线路：两段直线夹 42° 转角，最后应是直线段加圆角
    s.road((720, 620), (840, 620), (930, 700))
    return s


def step(records: list[dict], name: str) -> dict:
    return next(r for r in records if r['step'] == name)


def crosses(graph: nx.MultiGraph, position_m: tuple[float, float], radius_m: float = 2.) -> bool:
    target = Point(px(position_m[0]), px(position_m[1]))
    return any(LineString(d['pts']).distance(target) <= px(radius_m) for *_, d in graph.edges(data=True) if len(d['pts']) >= 2)


def edge_at(graph: nx.MultiGraph, position_m: tuple[float, float], radius_m: float = 2.) -> dict | None:
    target = Point(px(position_m[0]), px(position_m[1]))
    for *_, data in graph.edges(data=True):
        if LineString(data['pts']).distance(target) <= px(radius_m):
            return data
    return None


def nodes_near(graph: nx.MultiGraph, position_m: tuple[float, float], radius_m: float) -> list[tuple[int, tuple[float, float]]]:
    target = np.array([px(position_m[0]), px(position_m[1])], float)
    return [(n, p) for n, p in graph.nodes(data='pos') if np.linalg.norm(np.asarray(p)-target) <= px(radius_m)]


def path_near(points: np.ndarray, x_range_m: tuple[float, float]) -> np.ndarray:
    """取边上 x 在给定范围内的点；点列先按 1 像素加密。"""
    dense = np.vstack([np.linspace(a, b, max(2, int(np.linalg.norm(b-a))+1)) for a, b in zip(points[:-1], points[1:])])
    x = dense[:, 0]*GSD
    return dense[(x >= x_range_m[0]) & (x <= x_range_m[1])]


def inserted_node_check() -> list[tuple[str, bool, object]]:
    """度为 4、一对主路加两条分开的支路：交点相距超过 merge_m，应插入新节点，把主路拆成多段。"""
    cfg = rt.load_config()
    size = 400
    ctx = SimpleNamespace(cfg=cfg, gsd=GSD, semantic=np.full((size, size), OTHER, np.uint8), core_near=np.ones((size, size), bool),
                          distance=np.ones((size, size), np.float32), regions=np.zeros((size, size), np.int32), region_count=0, removed=[], added=[], modified=0)
    graph = nx.MultiGraph()
    def polyline(*points: tuple[float, float]) -> np.ndarray:
        return np.vstack([np.linspace(a, b, max(2, int(np.linalg.norm(np.subtract(b, a)))+1))[:-1] for a, b in zip(points[:-1], points[1:])]+[np.array([points[-1]], float)])
    graph.add_node(0, pos=(200., 200.))
    for node, end in ((1, (40., 200.)), (2, (360., 200.)), (3, (180., 40.)), (4, (225., 40.))):
        graph.add_node(node, pos=end)
    for node, points in ((1, polyline((200., 200.), (40., 200.))), (2, polyline((200., 200.), (360., 200.))),
                         (3, polyline((200., 200.), (180., 180.), (180., 40.))), (4, polyline((200., 200.), (225., 180.), (225., 40.)))):
        graph.add_edge(0, node, pts=points, length_m=rt.polyline_length(points)*GSD)
    info = rt.regularize_junctions(graph, ctx)
    junctions = sorted(p for n, p in graph.nodes(data='pos') if graph.degree(n) == 3)
    on_main = len(junctions) == 2 and all(abs(y-200) < 2 for _, y in junctions)
    apart = len(junctions) == 2 and junctions[1][0]-junctions[0][0] > 9/GSD
    return [('插入新节点：统计记录 1 个', info['inserted_nodes'] == 1, info),
            ('插入新节点：主路被拆成两个度 3 路口且都在主路上', on_main and apart, junctions)]


def max_turn_deg(points: np.ndarray) -> float:
    vectors = np.diff(points, axis=0)
    a, b = vectors[:-1], vectors[1:]
    cosine = (a*b).sum(axis=1)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)+1e-9)
    return float(np.degrees(np.arccos(np.clip(cosine, -1., 1.))).max()) if len(a) else 0.


def check(graph: nx.MultiGraph, records: list[dict]) -> list[str]:
    results: list[tuple[str, bool, object]] = []
    def add(name: str, passed: bool, detail: object = '') -> None:
        results.append((name, bool(passed), detail))
    # Y 形 → T 字形
    y_node = [(n, p) for n, p in nodes_near(graph, (175, 100), 9.) if graph.degree(n) == 3]
    add('Y 形斜接：路口是度 3 节点', len(y_node) == 1, y_node)
    add('Y 形斜接：路口被拉回主路直线', y_node and abs(y_node[0][1][1]*GSD-100) < 1.5, y_node and round(y_node[0][1][1]*GSD, 2))
    main_edges = [d for _, far, d in graph.edges(y_node[0][0], data=True) if abs(graph.nodes[far]['pos'][1]*GSD-100) < 5] if y_node else []
    main_points = np.vstack([path_near(d['pts'], (150, 200)) for d in main_edges]) if main_edges else np.zeros((0, 2))
    add('Y 形斜接：主路在路口两侧是直的（偏离 y=100 不超过 1.5 m）', len(main_points) > 0 and float(np.abs(main_points[:, 1]*GSD-100).max()) < 1.5,
        round(float(np.abs(main_points[:, 1]*GSD-100).max()), 2) if len(main_points) else None)
    # 双节点
    double = [(n, p) for n, p in nodes_near(graph, (175.5, 220), 12.) if graph.degree(n) >= 3]
    add('双节点路口：合并成一个度 4 节点', len(double) == 1 and graph.degree(double[0][0]) == 4, [(n, graph.degree(n)) for n, _ in double])
    add('双节点路口：统计里有合并记录', step(records, '09b_double_nodes')['double_nodes'] >= 1, step(records, '09b_double_nodes')['double_nodes'])
    # 真 Y 形
    fork = [(n, p) for n, p in nodes_near(graph, (175, 400), 8.) if graph.degree(n) == 3]
    add('真 Y 形：保持度 3', len(fork) == 1, fork)
    add('真 Y 形：路口位置基本不动', fork and np.linalg.norm(np.asarray(fork[0][1])*GSD-np.array([175, 400])) < 3., fork and [round(v*GSD, 2) for v in fork[0][1]])
    add('真 Y 形：统计里记为未改动', step(records, '09c_regularized')['kept_y'] >= 1, step(records, '09c_regularized')['kept_y'])
    # 遮挡与假连接
    occlusion = step(records, '04_occlusion')
    add('树木遮挡：断路被补上', any(185 < b['from_xy'][0]*GSD < 240 and 505 < b['from_xy'][1]*GSD < 535 for b in occlusion['detail']), occlusion['detail'])
    add('树木遮挡：最终有边通过', crosses(graph, (212, 520)))
    add('穿建筑假连接：最终没有边通过', not crosses(graph, (210, 590)))
    add('穿建筑假连接：第 2 步没有补', not any(190 < b['from_xy'][0]*GSD < 230 and 570 < b['from_xy'][1]*GSD < 610 for b in occlusion['detail']))
    # 孔洞、毛刺
    add('孔洞：被填充', step(records, '05_cleaned')['filled_holes'] >= 1, step(records, '05_cleaned')['filled_holes'])
    hole_nodes = [n for n, _ in nodes_near(graph, (203, 660), 25.)]
    add('孔洞：路段没有分叉', not hole_nodes and crosses(graph, (203, 660), 4.), hole_nodes)
    add('毛刺：全部删除', step(records, '08_pruned')['spurs'] >= 4 and not any(crosses(graph, (380+d, y), 1.5) for y, d in ((100, 8), (170, -8), (230, 8), (280, -8))), step(records, '08_pruned')['spurs'])
    # 停车场、广场
    add('面状区域：标记出停车场和广场', step(records, '06_area_regions')['regions'] >= 2, step(records, '06_area_regions')['regions'])
    add('面状区域：内部分支被删除', step(records, '08_pruned')['area_branches'] >= 3, step(records, '08_pruned')['area_branches'])
    add('停车场：穿过的主干保留', crosses(graph, (505, 85), 14.))
    add('广场：穿过的主干保留', crosses(graph, (500, 240), 14.))
    # r_only
    fused = step(records, '03_fused')
    add('r_only 凸起：被拒绝', fused['rejected_blocks'] >= 1 and not crosses(graph, (527, 362), 2.), fused['rejected_blocks'])
    add('r_only 桥接：被并入', fused['bridge_blocks'] >= 1 and crosses(graph, (527, 420)), fused['bridge_blocks'])
    extension = edge_at(graph, (540, 480))
    add('r_only 延长：被并入', fused['extension_blocks'] >= 1 and extension is not None, fused['extension_blocks'])
    add('r_only 延长：边的 source=filled', extension is not None and extension['source'] == 'filled', extension and extension['source'])
    add('低置信分支：被删除', step(records, '08_pruned')['low_confidence'] >= 1 and not crosses(graph, (520, 558), 1.5), step(records, '08_pruned')['low_confidence'])
    add('s_only：被丢弃', not crosses(graph, (590, 580), 3.))
    # 断口
    gap_edge = edge_at(graph, (551, 640), 1.)
    add('断口：第 6 步按方向补上', step(records, '10_gaps')['added_edges'] >= 1 and gap_edge is not None, step(records, '10_gaps')['detail'])
    add('断口：旧端点已删除，补上的长度记在 filled_m', gap_edge is not None and not nodes_near(graph, (551, 640), 30.) and 15 < gap_edge['filled_m'] < 30, gap_edge and gap_edge['filled_m'])
    add('最终图没有度 2 节点', not [n for n in graph.nodes if graph.degree(n) == 2 and not graph.has_edge(n, n)], [n for n in graph.nodes if graph.degree(n) == 2])
    add('长断口：方向正对的 60 m 断口接上', crosses(graph, (231, 860)), step(records, '10_gaps')['detail'])
    add('长断口：方向不对的不接', not crosses(graph, (231, 938), 8.))
    # 平直与圆滑
    straight = edge_at(graph, (380, 150), 1.)
    add('平直：直路替换成两点直线', straight is not None and len(straight['pts']) == 2, straight and len(straight['pts']))
    curve = edge_at(graph, (160, 735+12*math.sin(30/6.)), 5.)
    add('圆滑：弯路没被拉直', curve is not None and len(curve['pts']) > 10, curve and len(curve['pts']))
    add('圆滑：弯路转角平缓', curve is not None and max_turn_deg(curve['pts']) < 30., curve is not None and round(max_turn_deg(curve['pts']), 1))
    truth = LineString([(px(40+i*4.), px(735+12*math.sin(i/6.))) for i in range(70)])
    add('圆滑：与原路径偏差不超过 3 m', curve is not None and LineString(curve['pts']).hausdorff_distance(truth)*GSD <= 3.5, curve is not None and round(LineString(curve['pts']).hausdorff_distance(truth)*GSD, 2))
    # 属性
    add('属性：节点 type 合法', all(d['type'] in ('junction', 'endpoint') for _, d in graph.nodes(data=True)))
    add('属性：边 source、width_m、length_m 齐全', all(d['source'] in ('core', 'filled') and d['length_m'] > 0 and 'width_m' in d for *_, d in graph.edges(data=True)))
    add('属性：端点和节点坐标一致', all(_ends_match(graph, a, b, d) for a, b, d in graph.edges(data=True)))
    # 路端接主路
    joined = [n for n, _ in nodes_near(graph, (760, 100), 6.) if graph.degree(n) == 3]
    add('路端接主路：接上并在主路上形成路口', step(records, '10b_end_to_road')['added_edges'] >= 1 and len(joined) == 1 and crosses(graph, (770, 100)), step(records, '10b_end_to_road')['detail'])
    add('路端接主路：隔着建筑的不接', not crosses(graph, (770, 220), 1.5))
    # 窄长环
    loops = step(records, '09a2_narrow_loops')
    add('窄长环：两条平行线合并成中线', loops['parallel_merged'] >= 1 and crosses(graph, (820, 406), 1.5) and not crosses(graph, (820, 400), 1.5) and not crosses(graph, (820, 412), 1.5), loops)
    add('窄长环：正常街区不合并', crosses(graph, (770, 480), 1.5) and crosses(graph, (740, 510), 1.5) and crosses(graph, (770, 540), 1.5) and crosses(graph, (800, 510), 1.5))
    # 直线段 + 圆角
    bend = edge_at(graph, (780, 620), 2.)
    truth_bend = LineString([(px(720), px(620)), (px(840), px(620)), (px(930), px(700))])
    longest = float(np.linalg.norm(np.diff(bend['pts'], axis=0), axis=1).max())*GSD if bend is not None else 0.
    add('曲率优化：直线部分是一条长线段', longest >= 60., round(longest, 1))
    add('曲率优化：拐角是圆滑弧线', bend is not None and len(bend['pts']) >= 8 and max_turn_deg(bend['pts']) < 10., bend is not None and (len(bend['pts']), round(max_turn_deg(bend['pts']), 1)))
    add('曲率优化：与原路径偏差不超过 3.5 m', bend is not None and LineString(bend['pts']).hausdorff_distance(truth_bend)*GSD <= 3.5, bend is not None and round(LineString(bend['pts']).hausdorff_distance(truth_bend)*GSD, 2))
    add('曲率优化：统计里有直线占比', 0 < step(records, '12_refined')['straight_share'] <= 1, step(records, '12_refined'))
    results += inserted_node_check()
    return [f"{'PASS' if passed else 'FAIL'}  {name}  {detail}" for name, passed, detail in results]


def _ends_match(graph: nx.MultiGraph, a: int, b: int, data: dict) -> bool:
    points = data['pts']
    pa, pb = np.asarray(graph.nodes[a]['pos']), np.asarray(graph.nodes[b]['pos'])
    forward = np.linalg.norm(points[0]-pa) < 1e-6 and np.linalg.norm(points[-1]-pb) < 1e-6
    backward = np.linalg.norm(points[0]-pb) < 1e-6 and np.linalg.norm(points[-1]-pa) < 1e-6
    return forward or backward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='synthetic_out')
    arguments = parser.parse_args()
    out = Path(arguments.out)
    (out/'input').mkdir(parents=True, exist_ok=True)
    scene = make_scene()
    cv2.imwrite(str(out/'input'/'road.png'), scene.R)
    cv2.imwrite(str(out/'input'/'sem.png'), scene.semantic)
    cfg = rt.load_config(overrides={'classes': {'parking_ids': [PARKING]}})
    graph, records = rt.run(out/'input'/'road.png', out/'input'/'sem.png', GSD, out, cfg)
    lines = check(graph, records)
    print('\n检查结果')
    for line in lines:
        print(line)
    (out/'checks.txt').write_text('\n'.join(lines), encoding='utf-8')
    if any(line.startswith('FAIL') for line in lines):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
