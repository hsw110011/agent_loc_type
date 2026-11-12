"""语义路线：路线收到语义道路中线，连接关系不变，没有语义证据时保持原形。"""
from __future__ import annotations

import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import yaml
from shapely.geometry import LineString

from agent.map_preprocess.utils import global_road_graph as roads
from agent.map_preprocess.utils.semantic_route import collapse_paved_rings, merge_wide_junctions, route_along_semantic

PLANT, ROAD = 2, 0
SIZE = (400., 200.)


def build(lines: dict[tuple[str, str], list | None], points: dict[str, list]) -> nx.MultiGraph:
    vertices = [{'node_id': n, 'xy': xy} for n, xy in points.items()]
    edges = [{'u': u, 'v': v, 'polyline_xy': line or [points[u], points[v]], 'confidence': .9} for (u, v), line in lines.items()]
    return roads.compress_degree_two(roads.raw_to_graph({'vertices': vertices, 'edges': edges}), 1.)


def strip_labels() -> np.ndarray:
    """白色道路在 y=92~108，其余为植被；1 m/px。"""
    labels = np.full((200, 400), PLANT, np.uint8)
    labels[92:109] = ROAD
    return labels


def t_junction() -> nx.MultiGraph:
    points = {'a': [0., 85.], 'j': [200., 85.], 'b': [399., 85.], 'c': [200., 199.]}
    line_a = [[0., 85.], [100., 85.], [200., 85.]]
    line_b = [[200., 85.], [300., 85.], [399., 85.]]
    line_c = [[200., 85.], [200., 140.], [200., 199.]]
    return build({('a', 'j'): line_a, ('j', 'b'): line_b, ('j', 'c'): line_c}, points)


def route(graph: nx.MultiGraph, labels: np.ndarray) -> dict:
    return route_along_semantic(graph, labels, SIZE, 1., corridor_m=25., node_snap_m=20., close_node_m=32.)


def edge_between(graph: nx.MultiGraph, a: str, b: str) -> dict:
    return next(e for *_, e in graph.edges(data=True) if {e['u'], e['v']} == {a, b})


class SemanticRouteTests(unittest.TestCase):
    def assert_consistent(self, graph: nx.MultiGraph) -> None:
        for *_, edge in graph.edges(data=True):
            self.assertEqual(list(edge['polyline_xy'][0]), list(graph.nodes[edge['u']]['xy']))
            self.assertEqual(list(edge['polyline_xy'][-1]), list(graph.nodes[edge['v']]['xy']))
            self.assertAlmostEqual(edge['length_m'], roads.length_px(edge['polyline_xy']))

    def test_road_moves_onto_white_center_and_junction_snaps(self):
        graph = t_junction()
        stats = route(graph, strip_labels())
        self.assertGreater(stats['on_road_after'], stats['on_road_before'])
        self.assertEqual(stats['nodes_moved'], 1)
        self.assertAlmostEqual(graph.nodes['j']['xy'][1], 100., delta=2.)
        middle = LineString(edge_between(graph, 'a', 'j')['polyline_xy']).interpolate(.5, normalized=True)
        self.assertAlmostEqual(middle.y, 100., delta=3.)  # 原线偏 15 m 时，防拐进支路的牵引会留下约 2.5 m
        self.assert_consistent(graph)

    def test_connections_and_road_ends_do_not_change(self):
        graph = t_junction()
        before = {frozenset((e['u'], e['v'])) for *_, e in graph.edges(data=True)}
        route(graph, strip_labels())
        self.assertEqual({frozenset((e['u'], e['v'])) for *_, e in graph.edges(data=True)}, before)
        self.assertEqual(graph.nodes['a']['xy'], [0., 85.])  # 路端不动，贴图边的路仍然贴边
        self.assertEqual(graph.nodes['b']['xy'], [399., 85.])

    def test_stretch_without_white_stays_on_original_line(self):
        graph = t_junction()
        route(graph, strip_labels())
        points = np.array(edge_between(graph, 'j', 'c')['polyline_xy'])
        self.assertLess(np.abs(points[points[:, 1] > 125, 0]-200.).max(), 3.)

    def test_no_semantic_road_keeps_the_shape(self):
        graph = t_junction()
        stats = route(graph, np.full((200, 400), PLANT, np.uint8))
        self.assertEqual(stats['nodes_moved'], 0)
        for *_, edge in graph.edges(data=True):
            self.assertLess(np.abs(np.array(edge['polyline_xy'])[:, 1 if edge['v'] != 'c' else 0]
                                   -(85. if edge['v'] != 'c' else 200.)).max(), 3.)

    def test_duplicate_edges_are_not_collapsed_onto_one_line(self):
        points = {'w': [0., 100.], 'a': [100., 100.], 'b': [300., 100.], 'e': [399., 100.]}
        north = [[100., 100.], [200., 60.], [300., 100.]]
        south = [[300., 100.], [200., 140.], [100., 100.]]
        graph = build({('w', 'a'): None, ('a', 'b'): north, ('b', 'a'): south, ('b', 'e'): None}, points)
        stats = route(graph, strip_labels())
        self.assertEqual(stats['edges_kept'], 2)
        ys = sorted(np.array(e['polyline_xy'])[:, 1].min() for *_, e in graph.edges(data=True) if {e['u'], e['v']} == {'a', 'b'})
        self.assertLess(ys[0], 70.)
        self.assertGreater(ys[1], 90.)

    def _ring(self) -> nx.MultiGraph:
        points = {'a': [150., 70.], 'b': [190., 70.], 'c': [190., 110.], 'd': [150., 110.],
                  'w': [0., 70.], 'e': [399., 70.], 'n': [190., 0.], 's': [150., 199.], 'f': [399., 110.]}
        return build({('a', 'b'): None, ('b', 'c'): None, ('c', 'd'): None, ('d', 'a'): None,
                      ('w', 'a'): None, ('b', 'e'): None, ('n', 'b'): None, ('d', 's'): None, ('c', 'f'): None}, points)

    def test_paved_ring_collapses_to_one_junction(self):
        labels = np.full((200, 400), PLANT, np.uint8)
        labels[60:120, 140:200] = ROAD
        graph = self._ring()
        dropped = collapse_paved_rings(graph, labels, SIZE, .6, 300., 1.)
        self.assertEqual({d['reason'] for d in dropped}, {'paved_ring'})
        center = [n for n in graph.nodes if graph.nodes[n]['xy'] == [170., 90.]]
        self.assertEqual(len(center), 1)
        self.assertEqual(graph.degree(center[0]), 5)
        self.assertTrue(nx.is_connected(graph))

    def test_ring_around_a_block_is_kept(self):
        graph = self._ring()
        self.assertEqual(collapse_paved_rings(graph, np.full((200, 400), PLANT, np.uint8), SIZE, .6, 300., 1.), [])
        self.assertEqual(graph.number_of_edges(), 9)

    def test_dual_carriageway_keeps_one_line(self):
        points = {'w': [0., 100.], 'a': [150., 100.], 'b': [250., 100.], 'e': [399., 100.], 'm': [200., 90.], 'k': [200., 110.]}
        labels = np.full((200, 400), PLANT, np.uint8)
        labels[85:115] = ROAD
        graph = build({('w', 'a'): None, ('b', 'e'): None, ('a', 'm'): None, ('m', 'b'): None, ('a', 'k'): None, ('k', 'b'): None}, points)
        self.assertEqual(graph.number_of_edges(), 4)
        collapse_paved_rings(graph, labels, SIZE, .6, 300., 1.)
        self.assertEqual(graph.number_of_edges(), 1)  # 只剩一条线，两端的路压缩成一条贯通路

    def _pair(self) -> nx.MultiGraph:
        points = {'j': [190., 100.], 'k': [210., 100.], 'w': [0., 100.], 'e': [399., 100.], 'n': [190., 0.], 's': [210., 199.]}
        return build({('w', 'j'): None, ('j', 'k'): None, ('k', 'e'): None, ('n', 'j'): None, ('k', 's'): None}, points)

    def test_paired_junctions_on_wide_road_merge(self):
        graph = self._pair()
        dropped = merge_wide_junctions(graph, np.full((200, 400), ROAD, np.uint8), SIZE, 1.)
        self.assertEqual({d['reason'] for d in dropped}, {'wide_junction_merge'})
        self.assertEqual(graph.number_of_nodes(), 5)
        self.assertEqual(max(d for _, d in graph.degree), 4)

    def test_paired_junctions_on_narrow_street_stay(self):
        graph = self._pair()
        self.assertEqual(merge_wide_junctions(graph, strip_labels(), SIZE, 1.), [])
        self.assertEqual(graph.number_of_edges(), 5)

    def _ridge(self) -> np.ndarray:
        probability = np.zeros((200, 400), np.float32)
        for y in range(200):
            probability[y] = np.exp(-((y-70.)/8.)**2)
        return probability

    def test_route_prefers_probability_ridge_over_semantic_road(self):
        graph = t_junction()
        route_along_semantic(graph, strip_labels(), SIZE, 1., 40., 8., 16., self._ridge())
        edge = next(e for e in roads.edge_records(graph) if {e['u'], e['v']} == {'a', 'j'})
        self.assertAlmostEqual(float(np.median(np.asarray(edge['polyline_xy'])[:, 1])), 70., delta=4.)

    def test_weak_probability_falls_back_to_semantic_road(self):
        probability = self._ridge()
        probability[:, 60:140] = 0.  # 模型在这一段没有概率
        graph = t_junction()
        route_along_semantic(graph, strip_labels(), SIZE, 1., 40., 8., 16., probability)
        edge = next(e for e in roads.edge_records(graph) if {e['u'], e['v']} == {'a', 'j'})
        middle = np.asarray([p for p in edge['polyline_xy'] if 85. <= p[0] <= 115.])
        self.assertGreater(float(middle[:, 1].mean()), 80.)  # 概率断开处向语义路面靠拢

    def test_both_pipeline_configs_enable_it(self):
        directory = Path(__file__).resolve().parents[1]
        self.assertIs(yaml.safe_load((directory/'pipeline.yaml').read_text(encoding='utf-8'))['road_semantic_route_enabled'], True)


if __name__ == '__main__':
    unittest.main()
