"""道路拓扑优化：并行去重、近距连通、短路删除、曲率平滑。"""
from __future__ import annotations

import math
import tempfile
import unittest

import networkx as nx
import numpy as np
from shapely.geometry import LineString

from agent.map_preprocess.global_config import GlobalPreprocessConfig
from agent.map_preprocess.nodes import global_pipeline
from agent.map_preprocess.utils import global_road_graph as roads, road_refine


def build(lines: dict[tuple[str, str], list], points: dict[str, list]) -> nx.MultiGraph:
    """1 m/px 的压缩图：points 是节点坐标，lines 是 (u, v) -> 折线（省略时直连）。"""
    vertices = [{'node_id': n, 'xy': xy} for n, xy in points.items()]
    edges = [{'u': u, 'v': v, 'polyline_xy': line or [points[u], points[v]], 'confidence': .9}
             for (u, v), line in lines.items()]
    graph = roads.compress_degree_two(roads.raw_to_graph({'vertices': vertices, 'edges': edges}), 1.)
    score(graph)
    return graph


def score(graph: nx.MultiGraph) -> None:
    config = GlobalPreprocessConfig(segmenter='dummy')
    for *_, edge in graph.edges(data=True):
        edge['edge_score'] = roads.score_edge(edge, graph, config)


def assert_consistent(test: unittest.TestCase, graph: nx.MultiGraph) -> None:
    for a, b, edge in graph.edges(data=True):
        test.assertEqual({edge['u'], edge['v']}, {a, b})
        test.assertEqual(list(edge['polyline_xy'][0]), list(graph.nodes[edge['u']]['xy']))
        test.assertEqual(list(edge['polyline_xy'][-1]), list(graph.nodes[edge['v']]['xy']))
        test.assertAlmostEqual(edge['length_m'], roads.length_px(edge['polyline_xy']))


def through(graph: nx.MultiGraph, close: float = 32., cycle: float = 120., margin: float = 10.,
            size: tuple[float, float] = (1000., 1000.), support=None, min_support: float = 0.) -> list[dict]:
    return road_refine.enforce_through_network(graph, close_node_m=close, small_cycle_m=cycle, border_margin_m=margin,
                                               image_size=size, resolution=1., road_support_fn=support,
                                               duplicate_min_support=min_support)


def reasons(dropped: list[dict]) -> set[str]:
    return {d['reason'] for d in dropped}


class ThroughNetworkTests(unittest.TestCase):
    def test_dead_end_is_removed_in_layers_but_border_end_stays(self):
        points = {'w': [0., 500.], 'j': [300., 500.], 'e': [999., 500.], 's': [300., 700.], 'k': [300., 800.], 't': [400., 800.], 'q': [300., 900.]}
        graph = build({('w', 'j'): None, ('j', 'e'): None, ('j', 's'): None, ('s', 'k'): None, ('k', 't'): None, ('k', 'q'): None}, points)
        dropped = through(graph)
        self.assertEqual(reasons(dropped), {'dead_end_road'})
        self.assertEqual(graph.number_of_edges(), 1)  # 断头支路逐层删完，w-j-e 压缩成一条贯通路
        assert_consistent(self, graph)

    def test_disconnected_road_is_removed(self):
        points = {'a': [0., 100.], 'b': [999., 100.], 'c': [0., 800.], 'd': [200., 800.]}
        graph = build({('a', 'b'): None, ('c', 'd'): [[0., 800.], [100., 800.], [200., 800.]]}, points)
        through(graph, margin=250.)
        self.assertEqual({tuple(sorted((e['u'], e['v']))) for e in roads.edge_records(graph)}, {('a', 'b')})

    def _loop_graph(self, side: float) -> nx.MultiGraph:
        points = {'w': [0., 500.], 'a': [400., 500.], 'b': [400.+side, 500.], 'c': [400.+side/2, 500.-side], 'e': [999., 500.]}
        return build({('w', 'a'): None, ('a', 'b'): None, ('a', 'c'): None, ('b', 'c'): None, ('b', 'e'): None}, points)

    def test_small_cycle_keeps_low_confidence_through_road(self):
        points = {'w': [0., 500.], 'a': [400., 500.], 'b': [430., 500.], 'e': [999., 500.], 'c': [415., 470.]}
        graph = build({('w', 'a'): None, ('a', 'b'): None, ('a', 'c'): None, ('b', 'c'): None, ('b', 'e'): None}, points)
        for edge in roads.edge_records(graph):
            if {edge['u'], edge['v']} == {'a', 'b'}:
                graph[edge['u']][edge['v']][edge['edge_id']]['confidence'] = .5
        through(graph, close=0.)
        self.assertTrue(nx.is_connected(graph))
        self.assertAlmostEqual(sum(e['length_m'] for e in roads.edge_records(graph)), 999.)  # 主路直行段保留，绕行的两条边合并后被删

    def test_branch_along_trunk_gets_its_own_junction(self):
        points = {'w': [0., 500.], 'j': [300., 500.], 'e': [800., 500.], 'q': [330., 300.], 's': [300., 900.]}
        graph = build({('w', 'j'): None, ('j', 'e'): None, ('j', 's'): None,
                       ('j', 'q'): [[300., 500.], [330., 500.], [330., 300.]]}, points)
        result = road_refine.split_shared_stretches(graph, 10., 1.)
        self.assertEqual(result['splits'], 1)
        assert_consistent(self, graph)
        self.assertAlmostEqual(result['shared_m'][0], 30., delta=2.)
        fork = [n for n in graph.nodes if abs(graph.nodes[n]['xy'][0]-330.) < 3. and abs(graph.nodes[n]['xy'][1]-500.) < 3.]
        self.assertEqual(len(fork), 1)
        self.assertEqual(graph.degree(fork[0]), 3)
        self.assertIn('j', list(graph.neighbors(fork[0])))
        self.assertAlmostEqual(sum(e['length_m'] for e in roads.edge_records(graph)), 300.+500.+400.+200., delta=3.)  # 重合段只算一次

    def test_short_shared_stretch_is_left_alone(self):
        points = {'w': [0., 500.], 'j': [300., 500.], 'e': [800., 500.], 'q': [305., 300.]}
        graph = build({('w', 'j'): None, ('j', 'e'): None, ('j', 'q'): [[300., 500.], [305., 500.], [305., 300.]]}, points)
        self.assertEqual(road_refine.split_shared_stretches(graph, 10., 1.)['splits'], 0)
        self.assertEqual(road_refine.split_shared_stretches(graph, 0., 1.)['splits'], 0)
        self.assertEqual(graph.number_of_edges(), 3)

    def test_crossing_splits_into_two_nodes_of_three_roads(self):
        points = {'c': [500., 500.], 'w': [0., 500.], 'e': [999., 500.], 'n': [500., 0.], 's': [500., 999.]}
        graph = build({('w', 'c'): None, ('c', 'e'): None, ('n', 'c'): None, ('c', 's'): None}, points)
        self.assertEqual(road_refine.split_crowded_junctions(graph, 10., 1.), 0)  # 默认最多 4 条路，十字路口不拆
        self.assertEqual(road_refine.split_crowded_junctions(graph, 10., 1., 3), 1)
        assert_consistent(self, graph)
        self.assertEqual(graph.number_of_edges(), 5)
        self.assertEqual(sorted(d for _, d in graph.degree), [1, 1, 1, 1, 3, 3])
        self.assertTrue(nx.is_connected(graph))

    def test_five_way_junction_ends_with_at_most_three_roads(self):
        points = {'c': [500., 500.], **{f'p{i}': [500.+400.*math.cos(i*1.2566), 500.+400.*math.sin(i*1.2566)] for i in range(5)}}
        graph = build({('c', f'p{i}'): None for i in range(5)}, points)
        self.assertEqual(road_refine.split_crowded_junctions(graph, 10., 1.), 1)
        assert_consistent(self, graph)
        self.assertLessEqual(max(d for _, d in graph.degree), 4)

    def test_t_junction_is_left_alone(self):
        points = {'c': [500., 500.], 'w': [0., 500.], 'e': [999., 500.], 's': [500., 999.]}
        graph = build({('w', 'c'): None, ('c', 'e'): None, ('c', 's'): None}, points)
        self.assertEqual(road_refine.split_crowded_junctions(graph, 10., 1.), 0)

    def test_small_cycle_is_opened_without_disconnecting(self):
        graph = self._loop_graph(35.)
        dropped = through(graph, close=0.)
        self.assertIn('small_cycle', reasons(dropped))
        self.assertTrue(nx.is_connected(graph))
        self.assertEqual(nx.cycle_basis(nx.Graph(graph)), [])

    def test_large_cycle_is_kept(self):
        graph = self._loop_graph(200.)
        edges = graph.number_of_edges()
        self.assertEqual(through(graph, close=0.), [])
        self.assertEqual(graph.number_of_edges(), edges)

    def test_self_loop_is_removed_whatever_its_length(self):
        ring = [[400., 500.], [400., 300.], [700., 300.], [700., 500.], [400., 500.]]  # 周长 1000 m，远超小环上限
        points = {'w': [0., 500.], 'j': [400., 500.], 'e': [999., 500.], 's': [400., 999.]}
        graph = build({('w', 'j'): None, ('j', 'e'): None, ('j', 's'): None, ('j', 'j'): ring}, points)
        dropped = through(graph)
        self.assertEqual(reasons(dropped), {'self_loop'})
        self.assertEqual(graph.number_of_edges(), 3)
        self.assertFalse(any(a == b for a, b in graph.edges()))
        assert_consistent(self, graph)

    def _duplicate_graph(self) -> nx.MultiGraph:
        """a、b 之间一条走北侧、一条走南侧，周长 566 m，不算小环。"""
        points = {'w': [0., 500.], 'a': [400., 500.], 'b': [600., 500.], 'e': [999., 500.]}
        north = [[400., 500.], [500., 400.], [600., 500.]]
        south = [[600., 500.], [500., 600.], [400., 500.]]
        return build({('w', 'a'): None, ('a', 'b'): north, ('b', 'a'): south, ('b', 'e'): None}, points)

    @staticmethod
    def _by_side(north: float, south: float):
        return lambda line: south if max(p[1] for p in line) > 500 else north

    def test_weak_duplicate_is_removed_and_stronger_one_kept(self):
        graph = self._duplicate_graph()
        dropped = through(graph, support=self._by_side(.9, .2), min_support=.6)
        self.assertEqual(reasons(dropped), {'weak_duplicate'})
        self.assertEqual(dropped[0]['road_support'], .2)
        self.assertEqual(graph.number_of_edges(), 1)  # 只剩一条后 w-a-b-e 压缩成一条贯通路
        self.assertEqual(min(p[1] for p in roads.edge_records(graph)[0]['polyline_xy']), 400.)
        assert_consistent(self, graph)

    def test_two_supported_duplicates_both_stay(self):
        graph = self._duplicate_graph()
        self.assertEqual(through(graph, support=self._by_side(.9, .7), min_support=.6), [])
        self.assertEqual(graph.number_of_edges(), 4)

    def test_strongest_duplicate_stays_even_when_all_are_weak(self):
        graph = self._duplicate_graph()
        dropped = through(graph, support=self._by_side(.3, .2), min_support=.6)
        self.assertEqual(len(dropped), 1)
        self.assertTrue(nx.is_connected(graph))
        self.assertEqual(min(p[1] for p in roads.edge_records(graph)[0]['polyline_xy']), 400.)

    def test_duplicates_stay_without_probability_or_when_disabled(self):
        for support, min_support in ((None, .6), (self._by_side(.9, .2), 0.)):
            graph = self._duplicate_graph()
            self.assertEqual(through(graph, support=support, min_support=min_support), [])

    def test_road_support_measures_length_on_probable_road(self):
        probability = np.zeros((10, 10), np.float32)
        probability[5, :5] = 1.  # 只有左半段有道路概率；图像 100 px，概率图 10 格
        self.assertEqual(road_refine.road_support([[0., 80.], [100., 80.]], probability, (100., 100.), .18, 1.), 0.)
        self.assertEqual(road_refine.road_support([[0., 50.], [30., 50.]], probability, (100., 100.), .18, 1.), 1.)
        self.assertAlmostEqual(road_refine.road_support([[0., 50.], [100., 50.]], probability, (100., 100.), .18, 1.), .45, delta=.06)

    def test_support_fn_needs_saved_probability_map(self):
        config = GlobalPreprocessConfig(segmenter='dummy')
        with tempfile.TemporaryDirectory() as folder:
            state = {'road_graph_cache': {'path': f'{folder}/raw_graph.json'}, 'image': {'width': 10, 'height': 10},
                     'resolution_m_per_px': 1.}
            self.assertIsNone(global_pipeline._road_support_fn(state, config))
            np.save(f'{folder}/road_probability.npy', np.ones((10, 10), np.float32))
            self.assertEqual(global_pipeline._road_support_fn(state, config)([[1., 5.], [8., 5.]]), 1.)

    def test_close_junctions_are_merged_until_none_remain(self):
        points = {'w': [0., 500.], 'a': [400., 500.], 'b': [425., 500.], 'e': [999., 500.], 'n': [400., 0.], 's': [425., 999.]}
        graph = build({('w', 'a'): None, ('a', 'b'): None, ('b', 'e'): None, ('a', 'n'): None, ('b', 's'): None}, points)
        dropped = through(graph)
        self.assertEqual(reasons(dropped), {'close_junction_merge'})
        junctions = [n for n, d in graph.degree if d >= 3]
        self.assertEqual(len(junctions), 1)
        assert_consistent(self, graph)

    def test_merge_follows_confident_route_instead_of_pulling_road_ends(self):
        # 三角形路口：A-C 是低置信假连接，A-B、B-C 是真路。落点应为 B，北路沿 A-B 走，不抄 A-C 近路
        points = {'A': [500., 480.], 'B': [502., 500.], 'C': [520., 490.], 'n': [500., 0.], 's': [502., 999.], 'e': [999., 200.]}
        vertices = [{'node_id': k, 'xy': v} for k, v in points.items()]
        edges = [{'u': u, 'v': v, 'polyline_xy': [points[u], points[v]], 'confidence': c}
                 for u, v, c in (('A', 'B', .92), ('B', 'C', 1.), ('A', 'C', .42), ('n', 'A', 1.), ('B', 's', 1.), ('C', 'e', 1.))]
        graph = roads.compress_degree_two(roads.raw_to_graph({'vertices': vertices, 'edges': edges}), 1.)
        roads.merge_close_junctions(graph, 32., 1., 2.)
        hub = next(n for n, d in graph.degree if d == 3)
        self.assertEqual(graph.nodes[hub]['xy'], points['B'])
        north = next(e for e in roads.edge_records(graph) if points['n'] in (e['polyline_xy'][0], e['polyline_xy'][-1]))
        self.assertIn(points['A'], north['polyline_xy'])     # 原路形保留
        self.assertNotIn(points['C'], north['polyline_xy'])  # 不走假连接
        assert_consistent(self, graph)

    def test_short_stub_to_border_is_removed(self):
        points = {'w': [0., 25.], 'j': [400., 25.], 'e': [999., 25.], 'n': [400., 0.]}
        graph = build({('w', 'j'): None, ('j', 'e'): None, ('j', 'n'): None}, points)
        self.assertEqual(reasons(through(graph, close=32., cycle=0.)), {'short_border_stub'})
        self.assertEqual(graph.number_of_edges(), 1)  # 路口退化为度 2，两段并成一条
        assert_consistent(self, graph)

    def test_disabled_distances_skip_their_rule(self):
        points = {'w': [0., 500.], 'a': [400., 500.], 'b': [425., 500.], 'e': [999., 500.], 'n': [400., 0.], 's': [425., 999.]}
        graph = build({('w', 'a'): None, ('a', 'b'): None, ('b', 'e'): None, ('a', 'n'): None, ('b', 's'): None}, points)
        self.assertEqual(through(graph, close=0., cycle=0.), [])


class ContinuousSmoothingTests(unittest.TestCase):
    @staticmethod
    def _heading(polyline: list, from_end: bool) -> float:
        a, b = (polyline[-2], polyline[-1]) if from_end else (polyline[0], polyline[1])
        return math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))

    def test_tangent_is_continuous_where_a_road_passes_a_junction(self):
        # 主路在 j 处折 30°，支路垂直接入：主路配对后切向连续，支路不配对
        points = {'a': [50., 200.], 'j': [150., 200.], 'b': [236.6, 250.], 'n': [150., 100.]}
        graph = build({('a', 'j'): None, ('j', 'b'): None, ('j', 'n'): None}, points)
        stats = road_refine.smooth_roads(graph, 6., 1., (500., 500.))
        self.assertEqual(stats['through_links'], 1)
        edges = {frozenset((e['u'], e['v'])): e for e in roads.edge_records(graph)}
        into = edges[frozenset(('a', 'j'))]['polyline_xy']
        out = edges[frozenset(('j', 'b'))]['polyline_xy']
        into = into if into[-1] == [150., 200.] else into[::-1]
        out = out if out[0] == [150., 200.] else out[::-1]
        self.assertLess(abs(self._heading(into, True)-self._heading(out, False)), 8.)
        assert_consistent(self, graph)

    def test_nodes_stay_in_place_on_a_cross(self):
        points = {'w': [0., 100.], 'e': [200., 100.], 'n': [100., 0.], 's': [100., 200.], 'j': [100., 100.]}
        graph = build({('w', 'j'): None, ('j', 'e'): None, ('n', 'j'): None, ('j', 's'): None}, points)
        stats = road_refine.smooth_roads(graph, 6., 1., (500., 500.))
        self.assertEqual((stats['through_links'], stats['strokes']), (2, 2))
        assert_consistent(self, graph)

    def test_sharp_turn_is_not_treated_as_through_road(self):
        # 三条路都在同一侧呈扇形展开（相互夹角不超过 100°），没有哪两条是顺着穿过路口的
        j = np.array([200., 200.])
        tips = {name: (j+150.*np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])).tolist()
                for name, a in (('p', 0.), ('q', 50.), ('r', 100.))}
        graph = build({('j', name): None for name in tips}, {'j': j.tolist(), **tips})
        self.assertEqual(road_refine.smooth_roads(graph, 6., 1., (500., 500.))['through_links'], 0)

    def test_right_angle_corner_is_preserved(self):
        points = {'a': [100., 100.], 'b': [200., 200.]}
        graph = build({('a', 'b'): [[100., 100.], [200., 100.], [200., 200.]]}, points)
        stats = road_refine.smooth_roads(graph, 12., 1., (500., 500.))
        self.assertEqual(stats['corners'], 1)
        line = np.asarray(roads.edge_records(graph)[0]['polyline_xy'])
        self.assertLess(np.linalg.norm(line-np.array([200., 100.]), axis=1).min(), 3.)  # 没有被圆角切掉

    def test_gentle_curve_is_not_a_corner(self):
        arc = [[200.+100.*math.cos(t), 200.+100.*math.sin(t)] for t in np.linspace(0., math.pi/2, 40)]
        graph = build({('a', 'b'): arc}, {'a': arc[0], 'b': arc[-1]})
        self.assertEqual(road_refine.smooth_roads(graph, 12., 1., (500., 500.))['corners'], 0)

    def test_closed_loop_of_paired_edges_is_handled(self):
        points = {'a': [100., 100.], 'b': [300., 100.]}
        graph = build({('a', 'b'): [[100., 100.], [200., 40.], [300., 100.]], ('b', 'a'): [[300., 100.], [200., 160.], [100., 100.]]}, points)
        stats = road_refine.smooth_roads(graph, 6., 1., (500., 500.))
        self.assertEqual(stats['edges'], 2)
        assert_consistent(self, graph)


class ParallelTests(unittest.TestCase):
    def test_parallel_pair_sharing_both_junctions_loses_one(self):
        # 同一条宽路的两幅：A、B 之间两条相距 12 m 的线
        points = {'A': [0., 0.], 'B': [200., 0.], 'w': [-100., 0.], 'e': [300., 0.]}
        graph = build({('w', 'A'): None, ('B', 'e'): None}, points)
        graph.add_edge('A', 'B', key='p', edge_id='p', u='A', v='B', confidence=.9, length_m=200.,
                       polyline_xy=[[0., 0.], [200., 0.]], source='samroad', semantic_fraction=0.)
        line = [[0., 0.], [20., 12.], [180., 12.], [200., 0.]]
        graph.add_edge('A', 'B', key='q', edge_id='q', u='A', v='B', confidence=.5, length_m=roads.length_px(line),
                       polyline_xy=line, source='samroad', semantic_fraction=0.)
        score(graph)
        dropped = road_refine.merge_parallel_roads(graph, 15., 20., .5, 1.)
        self.assertEqual([d['reason'] for d in dropped], ['parallel_duplicate'])
        self.assertEqual(dropped[0]['edge_id'], 'q')
        self.assertEqual(nx.number_connected_components(graph), 1)
        assert_consistent(self, graph)

    def test_side_road_moves_to_kept_road(self):
        # 短边 c-d 被长边 a-b 覆盖；c、d 上各挂两条支路，去掉 c-d 后支路必须接到 a-b 上
        points = {'a': [0., 0.], 'b': [300., 0.], 'c': [50., 10.], 'd': [250., 10.], 'x': [-50., 200.], 'y': [100., 200.],
                  's': [200., 200.], 't': [350., 200.]}
        graph = build({('a', 'b'): None, ('c', 'd'): None, ('c', 'x'): None, ('c', 'y'): None,
                       ('d', 's'): None, ('d', 't'): None}, points)
        self.assertEqual(nx.number_connected_components(graph), 2)
        dropped = road_refine.merge_parallel_roads(graph, 15., 20., .5, 1.)
        self.assertEqual(len(dropped), 1)
        self.assertAlmostEqual(dropped[0]['length_m'], 200.)
        self.assertEqual(nx.number_connected_components(graph), 1)
        self.assertEqual(sorted(graph.nodes[n]['xy'] for n in graph.nodes if graph.degree(n) == 4),
                         [[50., 0.], [250., 0.]])
        assert_consistent(self, graph)

    def test_continuation_through_junction_is_not_parallel(self):
        # 同一条直路被路口切成两段：靠近路口的部分距离近、方向同，但不是并行
        points = {'a': [0., 0.], 'j': [40., 0.], 'b': [80., 0.], 's': [40., 100.]}
        graph = build({('a', 'j'): None, ('j', 'b'): None, ('j', 's'): None}, points)
        self.assertEqual(road_refine.merge_parallel_roads(graph, 15., 20., .5, 1.), [])

    def test_far_or_crossing_roads_are_kept(self):
        points = {'a': [0., 0.], 'b': [300., 0.], 'c': [0., 40.], 'd': [300., 40.]}
        graph = build({('a', 'b'): None, ('c', 'd'): None}, points)
        self.assertEqual(road_refine.merge_parallel_roads(graph, 15., 20., .5, 1.), [])
        points = {'a': [0., 0.], 'b': [300., 0.], 'c': [150., -100.], 'd': [150., 100.]}
        graph = build({('a', 'b'): None, ('c', 'd'): None}, points)
        self.assertEqual(road_refine.merge_parallel_roads(graph, 15., 20., .5, 1.), [])

    def test_road_that_leaves_the_other_is_kept(self):
        # 前 60% 贴着主路，后面走远：不是重复路，不能整条删
        points = {'a': [0., 0.], 'b': [300., 0.], 'c': [0., 8.], 'd': [200., 150.]}
        graph = build({('a', 'b'): None, ('c', 'd'): [[0., 8.], [150., 8.], [200., 150.]]}, points)
        self.assertEqual(road_refine.merge_parallel_roads(graph, 15., 20., .5, 1.), [])


class ConnectTests(unittest.TestCase):
    def test_dangling_end_within_range_splits_target(self):
        points = {'a': [0., 0.], 'b': [200., 0.], 'c': [100., 2.5], 'd': [100., 120.]}
        graph = build({('a', 'b'): None, ('c', 'd'): None}, points)
        stats = road_refine.connect_near_roads(graph, 3., 1.)
        self.assertEqual(stats['connected'], 1)
        self.assertEqual(stats['gaps_m'], [2.5])
        self.assertEqual(nx.number_connected_components(graph), 1)
        self.assertEqual(graph.number_of_edges(), 3)
        joint = next(n for n in graph.nodes if graph.degree(n) == 3)
        self.assertEqual(graph.nodes[joint]['xy'], [100., 0.])
        assert_consistent(self, graph)

    def test_dangling_end_out_of_range_stays(self):
        points = {'a': [0., 0.], 'b': [200., 0.], 'c': [100., 3.5], 'd': [100., 120.]}
        graph = build({('a', 'b'): None, ('c', 'd'): None}, points)
        self.assertEqual(road_refine.connect_near_roads(graph, 3., 1.)['connected'], 0)
        self.assertEqual(nx.number_connected_components(graph), 2)

    def test_two_tips_join_into_one_road(self):
        points = {'a': [0., 0.], 'b': [100., 0.], 'c': [102., 0.], 'd': [200., 0.]}
        graph = build({('a', 'b'): None, ('c', 'd'): None}, points)
        self.assertEqual(road_refine.connect_near_roads(graph, 3., 1.)['connected'], 1)
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertAlmostEqual(roads.edge_records(graph)[0]['length_m'], 200.)

    def test_stub_next_to_its_own_junction_does_not_make_a_small_loop(self):
        # 8 m 小支路的路端离相邻边不足 3 m：沿路网本来就连着，不再连
        points = {'a': [0., 0.], 'j': [100., 0.], 'b': [200., 0.], 't': [108., 2.], 'n': [100., 100.]}
        graph = build({('a', 'j'): None, ('j', 'b'): None, ('j', 't'): None, ('j', 'n'): None}, points)
        self.assertEqual(road_refine.connect_near_roads(graph, 3., 1.)['connected'], 0)

    def test_crossing_roads_stay_unconnected(self):
        # 两条边几何交叉、都没有悬空端在交叉点附近：不建路口
        points = {'a': [0., 0.], 'b': [200., 0.], 'c': [100., -100.], 'd': [100., 100.]}
        graph = build({('a', 'b'): None, ('c', 'd'): None}, points)
        self.assertEqual(road_refine.connect_near_roads(graph, 3., 1.)['connected'], 0)
        self.assertEqual(graph.number_of_edges(), 2)


class ShortRoadTests(unittest.TestCase):
    def grid(self) -> dict:
        return {'a': [0., 0.], 'j': [100., 0.], 'k': [200., 0.], 'b': [300., 0.], 'n': [100., 100.], 'm': [200., 100.]}

    def test_dangling_stub_shorter_than_limit_is_removed(self):
        points = {**self.grid(), 't': [150., -20.], 'i': [150., 0.]}
        graph = build({('a', 'j'): None, ('j', 'i'): None, ('i', 'k'): None, ('k', 'b'): None, ('j', 'n'): None,
                       ('k', 'm'): None, ('i', 't'): None}, points)
        self.assertEqual(road_refine.remove_short_roads(graph, 10., 32., 1.), [])   # 20 m 不算短
        dropped = road_refine.remove_short_roads(graph, 30., 32., 1.)
        self.assertEqual([(d['reason'], d['length_m']) for d in dropped], [('short_road', 20.)])
        # 支路删除后路口 i 退化为度 2，两侧并回一条边
        self.assertEqual(graph.number_of_edges(), 5)
        self.assertNotIn('i', graph)
        assert_consistent(self, graph)

    def test_short_edge_between_junctions_is_kept(self):
        points = {'a': [0., 0.], 'j': [100., 0.], 'k': [110., 0.], 'b': [210., 0.], 'n': [100., 100.], 'm': [110., 100.]}
        graph = build({('a', 'j'): None, ('j', 'k'): None, ('k', 'b'): None, ('j', 'n'): None, ('k', 'm'): None}, points)
        self.assertEqual(road_refine.remove_short_roads(graph, 30., 32., 1.), [])
        self.assertEqual(nx.number_connected_components(graph), 1)

    def test_small_loop_on_junction_is_removed_by_reach(self):
        circle = [[100.+20*math.sin(t), 20.-20*math.cos(t)] for t in np.linspace(0, 2*math.pi, 40)]
        circle[0] = circle[-1] = [100., 0.]
        points = {'a': [0., 0.], 'j': [100., 0.], 'b': [200., 0.], 'n': [100., -100.]}
        graph = build({('a', 'j'): None, ('j', 'b'): None, ('j', 'n'): None}, points)
        graph.add_edge('j', 'j', key='loop', edge_id='loop', u='j', v='j', confidence=None, source='semantic',
                       semantic_fraction=1., polyline_xy=circle, length_m=roads.length_px(circle))
        # 周长 125 m 比其他路都长，但离路口最远才 40 m
        self.assertEqual(road_refine.remove_short_roads(graph, 10., 32., 1.), [])
        self.assertEqual([d['edge_id'] for d in road_refine.remove_short_roads(graph, 10., 45., 1.)], ['loop'])

    def test_disabled(self):
        points = {**self.grid()}
        graph = build({('a', 'j'): None, ('j', 'k'): None, ('k', 'b'): None}, points)
        self.assertEqual(road_refine.remove_short_roads(graph, 0., 0., 1.), [])


class SmoothTests(unittest.TestCase):
    def test_right_angle_becomes_arc_within_sigma(self):
        line = [[0., 0.], [100., 0.], [100., 100.]]
        smooth = road_refine.smooth_polyline(line, 6.)
        self.assertEqual(smooth[0], line[0])
        self.assertEqual(smooth[-1], line[-1])
        self.assertLess(road_refine.max_turn_deg(smooth), 12)
        self.assertLess(LineString(smooth).hausdorff_distance(LineString(line)), 6.)

    def test_zigzag_chain_becomes_one_curve(self):
        # 15 m 一段的连续折线（圆弧的内接多边形）：平滑后不再有明显折点
        chain = [[200*math.cos(t), 200*math.sin(t)] for t in np.linspace(0, math.pi/2, 22)]
        self.assertGreater(road_refine.max_turn_deg(chain), 4)
        smooth = road_refine.smooth_polyline(chain, 6.)
        self.assertLess(road_refine.max_turn_deg(smooth), 1.5)

    def test_fold_back_spike_is_removed(self):
        line = [[0., 0.], [100., 0.], [100., 15.], [100.5, 0.], [200., 0.]]
        smooth = road_refine.smooth_polyline(line, 6.)
        self.assertLess(max(p[1] for p in smooth), 2.)
        self.assertLess(road_refine.max_turn_deg(smooth), 30)

    def test_long_armed_v_is_a_shape_not_a_spike(self):
        # 两臂 30 m 以上的 V 形只做圆角，不能被当成毛刺拉直
        line = [[0., 0.], [60., 0.], [10., 30.]]
        smooth = road_refine.smooth_polyline(line, 6.)
        self.assertGreater(roads.length_px(smooth), .8*roads.length_px(line))
        self.assertLess(LineString(smooth).hausdorff_distance(LineString(line)), 12.)

    def test_straight_line_stays_straight(self):
        smooth = np.asarray(road_refine.smooth_polyline([[0., 5.], [50., 5.], [100., 5.]], 6.))
        self.assertTrue(np.allclose(smooth[:, 1], 5.))

    def test_loop_keeps_its_size(self):
        square = [[0., 0.], [40., 0.], [40., 40.], [0., 40.], [0., 0.]]
        smooth = road_refine.smooth_polyline(square, 6.)
        self.assertEqual(smooth[0], smooth[-1])
        self.assertGreater(roads.length_px(smooth), 120)

    def test_short_edge_and_image_bounds(self):
        self.assertEqual(road_refine.smooth_polyline([[0., 0.], [1., 0.]], 6.)[-1], [1., 0.])
        smooth = np.asarray(road_refine.smooth_polyline([[0., 0.], [0., 50.], [50., 50.]], 6., (50., 50.)))
        self.assertGreaterEqual(smooth.min(), 0)
        self.assertLessEqual(smooth.max(), 50)

    def test_nodes_do_not_move_and_lengths_update(self):
        points = {'a': [0., 0.], 'j': [100., 0.], 'n': [100., 100.], 'b': [200., 0.]}
        graph = build({('a', 'j'): [[0., 0.], [50., 12.], [100., 0.]], ('j', 'n'): None, ('j', 'b'): None}, points)
        stats = road_refine.smooth_roads(graph, 6., 1., (500., 500.))
        self.assertEqual(stats['edges'], 3)
        self.assertLess(stats['max_turn_after_deg'], stats['max_turn_before_deg'])
        assert_consistent(self, graph)

    def test_disabled(self):
        graph = build({('a', 'b'): [[0., 0.], [50., 30.], [100., 0.]]}, {'a': [0., 0.], 'b': [100., 0.]})
        before = roads.edge_records(graph)[0]['polyline_xy']
        self.assertEqual(road_refine.smooth_roads(graph, 0., 1., (500., 500.))['edges'], 0)
        self.assertEqual(roads.edge_records(graph)[0]['polyline_xy'], before)


if __name__ == '__main__':
    unittest.main()
