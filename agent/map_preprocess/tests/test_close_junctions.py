"""过近路口合并：两端都是路口且短于阈值的边，两端合并为一个路口。"""
from __future__ import annotations

import unittest

from agent.map_preprocess.utils import global_road_graph as roads


def h_graph(gap: float):
    """两个路口 A(0,0)、B(gap,0) 由短边相连，各伸出上下两条 100 m 的路（1 m/px）。"""
    vertices = [{'node_id': 'A', 'xy': [0., 0.]}, {'node_id': 'B', 'xy': [gap, 0.]}]
    edges = [{'u': 'A', 'v': 'B', 'polyline_xy': [[0., 0.], [gap, 0.]], 'confidence': .9}]
    for name, x, y in (('a1', 0., -100.), ('a2', 0., 100.), ('b1', gap, -100.), ('b2', gap, 100.)):
        vertices.append({'node_id': name, 'xy': [x, y]})
        anchor = 'A' if name[0] == 'a' else 'B'
        edges.append({'u': anchor, 'v': name, 'polyline_xy': [[x, 0.], [x, y]], 'confidence': .9,
                      'source': 'semantic' if name == 'b2' else 'samroad'})
    return roads.compress_degree_two(roads.raw_to_graph({'vertices': vertices, 'edges': edges}), 1.)


class CloseJunctionTests(unittest.TestCase):
    def test_close_junctions_are_merged(self):
        graph = h_graph(20.)
        self.assertEqual(graph.number_of_edges(), 5)
        dropped = roads.merge_close_junctions(graph, 32., 1.)
        self.assertEqual([d['reason'] for d in dropped], ['close_junction_merge'])
        edges = roads.edge_records(graph)
        self.assertEqual(len(edges), 4)
        hubs = [n for n in graph.nodes if graph.degree(n) == 4]
        self.assertEqual(len(hubs), 1)
        # 沿用其中一个原始路口的坐标，不取均值
        self.assertIn(graph.nodes[hubs[0]]['xy'], ([0., 0.], [20., 0.]))
        for edge in edges:  # 折线端点与节点坐标一致
            self.assertEqual(edge['polyline_xy'][0], graph.nodes[edge['u']]['xy'])
            self.assertEqual(edge['polyline_xy'][-1], graph.nodes[edge['v']]['xy'])
        self.assertEqual(sorted(e['source'] for e in edges), ['samroad', 'samroad', 'samroad', 'semantic'])

    def test_merged_node_stays_on_the_higher_degree_junction(self):
        # A 是四岔路口，B 是三岔路口：合并点必须落在 A 上，不能落到两者中间
        vertices = [{'node_id': 'A', 'xy': [0., 0.]}, {'node_id': 'B', 'xy': [10., 8.]}]
        edges = [{'u': 'A', 'v': 'B', 'polyline_xy': [[0., 0.], [10., 8.]]}]
        for name, anchor, xy in (('w', 'A', [-100., 0.]), ('n', 'A', [0., -100.]), ('s', 'A', [0., 100.]),
                                 ('e', 'B', [110., 8.]), ('f', 'B', [10., 108.])):
            vertices.append({'node_id': name, 'xy': xy})
            edges.append({'u': anchor, 'v': name, 'polyline_xy': [([0., 0.] if anchor == 'A' else [10., 8.]), xy]})
        graph = roads.compress_degree_two(roads.raw_to_graph({'vertices': vertices, 'edges': edges}), 1.)
        roads.merge_close_junctions(graph, 16., 1.)
        hub = next(n for n in graph.nodes if graph.degree(n) == 5)
        self.assertEqual(graph.nodes[hub]['xy'], [0., 0.])

    def test_far_junctions_are_kept(self):
        graph = h_graph(50.)
        self.assertEqual(roads.merge_close_junctions(graph, 32., 1.), [])
        self.assertEqual(graph.number_of_edges(), 5)

    def test_chain_of_close_junctions_collapses(self):
        # 三个路口间隔 15 m 排成一列：全部并成一个
        vertices = [{'node_id': n, 'xy': [x, 0.]} for n, x in (('A', 0.), ('B', 15.), ('C', 30.))]
        edges = [{'u': 'A', 'v': 'B', 'polyline_xy': [[0., 0.], [15., 0.]]},
                 {'u': 'B', 'v': 'C', 'polyline_xy': [[15., 0.], [30., 0.]]}]
        for anchor, x in (('A', 0.), ('B', 15.), ('C', 30.)):
            for sign in (-1, 1):
                name = f'{anchor}{sign}'
                vertices.append({'node_id': name, 'xy': [x, 80.*sign]})
                edges.append({'u': anchor, 'v': name, 'polyline_xy': [[x, 0.], [x, 80.*sign]]})
        edges.append({'u': 'A', 'v': 'Aw', 'polyline_xy': [[0., 0.], [-80., 0.]]})
        vertices.append({'node_id': 'Aw', 'xy': [-80., 0.]})
        graph = roads.compress_degree_two(roads.raw_to_graph({'vertices': vertices, 'edges': edges}), 1.)
        roads.merge_close_junctions(graph, 32., 1.)
        self.assertEqual(max(d for _, d in graph.degree), 7)
        self.assertEqual(graph.number_of_edges(), 7)

    def test_long_chain_is_not_collapsed_into_one_point(self):
        # 4 个路口间隔 15 m（首尾 45 m > 32 m）：不能压成一个点，合并组直径 <= 32 m
        names = ('A', 'B', 'C', 'D')
        vertices = [{'node_id': n, 'xy': [15.*i, 0.]} for i, n in enumerate(names)]
        edges = [{'u': a, 'v': b, 'polyline_xy': [[15.*i, 0.], [15.*(i+1), 0.]]} for i, (a, b) in enumerate(zip(names, names[1:]))]
        for i, anchor in enumerate(names):
            for sign in (-1, 1):
                name = f'{anchor}{sign}'
                vertices.append({'node_id': name, 'xy': [15.*i, 80.*sign]})
                edges.append({'u': anchor, 'v': name, 'polyline_xy': [[15.*i, 0.], [15.*i, 80.*sign]]})
        graph = roads.compress_degree_two(roads.raw_to_graph({'vertices': vertices, 'edges': edges}), 1.)
        roads.merge_close_junctions(graph, 32., 1.)
        hubs = [graph.nodes[n]['xy'][0] for n in graph.nodes if graph.degree(n) >= 3]
        self.assertGreaterEqual(len(hubs), 2)
        self.assertGreaterEqual(max(hubs)-min(hubs), 15)


if __name__ == '__main__':
    unittest.main()


class PruneRedundantTests(unittest.TestCase):
    def test_fully_covered_block_is_removed_and_coverage_kept(self):
        import numpy as np
        from agent.map_preprocess.global_config import GlobalPreprocessConfig
        from agent.map_preprocess.utils.cross_edge_dedup import Candidate, prune_redundant
        config = GlobalPreprocessConfig(segmenter='dummy', road_sample_spacing_m=1., dedup_prune_max_loss_m=5.)
        # 100 个采样点：A 覆盖 0-59，B 覆盖 40-99，C 覆盖 30-69（被 A、B 完全盖住），D 覆盖 95-99 + 独占 3 点不存在
        blocks = [Candidate('0', 0., 0., 0., np.arange(0, 60)), Candidate('0', 1., 0., 0., np.arange(40, 100)),
                  Candidate('0', 2., 0., 0., np.arange(30, 70))]
        counts = np.zeros(100, np.int64)
        for block in blocks:
            counts[block.covered] += 1
        kept, removed = prune_redundant(blocks, counts, config, config.block_size_m)
        self.assertEqual(removed, 1)
        self.assertEqual([b.arc_m for b in kept], [0., 1.])
        self.assertTrue((counts > 0).all())
        # 独占覆盖 >= 5 m 的块不删
        self.assertEqual(prune_redundant(kept, counts, config, config.block_size_m)[1], 0)
