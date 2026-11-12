"""概率骨架补漏：只补路网漏掉、概率强、两端能接入的路段。"""
from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from agent.map_preprocess.utils import global_road_graph as roads
from agent.map_preprocess.utils.road_skeleton import fill_from_skeleton

SIZE = (400., 300.)
PLANT, ROAD, BUILDING = 2, 0, 1


def network() -> nx.MultiGraph:
    """1 m/px，上下两条横路，两端用竖路连成一个环。"""
    points = {'a': [20., 40.], 'b': [380., 40.], 'c': [380., 260.], 'd': [20., 260.]}
    edges = [{'u': u, 'v': v, 'polyline_xy': [points[u], points[v]], 'confidence': .9}
             for u, v in (('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a'))]
    return roads.compress_degree_two(roads.raw_to_graph({'vertices': [{'node_id': n, 'xy': xy} for n, xy in points.items()], 'edges': edges}), 1.)


def bar(probability: np.ndarray, top: int = 55, bottom: int = 245) -> None:
    probability[top:bottom, 196:205] = .8


class SkeletonFillTests(unittest.TestCase):
    def test_missing_road_is_connected_to_the_network(self):
        probability = np.zeros((300, 400), np.float32)
        bar(probability)
        graph = network()
        result = fill_from_skeleton(graph, probability, np.full((300, 400), PLANT, np.uint8), SIZE, 1., 40.)
        self.assertEqual(result['added'], 1)
        self.assertEqual(graph.number_of_edges(), 3)  # 环被两个接入点分成两段，再加补上的路
        self.assertTrue(nx.is_connected(graph))
        self.assertEqual(sorted(d for _, d in graph.degree).count(3), 2)

    def test_road_already_in_network_is_not_added(self):
        probability = np.zeros((300, 400), np.float32)
        probability[36:45, 60:340] = .8
        graph = network()
        self.assertEqual(fill_from_skeleton(graph, probability, np.full((300, 400), PLANT, np.uint8), SIZE, 1., 40.)['added'], 0)

    def test_road_with_a_free_end_stays_out(self):
        probability = np.zeros((300, 400), np.float32)
        bar(probability, 55, 170)  # 下端悬空，离路网 90 m
        graph = network()
        self.assertEqual(fill_from_skeleton(graph, probability, np.full((300, 400), PLANT, np.uint8), SIZE, 1., 40.)['added'], 0)

    def test_road_over_building_is_rejected(self):
        probability = np.zeros((300, 400), np.float32)
        bar(probability)
        labels = np.full((300, 400), BUILDING, np.uint8)
        graph = network()
        self.assertEqual(fill_from_skeleton(graph, probability, labels, SIZE, 1., 40.)['added'], 0)


if __name__ == '__main__':
    unittest.main()
