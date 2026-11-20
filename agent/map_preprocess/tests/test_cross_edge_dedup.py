"""cross_edge_dedup 的覆盖、确定性、拓扑及 Node 4 集成回归。"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from agent.map_preprocess import GlobalPreprocessConfig, load_manifest
from agent.map_preprocess.io import read_json, write_json
from agent.map_preprocess.nodes.global_pipeline import build_blocks_and_groups
from agent.map_preprocess.utils.cross_edge_dedup import (arc_positions, covered_arc_intervals,
    Candidate, cross_edge_dedup, interpolate_positions, merge_intervals, prune_redundant)
from agent.map_preprocess.utils.global_blocks import blocks_for_edge, connect_blocks, group_blocks


def edge(identifier, points, u=None, v=None):
    return {'edge_id': str(identifier), 'u': u or f'{identifier}a', 'v': v or f'{identifier}b',
            'polyline_xy': points, 'length_m': sum(math.dist(a,b) for a,b in zip(points,points[1:])),
            'edge_score': 1., 'confidence': 1.}


class DedupTests(unittest.TestCase):
    def setUp(self):
        # 以下大多是全局贪心的回归；walk 另见 WalkDedupTests。
        self.config = GlobalPreprocessConfig(segmenter='dummy', block_size_m=100., dedup_method='greedy')
        self.image = {'path':'unused.png','width':1000,'height':1000}

    def before(self, edges, config=None):
        cfg = config or self.config
        return [b for e in edges for b in blocks_for_edge(e,self.image,[],1.,cfg,warn=False)[0]]

    def dedup(self, edges, config=None):
        cfg = config or self.config
        return cross_edge_dedup(edges,self.before(edges,cfg),self.image,[],1.,cfg)

    def check_coverage(self, edges, blocks, stats, config=None):
        cfg = config or self.config
        points = np.concatenate([interpolate_positions(e,arc_positions(e['length_m'],cfg.road_sample_spacing_m),1.)
                                 for e in edges]) if edges else np.empty((0,2))
        hits = np.zeros(len(points),int)
        for b in blocks:
            x0,y0,x1,y1 = b['global_bbox']
            hits += ((points[:,0]>=x0-1e-9)&(points[:,0]<=x1+1e-9)&
                     (points[:,1]>=y0-1e-9)&(points[:,1]<=y1+1e-9))
        self.assertAlmostEqual(stats['road_coverage'],np.mean(hits>0) if len(hits) else 1.)
        self.assertAlmostEqual(stats['mean_coverage_count'],np.mean(hits) if len(hits) else 0.)
        self.assertGreaterEqual(stats['road_coverage'],cfg.min_road_coverage)
        self.assertEqual(stats['blocks_after'],len(blocks))
        for e in edges:
            segments = [(a,b) for block in blocks for identifier,a,b in block['covered_segments'] if identifier==e['edge_id']]
            for a,b in segments:
                self.assertGreaterEqual(a,0.)
                self.assertLessEqual(b,e['length_m']+1e-9)
            # 以更细的独立几何采样核对区间并集，避免只检查统计量自身。
            arcs = np.linspace(0,e['length_m'],max(2,int(e['length_m']*8)+1))
            xy = interpolate_positions(e,arcs,1.)
            actual = np.zeros(len(arcs),bool)
            for block in blocks:
                x0,y0,x1,y1 = block['global_bbox']
                actual |= ((xy[:,0]>=x0-1e-9)&(xy[:,0]<=x1+1e-9)&(xy[:,1]>=y0-1e-9)&(xy[:,1]<=y1+1e-9))
            restored = np.zeros(len(arcs),bool)
            for a,b in segments:
                restored |= (arcs>=a-1e-9)&(arcs<=b+1e-9)
            np.testing.assert_array_equal(actual,restored)

    def test_crossroads_reduction_topology_and_stability(self):
        roads = [edge('L',[[50,200],[200,200]],v='J'),edge('R',[[200,200],[350,200]],u='J'),
                 edge('U',[[200,50],[200,200]],v='J'),edge('D',[[200,200],[200,350]],u='J')]
        cfg = replace(self.config,min_road_coverage=1.)
        blocks,coverage,stats = self.dedup(roads,cfg)
        self.assertLess(len(blocks),stats['blocks_before'])
        self.check_coverage(roads,blocks,stats,cfg)
        repeated,_,again = self.dedup(list(reversed(roads)),cfg)
        self.assertEqual(blocks,repeated)
        self.assertEqual(stats,again)
        self.assertTrue(all(m['coverage_ratio']==1 for m in coverage))
        by_id = {b['block_id']:b for b in blocks}
        for b in blocks:
            self.assertNotIn(b['block_id'],b['predecessor_ids']+b['successor_ids'])
            for identifier in b['successor_ids']+b['predecessor_ids']:
                self.assertIn(identifier,by_id)
                self.assertIn(b['block_id'],by_id[identifier]['predecessor_ids']+by_id[identifier]['successor_ids'])
        self.assertTrue(any(len({s[0] for s in b['covered_segments']})>1 for b in blocks))
        groups = group_blocks(roads,blocks,cfg)
        self.assertEqual({g['edge_id'] for g in groups},{b['edge_id'] for b in blocks})
        self.assertTrue(all(by_id[identifier]['edge_id']==g['edge_id'] for g in groups for identifier in g['block_ids']))

    def test_greedy_pruning_requires_high_area_overlap(self):
        for separation, expected in ((25., 1), (25.01, 0)):
            with self.subTest(separation=separation):
                candidates = [Candidate('a', 0., 0., 0., np.array([0])),
                              Candidate('b', 0., separation, 0., np.array([0]))]
                counts = np.array([2])
                kept, pruned = prune_redundant(candidates, counts, self.config, 100.)
                self.assertEqual(pruned, expected)
                self.assertEqual(len(kept), 2-expected)
                self.assertEqual(counts[0], 2-expected)

    def test_parallel_streets_reuse_blocks_and_exact_intervals(self):
        roads = [edge('a',[[0,100],[300,100]]),edge('b',[[0,130],[300,130]])]
        blocks,coverage,stats = self.dedup(roads)
        self.assertLess(len(blocks),stats['blocks_before'])
        self.check_coverage(roads,blocks,stats)
        # 平局按坐标确定中心归属，另一街道可完全由邻边块覆盖，不能丢道路。
        self.assertTrue(all(b['edge_id']=='a' for b in blocks))
        self.assertEqual(roads[1]['block_ids'],[])
        self.assertEqual(set(roads[1]['covering_block_ids']),{b['block_id'] for b in blocks})
        for e in roads:
            intervals = merge_intervals([(a,b) for block in blocks for ident,a,b in block['covered_segments'] if ident==e['edge_id']])
            self.assertEqual(intervals,[(0.,300.)])
        groups = group_blocks(roads,blocks,self.config)
        self.assertTrue(all(g['edge_id']=='a' for g in groups))
        self.assertEqual(set(identifier for g in groups for identifier in g['block_ids']),{b['block_id'] for b in blocks})
        self.assertTrue(all(m['coverage_ratio']==1 for m in coverage))

    def test_exact_clipping_reentrant_curves_and_tangencies(self):
        road = edge('loop',[[0,0],[100,0],[100,100],[0,100],[0,0]])
        self.assertEqual(covered_arc_intervals(road,[-10,-10,10,10],1.),[(0.,10.),(390.,400.)])
        self.assertEqual(covered_arc_intervals(edge('x',[[0,0],[100,100]]),[50,0,60,50],1.),
                         [(math.sqrt(2)*50,math.sqrt(2)*50)])
        cfg = replace(self.config,block_size_m=80.,min_road_coverage=1.)
        blocks,_,stats = self.dedup([road],cfg)
        self.check_coverage([road],blocks,stats,cfg)
        self.assertTrue(all(b['size_px']==80 for b in blocks))

    def test_endpoints_empty_unreachable_and_validation(self):
        np.testing.assert_array_equal(arc_positions(25,10),[0,10,20,25])
        np.testing.assert_array_equal(arc_positions(20,10),[0,10,20])
        blocks,coverage,stats = self.dedup([])
        self.assertEqual((blocks,coverage),([],[]))
        self.assertEqual(stats['road_coverage'],1.)
        with self.assertWarnsRegex(RuntimeWarning,'无新增覆盖'):
            blocks,_,stats = self.dedup([edge('a',[[0,0],[300,0]])],replace(self.config,dedup_candidate_spacing_m=300))
        self.assertLess(stats['road_coverage'],.995)
        for change in ({'dedup_enabled':1},{'dedup_candidate_spacing_m':0},{'road_sample_spacing_m':float('nan')},
                       {'min_road_coverage':0},{'min_road_coverage':1.1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.config,**change)
        self.assertTrue(GlobalPreprocessConfig().dedup_enabled)

    def test_greedy_matches_full_recompute_tie_breaking(self):
        roads = [edge('a',[[0,0],[200,0]]),edge('b',[[0,30],[200,30]])]
        cfg = replace(self.config,min_road_coverage=1.)
        blocks,_,_ = self.dedup(roads,cfg)
        points = np.concatenate([interpolate_positions(e,arc_positions(e['length_m'],1.),1.) for e in roads])
        candidates=[]
        for e in roads:
            arcs=arc_positions(e['length_m'],10.)
            for arc,(x,y) in zip(arcs,interpolate_positions(e,arcs,1.)):
                mask=(abs(points[:,0]-x)<=50)&(abs(points[:,1]-y)<=50)
                candidates.append(((x,y,e['edge_id'],arc),mask))
        chosen,covered=[],np.zeros(len(points),bool)
        while not covered.all():
            tie,mask=min(candidates,key=lambda candidate:(-np.count_nonzero(candidate[1]&~covered),candidate[0]))
            self.assertTrue(np.any(mask&~covered))
            covered|=mask
            chosen.append(tie)
        self.assertEqual({(b['global_xy'][0],b['global_xy'][1],b['edge_id'],b['arc_length_m']) for b in blocks},set(chosen))

    def test_disconnected_crossings_do_not_create_topology_links(self):
        # 交叉处由一个共享块覆盖，但其它两个块不能凭二维交叉直接相连。
        roads=[edge('h',[[0,100],[200,100]]),edge('v',[[100,0],[100,200]])]
        blocks,_,_ = self.dedup(roads,replace(self.config,block_size_m=40.,min_road_coverage=1.))
        for a in blocks:
            for b in blocks:
                if a['block_id']==b['block_id']:
                    continue
                shared={s[0] for s in a['covered_segments']}&{s[0] for s in b['covered_segments']}
                if not shared:
                    self.assertNotIn(b['block_id'],a['predecessor_ids']+a['successor_ids'])

    def test_full_graph_defaults_config_loading_and_crop(self):
        from PIL import Image
        from agent.map_preprocess import load_config, run_pipeline, crop_block
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            image=root/'streets.png'
            Image.new('RGB',(512,512),(20,30,40)).save(image)
            raw=root/'raw.json'
            write_json(raw,{'vertices':[{'node_id':str(i),'xy':p} for i,p in enumerate(
                [[0,100],[300,100],[0,115],[300,115]])],
                'edges':[{'u':'0','v':'1','confidence':1.},{'u':'2','v':'3','confidence':1.}], 'metadata':{}})
            path=root/'config.json'
            write_json(path,{'workflow':'global_blocks','segmenter':'dummy','dummy_graph_path':str(raw),
                'output_dir':str(root/'out'),'cache_dir':str(root/'cache'),'parallel_merge_dist_m':0})
            cfg=load_config(path)
            self.assertTrue(cfg.dedup_enabled)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                state=run_pipeline(str(image),cfg)
            manifest=load_manifest(state['manifest_path'])
            self.assertLess(manifest['dedup_stats']['blocks_after'],manifest['dedup_stats']['blocks_before'])
            self.assertGreaterEqual(manifest['dedup_stats']['road_coverage'],.995)
            self.assertEqual(crop_block(manifest,manifest['blocks'][0]['block_id']).shape,(75,75,3))
            self.assertTrue(all('covered_segments' in block for block in manifest['blocks']))

    def test_metre_pixel_conversion(self):
        road=edge('a',[[0,100],[150,100]])
        road['length_m']=300.
        cfg=replace(self.config,min_road_coverage=1.)
        before=blocks_for_edge(road,self.image,[],2.,cfg,warn=False)[0]
        blocks,coverage,stats=cross_edge_dedup([road],before,self.image,[],2.,cfg)
        self.assertEqual(stats['road_coverage'],1.)
        self.assertEqual(coverage[0]['covered_intervals_m'],[(0.,300.)])
        for block in blocks:
            self.assertEqual(block['size_px'],50)
            self.assertAlmostEqual(block['arc_length_px'],block['arc_length_m']/2)
            self.assertAlmostEqual(block['global_xy'][0],block['arc_length_px'])


    def test_node_integration_disabled_equivalence_and_budget(self):
        renders=patch('agent.map_preprocess.nodes.global_pipeline.render_block_layout')
        render=renders.start()
        self.addCleanup(renders.stop)
        topology=patch('agent.map_preprocess.nodes.global_pipeline.render_road_topology')
        topology.start()
        self.addCleanup(topology.stop)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            roads=[edge('a',[[0,100],[300,100]]),edge('b',[[0,130],[300,130]])]
            clean=root/'clean.json'
            write_json(clean,{'edges':roads,'nodes':[{'node_id':e[k],'xy':e['polyline_xy'][0 if k=='u' else -1]}
                for e in roads for k in ('u','v')],'dropped_edges':[],'filter_stats':{},
                'branch_review_path':None,'review_decisions':[]})
            raw=root/'raw.json'
            write_json(raw,{'metadata':{}})
            state={'task_id':'test','image_path':'unused.png','image':self.image,'tiles':[],
                'resolution_m_per_px':1.,'resolution_source':'fallback','input_sha256':'test',
                'output_dir':str(root),'clean_road_graph_path':str(clean),'raw_road_graph_path':str(raw),
                'clean_graph_overview_path':'unused.png','road_graph_cache':{'path':str(raw)}}
            # 关闭时不执行去重，逐字段核对原布块/关联/分组结果。
            cfg=replace(self.config,dedup_enabled=False)
            expected_edges=read_json(clean)['edges']
            expected_blocks=self.before(expected_edges,cfg)
            connect_blocks(expected_edges,expected_blocks)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                with patch('agent.map_preprocess.nodes.global_pipeline.cross_edge_dedup',side_effect=AssertionError('disabled')):
                    result=build_blocks_and_groups(state,config=cfg)
            old=load_manifest(result['manifest_path'])
            self.assertEqual(old['blocks'],expected_blocks)
            self.assertEqual(old['groups'],group_blocks(expected_edges,expected_blocks,cfg))
            self.assertNotIn('dedup_stats',old)
            result=build_blocks_and_groups(state,config=self.config)
            manifest=load_manifest(result['manifest_path'])
            self.assertLess(len(manifest['blocks']),len(old['blocks']))
            self.assertGreaterEqual(manifest['dedup_stats']['road_coverage'],.995)
            self.assertTrue(all('covered_segments' in b for b in manifest['blocks']))
            # 预算应在去重后判定：不能按旧8块预先丢边。
            result=build_blocks_and_groups(state,config=replace(self.config,max_blocks_total=len(manifest['blocks'])))
            self.assertEqual(len(load_manifest(result['manifest_path'])['edges']),2)
            with self.assertWarnsRegex(RuntimeWarning,'块预算触发'):
                result=build_blocks_and_groups(state,config=replace(self.config,max_blocks_total=0))
            empty=load_manifest(result['manifest_path'])
            self.assertEqual(empty['blocks'],[])
            self.assertEqual(len(empty['dropped_edges']),2)
            self.assertEqual(empty['dedup_stats']['road_sample_count'],0)
        # 开启去重时输出第一次布块与最终布块两张对照图
        names={Path(call.args[1]).name for call in render.call_args_list}
        self.assertTrue({'03_blocks_initial.png','04_blocks_final.png'} <= names)


class WalkDedupTests(unittest.TestCase):
    def setUp(self):
        self.config = GlobalPreprocessConfig(segmenter='dummy')
        self.image = {'path':'unused.png','width':2000,'height':2000}

    def run_walk(self, edges, config=None):
        cfg = config or self.config
        before = [b for e in edges for b in blocks_for_edge(e,self.image,[],1.,cfg,warn=False)[0]]
        return cross_edge_dedup(edges,before,self.image,[],1.,cfg)

    def test_straight_road_spacing(self):
        for axis, endpoint in enumerate(([400, 500], [100, 800])):
            with self.subTest(axis=axis):
                blocks, coverage, stats = self.run_walk([edge('a', [[100,500], endpoint])])
                self.assertEqual([b['arc_length_m'] for b in blocks], [28.125, 76.875, 125.625, 174.375, 223.125, 271.875])
                for first, second in zip(blocks, blocks[1:]):
                    self.assertAlmostEqual(first['global_bbox'][axis+2]-second['global_bbox'][axis], 26.25)
                self.assertEqual(stats['pruned_redundant_blocks'], 0)
                self.assertEqual(coverage[0]['covered_intervals_m'], [(0., 300.)])

    def test_short_tail_kept(self):
        blocks, coverage, stats = self.run_walk([edge('a', [[0,500], [76,500]])])
        self.assertEqual(len(blocks), 2)
        self.assertAlmostEqual(blocks[1]['arc_length_m']-blocks[0]['arc_length_m'], 48.75)
        self.assertEqual(coverage[0]['covered_intervals_m'], [(0., 76.)])
        self.assertEqual(stats['road_coverage'], 1.)

    def test_high_overlap_does_not_erase_subsample_road_tail(self):
        roads = [edge('long', [[0, 500], [75, 500]]),
                 edge('short', [[37.6, 498], [37.6, 537.6], [37.6, 536.7]])]
        # 短边框重叠至少 75%，但含一小段采样点没有命中的折返道路。
        blocks, coverage, _ = self.run_walk(roads, replace(self.config, road_sample_spacing_m=10.))
        self.assertEqual(len(blocks), 2)
        self.assertAlmostEqual(coverage[1]['coverage_ratio'], 1.)

    def test_low_overlap_keeps_different_context_on_covered_road(self):
        roads = [edge('long', [[0, 500], [75, 500]]), edge('short', [[.1, 530], [.2, 530]])]
        blocks, _, stats = self.run_walk(roads)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(stats['pruned_redundant_blocks'], 0)

    def test_junction_keeps_short_branch_below_overlap_threshold(self):
        main = edge('main', [[0, 500], [75, 500]])
        branch = edge('branch', [[75, 500], [75, 530]])
        branch['u'] = main['v']
        blocks, coverage, stats = self.run_walk([main, branch])
        self.assertEqual({b['edge_id'] for b in blocks}, {'main', 'branch'})
        self.assertEqual(stats['pruned_redundant_blocks'], 0)
        self.assertTrue(all(record['coverage_ratio'] == 1. for record in coverage))

    def test_prune_requires_at_least_75_percent_area_overlap(self):
        for separation, expected in ((18.75, 1), (18.76, 0)):
            with self.subTest(separation=separation):
                roads = [edge('long', [[0, 500], [75, 500]]),
                         edge('short', [[30, 500+separation], [45, 500+separation]])]
                blocks, coverage, stats = self.run_walk(roads)
                self.assertEqual(len(blocks), 2-expected)
                self.assertEqual(stats['pruned_redundant_blocks'], expected)
                self.assertTrue(all(record['coverage_ratio'] == 1. for record in coverage))

    def test_long_edge_first_and_cross_edge_covered(self):
        roads=[edge('long',[[0,500],[360,500]]),edge('short',[[180,500],[180,520]])]
        blocks, coverage, stats = self.run_walk(roads)
        self.assertEqual({b['edge_id'] for b in blocks},{'long'})
        self.assertEqual(stats['pruned_redundant_blocks'], 1)
        self.assertEqual(coverage[1]['covered_intervals_m'], [(0., 20.)])
        self.assertEqual(roads[1]['covering_block_ids'], ['Elong_B3'])

    def test_deterministic_and_validation(self):
        roads=[edge('a',[[0,500],[300,500]]),edge('b',[[150,300],[150,700]])]
        first,_,_=self.run_walk(roads)
        again,_,_=self.run_walk(list(reversed(roads)))
        self.assertEqual([b['global_xy'] for b in first],[b['global_xy'] for b in again])
        for change in ({'dedup_method':'random'},{'dedup_min_gain_m':-1}):
            with self.assertRaises(ValueError):
                replace(self.config,**change)


if __name__ == '__main__':
    unittest.main()
