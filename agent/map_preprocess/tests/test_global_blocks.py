"""不加载权重、不启动 VLM 的全局四节点合成回归。"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import math
import tempfile
import unittest
from unittest.mock import patch
import warnings
import numpy as np
from PIL import Image

from agent.map_preprocess import (GlobalPreprocessConfig, run_pipeline, load_manifest,
    crop_block, crop_context, crop_blocks_for_group)
from agent.map_preprocess.config import load_config
from agent.map_preprocess.io import read_json, resolve_path, write_json
from agent.map_preprocess.debug_overview import render_overview
from agent.map_preprocess.nodes.global_pipeline import load_image_and_resolution, clean_road_graph
from agent.map_preprocess.utils.global_blocks import block_positions, blocks_for_edge
from agent.map_preprocess.utils import global_road_graph as roads, road_refine
from agent.map_preprocess.utils.branch_review import EdgeDecision, RuleBasedReviewer, validated_decisions
from percection_tool.dummy import DummyRoadGraphExtractor


def raw_graph(points, pairs, confidences=None):
    return {'vertices': [{'node_id': str(i), 'xy': point} for i, point in enumerate(points)],
            'edges': [{'u': str(a), 'v': str(b), 'polyline_xy': [points[a], points[b]],
                       'confidence': confidences[i] if confidences else 1.} for i, (a,b) in enumerate(pairs)],
            'metadata': {}}


class GlobalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = self.root/'map.png'
        yy, xx = np.indices((700, 900))
        self.rgb = np.stack((xx % 256, yy % 256, (xx+yy) % 256), axis=-1).astype(np.uint8)
        Image.fromarray(self.rgb).save(self.image)
        # 用自定义尺寸覆盖裁剪契约；正式默认尺寸和去重在独立测试中验证。
        self.config = GlobalPreprocessConfig(segmenter='dummy', dedup_enabled=False, block_size_m=100.,
            block_overlap_ratio=.10, output_dir=str(self.root/'out'),
            cache_dir=str(self.root/'cache'), min_component_length_m=0, min_spur_length_m=0,
            parallel_merge_dist_m=0, min_edge_score=0, road_connect_dist_m=0, short_road_max_m=0, small_loop_max_reach_m=0, road_smooth_sigma_m=0)

    def run_map(self, config=None, extractor=None, reviewer=None):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            state = run_pipeline(str(self.image), config or self.config, extractor=extractor, reviewer=reviewer)
        return state, load_manifest(state['manifest_path'])

    def test_config_validation_and_loading(self):
        path = self.root/'settings.yaml'
        path.write_text('workflow: global_blocks\nsegmenter: dummy\ntile_grid: 0\n')
        self.assertIsInstance(load_config(path), GlobalPreprocessConfig)
        for kwargs in ({'block_overlap_ratio': 1}, {'group_overlap_blocks': 3},
                       {'max_blocks_total': -1}, {'context_size_m': 0}, {'tile_grid': -1},
                       {'edge_score_weights': {}}, {'route_polyline': [(0,0)]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.config, **kwargs)

    def test_resolution_and_nonoverlapping_tile_metadata(self):
        config = replace(self.config, zoom=18, center_lat=60)
        state = load_image_and_resolution({'image_path': str(self.image)}, config=config)
        self.assertEqual(state['resolution_source'], 'web_mercator')
        self.assertAlmostEqual(state['resolution_m_per_px'], 156543.03392*.5/2**18)
        coverage = np.zeros((700, 900), dtype=np.uint8)
        for tile in state['tiles']:
            self.assertNotIn('path', tile)
            coverage[tile['y0']:tile['y1'], tile['x0']:tile['x1']] += 1
        self.assertTrue((coverage == 1).all())
        self.assertEqual(load_image_and_resolution({'image_path': str(self.image)},
            config=replace(config, tile_grid=0))['tiles'], [])
        with self.assertWarnsRegex(RuntimeWarning, 'fallback_resolution'):
            result = load_image_and_resolution({'image_path': str(self.image)}, config=self.config)
        self.assertEqual(result['resolution_source'], 'fallback')

    def test_geotiff_priority_original_grid(self):
        import rasterio
        from rasterio.transform import from_origin
        path = self.root/'geo.tif'
        with rasterio.open(path, 'w', driver='GTiff', width=900, height=700, count=3,
                           dtype='uint8', crs='EPSG:32650', transform=from_origin(500000, 4000000, .5, .5)) as target:
            target.write(self.rgb.transpose(2,0,1))
        state = load_image_and_resolution({'image_path': str(path)}, config=replace(self.config, zoom=1, center_lat=0))
        self.assertEqual(state['resolution_m_per_px'], .5)
        self.assertEqual(state['resolution_source'], 'geotiff')
        self.assertEqual((state['image']['width'], state['image']['height']), (900,700))

    def test_full_pipeline_ids_groups_crops_and_placeholders(self):
        config = replace(self.config, semantic_features_enabled=True, elevation_features_enabled=True,
                         vlm_candidate_detection_enabled=True)
        state, manifest = self.run_map(config)
        _, again = self.run_map(config)
        self.assertEqual(manifest['format_version'], 2)
        self.assertEqual([b['block_id'] for b in manifest['blocks']], [b['block_id'] for b in again['blocks']])
        self.assertEqual(len(manifest['edges']), 1)
        self.assertEqual(len(manifest['blocks']), 8)
        self.assertTrue(all(m['coverage_ratio'] == 1 for m in manifest['coverage']))
        self.assertTrue(all(manifest['extensions'][name]['implemented'] is False for name in
                            ('semantic_features', 'elevation_features', 'vlm_candidate_detection')))
        self.assertEqual(manifest['extensions']['block_features'], {'status': 'skipped'})
        group = manifest['groups'][0]
        self.assertEqual(group['block_ids'][-1], manifest['groups'][1]['block_ids'][0])
        self.assertEqual(len(crop_blocks_for_group(manifest, group['group_id'])), 3)
        block = manifest['blocks'][0]
        pixels = crop_block(state['manifest_path'], block['block_id'])
        self.assertEqual(pixels.shape, (100,100,3))
        x0,y0,x1,y1 = block['crop_bbox']
        np.testing.assert_array_equal(pixels, self.rgb[y0:y1,x0:x1])
        self.assertEqual(crop_context(manifest, block['block_id']).shape, (300,300,3))
        directory = resolve_path(state['output_dir'])
        # 任务 ID：时间（到秒）+ 6 位随机后缀，即目录名
        self.assertRegex(state['task_id'], r'^\d{8}_\d{6}_[0-9a-f]{6}$')
        self.assertEqual(directory.name, state['task_id'])
        # 任务目录只有 state/（状态与数据）和 visualization/（图片）两个子目录
        self.assertEqual(sorted(p.name for p in directory.iterdir()), ['state', 'visualization'])
        self.assertEqual(resolve_path(state['manifest_path']), directory/'state'/'manifest.json')
        for name in ('config.json', 'run.log', 'state.json', 'raw_road_graph.json', 'clean_road_graph.json'):
            self.assertTrue((directory/'state'/name).is_file(), name)
        snapshots = [p.name for p in sorted((directory/'state'/'node_states').iterdir())]
        # 编号即执行顺序（本例启用了 3 个占位节点）
        self.assertEqual(snapshots[:4], ['01_load_image_and_resolution.json', '02_extract_road_graph.json',
                                         '03_clean_road_graph.json', '04_build_blocks_and_groups.json'])
        self.assertEqual(snapshots[-1], f'{len(snapshots):02d}_refresh_global_manifest.json')
        self.assertEqual(sorted(manifest['files']['node_states']), [p[:-5] for p in snapshots])
        figures = {p.name for p in (directory/'visualization').iterdir()}
        self.assertTrue({'01_road_cleaning.png', '02_road_topology.png', '04_blocks_final.png'} <= figures)
        self.assertFalse((directory/'tiles').exists())
        self.assertFalse((directory/'debug_blocks').exists())
        self.assertNotIn('blocks', state)
        output = self.root/'overview.png'
        render_overview(state['manifest_path'], output)
        self.assertTrue(output.is_file())

    def test_cache_hit_and_invalidation(self):
        class Counting:
            cache_key = 'counting-v1'
            calls = 0
            def extract(inner, image):
                inner.calls += 1
                return DummyRoadGraphExtractor().extract(image)
        extractor = Counting()
        first, _ = self.run_map(extractor=extractor)
        second, _ = self.run_map(extractor=extractor)
        self.assertFalse(first['road_graph_cache']['hit'])
        self.assertTrue(second['road_graph_cache']['hit'])
        self.assertEqual(extractor.calls, 1)
        self.run_map(replace(self.config, fallback_resolution_m_per_px=2), extractor=extractor)
        self.assertEqual(extractor.calls, 2)
        Image.fromarray(255-self.rgb).save(self.image)
        self.run_map(extractor=extractor)
        self.assertEqual(extractor.calls, 3)

    def test_dummy_file_and_no_cache(self):
        raw = raw_graph([[1,1],[50,1]], [(0,1)])
        path = self.root/'raw.json'
        write_json(path, raw)
        state, manifest = self.run_map(replace(self.config, dummy_graph_path=str(path), cache_enabled=False, tile_grid=0))
        self.assertEqual(len(manifest['blocks']), 1)
        self.assertIsNone(manifest['blocks'][0]['tile_id'])
        self.assertEqual(read_json(state['raw_road_graph_path']), raw)
        for mode in ('edge', 'constant', 'reflect'):
            pixels = crop_block(manifest, 'E0_B0', mode)
            self.assertEqual(pixels.shape, (100,100,3))
            if mode == 'edge':
                np.testing.assert_array_equal(pixels[0,0], self.rgb[0,0])
        with self.assertRaises(ValueError):
            crop_context({**manifest, 'blocks': [{**manifest['blocks'][0], 'context': None}]}, 'E0_B0')

    def test_compression_preserves_polyline_confidence_and_geometric_crossings(self):
        raw = raw_graph([[0,0],[40,0],[100,0],[50,-10],[50,10]], [(0,1),(1,2),(3,4)], [.5,1.,1.])
        graph = roads.compress_degree_two(roads.raw_to_graph(raw), 2.)
        self.assertEqual(graph.number_of_edges(), 2)
        self.assertEqual(graph.number_of_nodes(), 4)
        main = max(roads.edge_records(graph), key=lambda e:e['length_m'])
        self.assertEqual(len(main['polyline_xy']), 3)
        self.assertAlmostEqual(main['confidence'], .8)
        self.assertEqual(main['length_m'], 200.)

    def test_cycle_parallel_and_spur_tools(self):
        raw = raw_graph([[0,0],[100,0],[50,50]], [(0,1),(1,2),(2,0)])
        graph = roads.compress_degree_two(roads.raw_to_graph(raw), 1)
        self.assertEqual(graph.number_of_edges(),1)
        self.assertEqual(len(roads.edge_records(graph)[0]['polyline_xy']),4)
        # 主路有两个端点短枝，平行的重复边保留更高分的一条。
        raw = raw_graph([[0,0],[100,0],[0,1],[100,1],[100,4],[500,0],[501,0]],
                        [(0,1),(2,3),(1,4),(5,6)], [1.,.1,1.,1.])
        graph = roads.compress_degree_two(roads.raw_to_graph(raw), 1)
        # compression 将末端链并入 104m 主路；另建 junction 以独立验证 spur。
        graph.add_node('spur', xy=[0,2])
        graph.add_edge('0','spur', key='spur', u='0',v='spur',edge_id='spur',
                       polyline_xy=[[0,0],[0,2]],length_m=2.,confidence=1.)
        dropped = roads.prune_short_spurs(graph, 3)
        self.assertIn('short_spur', {e['reason'] for e in dropped})
        # 独立平行图，不受上面的 degree-2 压缩形状影响。
        graph = roads.compress_degree_two(roads.raw_to_graph(raw_graph([[0,0],[100,0],[0,1],[100,1]],
            [(0,1),(2,3)], [1.,.1])),1.)
        for *_, edge in graph.edges(data=True):
            edge['edge_score'] = roads.score_edge(edge, graph, self.config)
        self.assertEqual(len(road_refine.merge_parallel_roads(graph,2.,10.,.5,1.)),1)
        self.assertEqual(roads.edge_records(graph)[0]['confidence'],1.)

    def test_topology_refinement_runs_in_cleaning(self):
        # 主路 + 两条支路；重复线 10-11 贴着 0-1；6-7 是短支路；8-9 的路端离主路 2.5 m；1-12-4 带 33° 缓弯（低于拐角阈值，应被平滑）
        raw = raw_graph([[10,100],[210,100],[410,100],[610,100],[210,300],[410,300],[310,100],[310,120],
                         [500,102.5],[500,300],[30,108],[190,108],[240,200]],
            [(0,1),(1,6),(6,2),(2,3),(1,12),(12,4),(2,5),(6,7),(8,9),(10,11)], [1.]*9+[.5])
        state,_ = self.run_map()
        write_json(state['raw_road_graph_path'],raw)
        config = replace(self.config, parallel_merge_dist_m=18, road_connect_dist_m=3, short_road_max_m=30,
                         road_smooth_sigma_m=6)
        cleaned = read_json(clean_road_graph(state, config=config)['clean_road_graph_path'])
        self.assertEqual(sorted(e['reason'] for e in cleaned['dropped_edges']), ['parallel_duplicate','short_road'])
        self.assertEqual(cleaned['filter_stats']['connect_roads']['connected'], 1)
        self.assertEqual(cleaned['filter_stats']['smooth_roads']['edges'], len(cleaned['edges']))
        nodes = {n['node_id']: n['xy'] for n in cleaned['nodes']}
        for edge in cleaned['edges']:
            self.assertEqual(edge['polyline_xy'][0], nodes[edge['u']])
            self.assertEqual(edge['polyline_xy'][-1], nodes[edge['v']])
            self.assertLess(road_refine.max_turn_deg(edge['polyline_xy']), 30)
            self.assertAlmostEqual(edge['length_m'], roads.length_px(edge['polyline_xy']))

    def test_cleaning_reasons_constraints_and_review(self):
        raw = raw_graph([[10,100],[110,100],[210,100],[110,103],[600,500],[601,500],[10,300],[210,300]],
            [(0,1),(1,2),(1,3),(4,5),(6,7)], [1.,1.,1.,1.,.01])
        state,_ = self.run_map()
        write_json(state['raw_road_graph_path'],raw)
        config = replace(self.config, min_component_length_m=5, min_spur_length_m=5,
            min_edge_score=.5, edge_score_weights={'length':0,'confidence':1,'straightness':0,'connectivity':0})
        update = clean_road_graph(state, config=config)
        cleaned = read_json(update['clean_road_graph_path'])
        self.assertEqual(len(cleaned['edges']),2)
        self.assertEqual({e['reason'] for e in cleaned['dropped_edges']}, {'short_component','short_spur','low_edge_score'})
        config = replace(config, min_edge_score=0, route_polyline=((0,100),(300,100)), route_buffer_m=5)
        update = clean_road_graph(state, config=config)
        self.assertIn('external_constraint', {e['reason'] for e in read_json(update['clean_road_graph_path'])['dropped_edges']})

    def test_review_rejection_and_invalid_fallback(self):
        class Reject:
            def review(inner, image, annotated, edges):
                self.assertTrue(resolve_path(annotated).is_file())
                return [EdgeDecision(e['edge_id'],False,'non_road',False,'test') for e in edges]
        _, disabled = self.run_map(reviewer=Reject())
        self.assertTrue(disabled['blocks'])
        _, rejected = self.run_map(replace(self.config, branch_review_enabled=True), reviewer=Reject())
        self.assertFalse(rejected['blocks'])
        self.assertEqual(rejected['dropped_edges'][0]['reason'],'branch_review')
        class Invalid:
            def review(inner, image, annotated, edges):
                return [{'edge_id': edges[0]['edge_id'], 'keep': 'false'}]
        _, kept = self.run_map(replace(self.config, branch_review_enabled=True),reviewer=Invalid())
        self.assertTrue(kept['blocks'])
        with self.assertWarnsRegex(RuntimeWarning,'默认保留'):
            self.assertTrue(validated_decisions([{'edge_id':'x'}],[])[0]['keep'])
        _, passthrough = self.run_map(replace(self.config,branch_review_enabled=True))
        self.assertEqual(disabled['blocks'],passthrough['blocks'])
        _, none = self.run_map(replace(self.config,branch_review_enabled=True,reviewer_type='rule_based',review_top_n_edges=0))
        self.assertFalse(none['blocks'])

    def test_lengths_curves_coverage_overlap_and_sparse_warning(self):
        cases = [(50,1),(100,1),(101,2),(190,2),(280,3),(281,4),(550,6)]
        for length,count in cases:
            with self.subTest(length=length):
                edge = {'edge_id':'0','u':'a','v':'b','length_m':length,'polyline_xy':[[0,0],[length,0]]}
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    blocks, metric = blocks_for_edge(edge, {'width':1000,'height':1000}, [],1.,self.config)
                self.assertEqual(len(blocks),count)
                self.assertAlmostEqual(metric['coverage_ratio'],1)
                if count > 1:
                    self.assertAlmostEqual(metric['min_overlap_ratio'], .1)
                    self.assertAlmostEqual(metric['max_overlap_ratio'], .1)
                    self.assertAlmostEqual(blocks[0]['arc_length_m']+blocks[-1]['arc_length_m'], length)
        edge = {'edge_id':'c','u':'a','v':'b','length_m':600,'polyline_xy':[[10,10],[310,10],[310,310]]}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            blocks, metric = blocks_for_edge(edge,{'width':700,'height':700}, [],1.,self.config)
        self.assertEqual(metric['coverage_ratio'],1)
        self.assertTrue(any(b['tangent_xy'][1] > .9 for b in blocks))
        with self.assertWarnsRegex(RuntimeWarning,'覆盖空洞'):
            positions = block_positions(600,replace(self.config,long_edge_strategy='sparse3'))
        self.assertEqual(len(positions),3)

    def test_junction_adjacency_budget_and_main_trunk(self):
        raw = raw_graph([[300,300],[10,300],[700,300],[300,600],[750,600],[850,600]],
                        [(0,1),(0,2),(0,3),(4,5)], [1.,1.,1.,.1])
        path = self.root/'network.json'
        write_json(path,raw)
        config = replace(self.config,dummy_graph_path=str(path))
        _, full = self.run_map(config)
        incident = []
        for edge in full['edges']:
            if '0' in (edge['u'],edge['v']):
                identifier = edge['block_ids'][0 if edge['u']=='0' else -1]
                incident.append(next(b for b in full['blocks'] if b['block_id']==identifier))
        self.assertEqual(len(incident),3)
        ids={b['block_id'] for b in incident}
        for block in incident:
            self.assertTrue(ids-{block['block_id']} <= set(block['predecessor_ids']+block['successor_ids']))
        _, budget = self.run_map(replace(config,max_blocks_total=8))
        self.assertLessEqual(len(budget['blocks']),8)
        self.assertTrue(any(e['reason']=='max_blocks_total' for e in budget['dropped_edges']))
        for edge in budget['edges']:
            original = next(e for e in full['edges'] if e['edge_id']==edge['edge_id'])
            self.assertEqual(edge['block_ids'],original['block_ids'])
        _, trunk = self.run_map(replace(config,keep_main_trunk_only=True))
        self.assertEqual(len(trunk['edges']),3)
        self.assertEqual(trunk['dropped_edges'][0]['reason'],'outside_main_trunk')
        _, empty = self.run_map(replace(config,max_blocks_total=0))
        self.assertFalse(empty['blocks'])
        self.assertFalse(empty['nodes'])

    def test_route_lonlat_and_roi(self):
        roi = self.root/'roi.png'
        mask = np.zeros((700,900),np.uint8)
        mask[340:360,:]=255
        Image.fromarray(mask).save(roi)
        _, kept = self.run_map(replace(self.config,roi_mask=str(roi),route_buffer_m=0))
        self.assertTrue(kept['edges'])
        mask[:]=0
        Image.fromarray(mask).save(roi)
        _, rejected = self.run_map(replace(self.config,roi_mask=str(roi),route_buffer_m=0))
        self.assertFalse(rejected['edges'])
        with self.assertRaisesRegex(ValueError,'CRS/affine'):
            self.run_map(replace(self.config,route_coordinate_system='lonlat',route_polyline=((110,30),(111,30))))

    def test_boundary_one_pixel_crop_and_cache_recovery(self):
        path = self.root/'boundary.json'
        write_json(path,raw_graph([[900,100],[900,110]],[(0,1)]))
        config = replace(self.config,dummy_graph_path=str(path),block_size_m=1.,context_size_m=None)
        state,manifest = self.run_map(config)
        block = manifest['blocks'][0]
        self.assertEqual(block['global_xy'][0],900)
        pixels = crop_block(manifest,block['block_id'])
        np.testing.assert_array_equal(pixels[0,0],self.rgb[100,899])
        self.assertFalse(crop_block(manifest,block['block_id'],'constant').any())
        cache = resolve_path(state['road_graph_cache']['path'])
        cache.write_text('{incomplete')
        state,_ = self.run_map(config)
        self.assertFalse(state['road_graph_cache']['hit'])


    def test_malformed_review_and_parallel_raw_multiedges(self):
        edge = {'edge_id':'x'}
        for bad in (None, [], [{'edge_id':'x', 'keep':False, 'is_main_road':False,
                              'road_type':[], 'reason':'bad'}]):
            with self.subTest(bad=bad), self.assertWarnsRegex(RuntimeWarning,'默认保留'):
                self.assertTrue(validated_decisions([edge],bad)[0]['keep'])
        raw = raw_graph([[10,10],[100,10]], [(0,1),(0,1)], [.1,.9])
        graph = roads.compress_degree_two(roads.raw_to_graph(raw), 1.)
        self.assertEqual(graph.number_of_edges(),2)
        for *_, data in graph.edges(data=True):
            data['edge_score'] = roads.score_edge(data, graph, self.config)
        road_refine.merge_parallel_roads(graph, 1., 10., .5, 1.)
        self.assertEqual(roads.edge_records(graph)[0]['confidence'],.9)


    def test_empty_graph_and_debug_crop(self):
        path = self.root/'empty.json'
        write_json(path,{'vertices':[],'edges':[],'metadata':{}})
        _, manifest = self.run_map(replace(self.config,dummy_graph_path=str(path)))
        self.assertFalse(manifest['blocks'])
        state, manifest = self.run_map(replace(self.config,debug=True,debug_block_ids=('E0_B0',)))
        self.assertTrue(resolve_path(manifest['files']['debug_blocks']['E0_B0']).is_file())


if __name__ == '__main__':
    unittest.main()
