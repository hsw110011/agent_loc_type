"""Node 8 退化区域空间融合离线回归：纯几何，不加载模型、不调用 VLM。"""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from PIL import Image

from agent.map_preprocess.global_config import GlobalPreprocessConfig, NODE_ORDER
from agent.map_preprocess.io import read_json, write_json
from agent.map_preprocess.nodes import degradation_fusion as df
from agent.map_preprocess.utils.patch_fusion import fuse_patches, has_area_overlap

BACKGROUND = (90, 90, 90)


def patch(patch_id: str, bbox: list, kind: str = 'degradation') -> dict:
    return {'patch_id': patch_id, 'bbox': bbox, 'degradation_type': kind}


def groups(result: dict) -> list[list[str]]:
    return [region['source_patch_ids'] for region in result['regions']]


class RequiredCases(unittest.TestCase):
    def test_case1_area_overlap_merges(self):
        result = fuse_patches([patch('A', [0, 0, 100, 100]), patch('B', [50, 50, 150, 150])])
        self.assertEqual(groups(result), [['A', 'B']])
        self.assertEqual(result['regions'][0]['bbox'], [0, 0, 150, 150])

    def test_case2_shared_edge_does_not_merge(self):
        result = fuse_patches([patch('A', [0, 0, 100, 100]), patch('B', [100, 0, 200, 100])])
        self.assertEqual(groups(result), [['A'], ['B']])

    def test_case3_shared_vertex_does_not_merge(self):
        result = fuse_patches([patch('A', [0, 0, 100, 100]), patch('B', [100, 100, 200, 200])])
        self.assertEqual(groups(result), [['A'], ['B']])

    def test_case4_chain_is_one_component(self):
        result = fuse_patches([patch('A', [0, 0, 100, 100]), patch('B', [50, 0, 150, 100]), patch('C', [120, 0, 220, 100])])
        self.assertEqual(groups(result), [['A', 'B', 'C']])
        self.assertEqual(result['regions'][0]['bbox'], [0, 0, 220, 100])

    def test_case5_separated_patches_stay_apart(self):
        result = fuse_patches([patch('A', [0, 0, 100, 100]), patch('B', [300, 300, 400, 400])])
        self.assertEqual(groups(result), [['A'], ['B']])

    def test_case6_different_types_stay_apart_by_default(self):
        patches = [patch('A', [0, 0, 100, 100]), patch('B', [50, 50, 150, 150], 'longitudinal_degradation')]
        result = fuse_patches(patches)
        self.assertEqual(groups(result), [['A'], ['B']])
        self.assertEqual([r['degradation_type'] for r in result['regions']], ['degradation', 'longitudinal_degradation'])

    def test_case7_merged_bbox_does_not_capture_third_patch(self):
        a, b, c = [0, 0, 100, 100], [50, 50, 150, 150], [120, 0, 150, 40]
        self.assertFalse(has_area_overlap(a, c) or has_area_overlap(b, c))
        self.assertTrue(has_area_overlap([0, 0, 150, 150], c))  # 融合后的 BBox 确实覆盖 C，但不得作为依据
        result = fuse_patches([patch('A', a), patch('B', b), patch('C', c)])
        self.assertEqual(groups(result), [['A', 'B'], ['C']])
        self.assertEqual(result['regions'][0]['bbox'], [0, 0, 150, 150])

    def test_case8_single_patch_is_kept(self):
        result = fuse_patches([patch('A', [10, 20, 30, 50])])
        self.assertEqual(len(result['regions']), 1)
        region = result['regions'][0]
        self.assertEqual((region['merged_id'], region['source_patch_ids'], region['source_patch_count']), ('Merged_1', ['A'], 1))
        self.assertEqual(region['bbox'], [10, 20, 30, 50])
        self.assertEqual((region['union_area_px2'], region['bbox_area_px2'], region['source_area_px2']), (600, 600, 600))

    def test_case9_empty_input(self):
        self.assertEqual(fuse_patches([]), {'regions': [], 'excluded': [], 'rejected': []})


class GeometryTests(unittest.TestCase):
    def test_float_touching_is_not_overlap(self):
        self.assertFalse(has_area_overlap([0., 0., 1.5, 1.], [1.5, 0., 3., 1.]))
        self.assertTrue(has_area_overlap([0., 0., 1.5, 1.], [1.25, 0., 3., 1.]))

    def test_areas_before_and_after_fusion(self):
        region = fuse_patches([patch('A', [0, 0, 100, 100]), patch('B', [50, 50, 150, 150])])['regions'][0]
        self.assertEqual(region['source_area_px2'], 20000)
        self.assertEqual(region['union_area_px2'], 17500)
        self.assertEqual(region['bbox_area_px2'], 22500)

    def test_l_shape_keeps_true_outline(self):
        region = fuse_patches([patch('A', [0, 0, 100, 50]), patch('B', [0, 0, 50, 100])])['regions'][0]
        self.assertEqual(region['union_polygon']['type'], 'Polygon')
        outline = region['union_polygon']['coordinates'][0]
        self.assertEqual(len(outline), 7)  # 6 个顶点 + 闭合点
        self.assertEqual(outline[0], outline[-1])
        self.assertNotIn([100, 100], outline)  # 外接矩形的角不属于真实几何
        self.assertEqual(region['union_area_px2'], 7500)
        self.assertEqual(region['bbox'], [0, 0, 100, 100])

    def test_ring_of_patches_keeps_hole(self):
        bars = [[0, 0, 100, 20], [80, 0, 100, 100], [0, 80, 100, 100], [0, 0, 20, 100]]
        result = fuse_patches([patch(f'P{i}', bar) for i, bar in enumerate(bars)])
        self.assertEqual(len(result['regions']), 1)
        region = result['regions'][0]
        self.assertEqual(len(region['union_polygon']['coordinates']), 2)
        self.assertEqual(region['union_area_px2'], 6400)
        self.assertEqual(region['bbox_area_px2'], 10000)

    def test_collinear_vertices_are_removed_from_chain(self):
        region = fuse_patches([patch('A', [0, 0, 100, 100]), patch('B', [50, 0, 150, 100]), patch('C', [120, 0, 220, 100])])['regions'][0]
        self.assertEqual(len(region['union_polygon']['coordinates'][0]), 5)
        self.assertEqual(region['union_area_px2'], 22000)

    def test_identical_patches_merge(self):
        result = fuse_patches([patch('A', [0, 0, 10, 10]), patch('B', [0, 0, 10, 10])])
        self.assertEqual(groups(result), [['A', 'B']])
        self.assertEqual(result['regions'][0]['union_area_px2'], 100)

    def test_non_integer_coordinates_are_preserved(self):
        region = fuse_patches([patch('A', [0.5, 0, 10, 10])])['regions'][0]
        self.assertEqual(region['bbox'], [0.5, 0, 10, 10])


class InputHandlingTests(unittest.TestCase):
    def test_only_degraded_types_enter(self):
        result = fuse_patches([patch('A', [0, 0, 10, 10]), patch('B', [0, 0, 10, 10], 'non_degraded'), patch('C', [0, 0, 10, 10], 'unknown')])
        self.assertEqual(groups(result), [['A']])
        self.assertEqual([e['patch_id'] for e in result['excluded']], ['B', 'C'])

    def test_invalid_input_is_recorded_not_raised(self):
        bad = {'not_four': [0, 0, 1], 'text': [0, 0, 'x', 1], 'nan': [0, 0, float('nan'), 1], 'inf': [0, 0, float('inf'), 1],
               'flag': [False, 0, True, 1], 'inverted': [10, 0, 0, 10], 'zero_width': [5, 0, 5, 10], 'none': None}
        items = [patch(name, value) for name, value in bad.items()]
        result = fuse_patches([*items, patch('ok', [0, 0, 10, 10]), {'bbox': [0, 0, 1, 1]}, 'junk'])
        self.assertEqual(groups(result), [['ok']])
        self.assertEqual({r.get('patch_id') for r in result['rejected']}, {*bad, None})
        self.assertEqual(len(result['rejected']), len(bad)+2)

    def test_duplicate_patch_id_is_rejected(self):
        result = fuse_patches([patch('A', [0, 0, 10, 10]), patch('A', [500, 500, 510, 510])])
        self.assertEqual(groups(result), [['A']])
        self.assertEqual(result['regions'][0]['bbox'], [0, 0, 10, 10])
        self.assertEqual(result['rejected'][0]['reason'], 'duplicate_patch_id')

    def test_cross_type_extension(self):
        patches = [patch('A', [0, 0, 100, 100]), patch('B', [50, 50, 150, 150], 'longitudinal_degradation')]
        region = fuse_patches(patches, cross_type=True)['regions'][0]
        self.assertEqual(region['source_patch_ids'], ['A', 'B'])
        self.assertEqual(region['degradation_type'], 'mixed')
        self.assertEqual(region['source_degradation_types'], ['degradation', 'longitudinal_degradation'])

    def test_allowed_types_are_configurable(self):
        result = fuse_patches([patch('A', [0, 0, 10, 10]), patch('B', [0, 0, 10, 10], 'longitudinal_degradation')], allowed_types=['degradation'])
        self.assertEqual(groups(result), [['A']])

    def test_output_is_independent_of_input_order(self):
        rng = random.Random(7)
        patches = [patch(f'P{i}', [x, y, x+rng.randint(20, 90), y+rng.randint(20, 90)], rng.choice(['degradation', 'longitudinal_degradation']))
                   for i in range(60) for x, y in [(rng.randint(0, 900), rng.randint(0, 900))]]
        expected = fuse_patches(patches)
        for _ in range(5):
            rng.shuffle(patches)
            self.assertEqual(fuse_patches(patches), expected)

    def test_components_match_brute_force(self):
        rng = random.Random(11)
        patches = [patch(f'P{i}', [x, y, x+rng.randint(1, 80), y+rng.randint(1, 80)]) for i in range(80)
                   for x, y in [(rng.randint(0, 500), rng.randint(0, 500))]]
        regions = fuse_patches(patches)['regions']
        component = {pid: index for index, region in enumerate(regions) for pid in region['source_patch_ids']}
        self.assertEqual(len(component), len(patches))  # 所有 Patch 都保留在且只在一个区域
        for region in regions:
            members = [p for p in patches if p['patch_id'] in region['source_patch_ids']]
            reached, frontier = {members[0]['patch_id']}, [members[0]]
            while frontier:  # 每个分组都必须能只经由原始 Patch 的重叠互相到达
                current = frontier.pop()
                for other in members:
                    if other['patch_id'] not in reached and has_area_overlap(current['bbox'], other['bbox']):
                        reached.add(other['patch_id'])
                        frontier.append(other)
            self.assertEqual(reached, set(region['source_patch_ids']))
        for a in patches:
            for b in patches:
                if has_area_overlap(a['bbox'], b['bbox']):
                    self.assertEqual(component[a['patch_id']], component[b['patch_id']])

    def test_input_is_not_modified(self):
        patches = [patch('B', [50, 50, 150, 150]), patch('A', [0, 0, 100, 100])]
        snapshot = deepcopy(patches)
        result = fuse_patches(patches)
        self.assertEqual(patches, snapshot)
        self.assertEqual({p['patch_id']: p['bbox'] for p in result['regions'][0]['source_patches']},
                         {'A': [0, 0, 100, 100], 'B': [50, 50, 150, 150]})

    def test_result_is_json_serializable(self):
        result = fuse_patches([patch('A', [0, 0, 100, 100]), patch('B', [50, 50, 150, 150])])
        self.assertEqual(json.loads(json.dumps(result)), result)


def extent(block_id: str, xy: list, size, label: str = '退化', kind: str = 'semantic', **overrides) -> dict:
    tiers = [{'size_m': 64, 'state': '退化', 'degradation_type': '退化'}]
    if size != 64:
        tiers.append({'size_m': size, 'state': '退化', 'degradation_type': label})
    else:
        tiers[0]['degradation_type'] = label
    return {'block_id': block_id, 'global_xy': xy, 'review_type': kind, 'status': 'completed', 'skip_reason': None,
            'max_size_m': size, 'tiers': tiers, **overrides}


class AdapterTests(unittest.TestCase):
    def test_window_matches_node7_tier_window(self):
        record = extent('E0_B1', [1570.5, 9903.5], 64)
        self.assertEqual(df.patch_from_extent(record, 0.05000043545170622)['bbox'], [930, 9264, 2210, 10544])

    def test_label_comes_from_largest_confirmed_tier(self):
        result = df.patch_from_extent(extent('E0_B0', [500, 500], 80, '纵向退化'), .5)
        self.assertEqual((result['degradation_type'], result['size_m'], result['patch_id']), ('longitudinal_degradation', 80, 'E0_B0_semantic'))

    def test_unrecognized_label_is_unknown(self):
        self.assertEqual(df.patch_from_extent(extent('E0_B0', [500, 500], 64, '??'), .5)['degradation_type'], 'unknown')

    def test_unconfirmed_branch_is_not_an_exclusion(self):
        skipped = extent('E1_B0', [500, 500], None, status='skipped', skip_reason='node6_not_confirmed')
        patches, excluded, not_degraded = df.patches_from_extents([extent('E0_B0', [500, 500], 64), skipped], .5)
        self.assertEqual([p['patch_id'] for p in patches], ['E0_B0_semantic'])
        self.assertEqual((excluded, not_degraded), ([], [{'patch_id': 'E1_B0_semantic', 'reason': 'node6_not_confirmed'}]))

    def test_incomplete_base_keeps_node6_coverage(self):
        incomplete = {**extent('E2_B0', [500, 500], 64), 'stop_reason': 'base_incomplete',
                      'node6': {'coverage': {'image_global_bbox': [470, 480, 564, 564]}}}
        patches, excluded, _ = df.patches_from_extents([incomplete], .5)
        self.assertEqual((patches[0]['bbox'], patches[0]['size_m'], excluded), ([470, 480, 564, 564], 64, []))


class NodeTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        records = Path(folder.name)/'task'/'state'
        records.mkdir(parents=True)
        self.manifest_path = records/'manifest.json'
        self.extents_path = records/'degradation_extents.json'
        self.image_path = Path(folder.name)/'image.png'
        Image.new('RGB', (2000, 2000), BACKGROUND).save(self.image_path)
        self.manifest = {'format_version': 2, 'workflow': 'global_blocks', 'task_id': 't', 'image': {'path': str(self.image_path), 'width': 2000, 'height': 2000},
                         'resolution_m_per_px': 1., 'resolution_source': 'geotiff', 'input_sha256': 'x', 'tiles': [], 'nodes': [], 'edges': [],
                         'blocks': [], 'groups': [], 'dropped_edges': [], 'filter_stats': {}, 'coverage': [], 'review_decisions': [],
                         'extensions': {}, 'config': {}, 'files': {'degradation_extents': str(self.extents_path)}}
        self.config = GlobalPreprocessConfig(segmenter='dummy', block_size_m=64, block_features_enabled=True, vlm_review_enabled=True,
                                             vlm_expansion_enabled=True, degradation_fusion_enabled=True)

    def write(self, extents: list[dict]) -> None:
        write_json(self.manifest_path, self.manifest)
        write_json(self.extents_path, {'resolution_m_per_px': 1., 'extents': extents})

    def test_outputs_and_manifest(self):
        before = [extent('E0_B0', [100, 100], 64), extent('E0_B1', [140, 100], 64), extent('E5_B0', [1000, 1000], 64),
                  extent('E6_B0', [50, 50], None, status='skipped', skip_reason='node6_not_confirmed')]
        self.write(before)
        summary = df.run_degradation_fusion(self.manifest_path, self.config)
        self.assertEqual((summary['patch_count'], summary['region_count'], summary['merged_region_count'], summary['excluded_count'],
                          summary['not_degraded_count']), (3, 2, 1, 0, 1))
        document = read_json(self.manifest_path.parent/'degradation_regions.json')
        self.assertEqual(document['coordinate_system'], 'global_pixel')
        self.assertEqual([r['source_patch_ids'] for r in document['regions']], [['E0_B0_semantic', 'E0_B1_semantic'], ['E5_B0_semantic']])
        self.assertEqual(document['regions'][0]['bbox'], [68, 68, 172, 132])
        self.assertEqual(read_json(self.extents_path)['extents'], before)
        manifest = read_json(self.manifest_path)
        self.assertEqual(manifest['extensions']['degradation_fusion'], summary)
        self.assertIn('degradation_regions', manifest['files'])
        self.assertEqual(manifest['files']['degradation_fusion_overview'], summary['overview'])
        self.assertTrue((self.manifest_path.parents[1]/'visualization'/'13_degradation_fusion.png').is_file())

    def test_overview_fills_only_true_union(self):
        output = self.manifest_path.parent/'overview.png'
        bars = [[300, 0, 400, 20], [380, 0, 400, 100], [300, 80, 400, 100], [300, 0, 320, 100]]
        patches = [patch('A', [100, 100, 200, 200]), patch('B', [150, 150, 250, 250]), *(patch(f'R{i}', bar) for i, bar in enumerate(bars))]
        df.render_overview(self.manifest, self.config, fuse_patches(patches)['regions'], output)
        with Image.open(output) as image:
            pixels = image.convert('RGB')
            self.assertNotEqual(pixels.getpixel((130, 135)), BACKGROUND)  # 并集内
            self.assertEqual(pixels.getpixel((230, 120)), BACKGROUND)     # 外接矩形内、并集外
            self.assertNotEqual(pixels.getpixel((310, 50)), BACKGROUND)   # 环形区域的实体部分
            self.assertEqual(pixels.getpixel((350, 50)), BACKGROUND)      # 环形区域的洞

    def test_empty_extents(self):
        self.write([])
        self.assertEqual(df.run_degradation_fusion(self.manifest_path, self.config)['region_count'], 0)
        self.assertEqual(read_json(self.manifest_path.parent/'degradation_regions.json')['regions'], [])
        self.assertTrue((self.manifest_path.parents[1]/'visualization'/'13_degradation_fusion.png').is_file())

    def test_invalid_extent_is_logged_and_recorded(self):
        self.write([extent('E0_B0', [100, 100], 0), extent('E0_B1', [140, 100], 64)])
        with self.assertLogs('map_preprocess.degradation_fusion', level='WARNING'):
            summary = df.run_degradation_fusion(self.manifest_path, self.config)
        self.assertEqual((summary['region_count'], summary['rejected_count']), (1, 1))

    def test_requires_node7_output(self):
        self.manifest['files'] = {}
        write_json(self.manifest_path, self.manifest)
        with self.assertRaisesRegex(ValueError, 'Node 7'):
            df.run_degradation_fusion(self.manifest_path, self.config)

    def test_node_skips_when_node7_did_not_run(self):
        for status in ('skipped', 'dry_run'):
            update = df.fuse_degraded_patches({'vlm_expansion': {'status': status}}, config=self.config)
            self.assertEqual(update, {'degradation_fusion': {'status': 'skipped', 'reason': f'node7_{status}'}})

    def test_node_returns_state_increment(self):
        self.write([extent('E0_B0', [100, 100], 64)])
        update = df.fuse_degraded_patches({'manifest_path': str(self.manifest_path), 'task_id': 't', 'vlm_expansion': {'status': 'completed'}}, config=self.config)
        self.assertEqual(list(update), ['degradation_fusion'])
        self.assertEqual(update['degradation_fusion']['status'], 'completed')

    def test_graph_order_and_run_until(self):
        from agent.map_preprocess.graph import build_graph
        nodes = list(build_graph(config=self.config).get_graph().nodes)
        self.assertLess(nodes.index('vlm_expand_blocks'), nodes.index('fuse_degraded_patches'))
        self.assertLess(nodes.index('fuse_degraded_patches'), nodes.index('refresh_global_manifest'))
        self.assertIn('fuse_degraded_patches', NODE_ORDER)
        stopped = list(build_graph(config=replace(self.config, run_until='vlm_expand_blocks')).get_graph().nodes)
        self.assertNotIn('fuse_degraded_patches', stopped)
        self.assertIn('vlm_expand_blocks', stopped)

    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, 'vlm_expansion_enabled'):
            GlobalPreprocessConfig(segmenter='dummy', degradation_fusion_enabled=True)
        for bad in ((), 'degradation', ('degradations',), ('degradation', 'unknown'), ('non_degraded',), (['degradation'],)):
            with self.assertRaisesRegex(ValueError, 'degradation_fusion_types', msg=repr(bad)):
                GlobalPreprocessConfig(segmenter='dummy', degradation_fusion_types=bad)
        self.assertEqual(GlobalPreprocessConfig(segmenter='dummy', degradation_fusion_types=['longitudinal_degradation']).degradation_fusion_types,
                         ['longitudinal_degradation'])  # YAML 读入的是列表
        with self.assertRaisesRegex(ValueError, 'degradation_fusion_cross_type'):
            GlobalPreprocessConfig(segmenter='dummy', degradation_fusion_cross_type='yes')


if __name__ == '__main__':
    unittest.main()
