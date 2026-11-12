"""Node 5 离线回归：假语义工具 + 合成 DSM，不加载权重、不需要 GPU。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
from PIL import Image

from agent.map_preprocess import GlobalPreprocessConfig, run_pipeline, load_manifest
from agent.map_preprocess.io import resolve_path
from agent.map_preprocess.nodes.block_features import _degradation, run_block_features
from agent.map_preprocess.utils import block_features as bf
from agent.map_preprocess.utils import vlm_review_inputs as vi
from agent.map_preprocess.utils.dsm import inspect_dsm, read_block_dsm

ROAD_RGB, BUILDING_RGB, PLANT_RGB = (128, 128, 128), (200, 30, 30), (30, 200, 30)
GROUND_M = 100.


def write_raster(path: Path, array: np.ndarray, crs: str | None = None, transform=None, nodata=None) -> Path:
    import rasterio
    bands = array[None] if array.ndim == 2 else np.moveaxis(array, 2, 0)
    profile = {'driver': 'GTiff', 'width': bands.shape[2], 'height': bands.shape[1], 'count': bands.shape[0],
               'dtype': str(bands.dtype)}
    if crs is not None:
        profile.update(crs=crs, transform=transform)
    if nodata is not None:
        profile['nodata'] = nodata
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with rasterio.open(path, 'w', **profile) as target:
            target.write(bands)
    return path


class FakeSemantic:
    """按颜色分类；黑色背景给低置信度，应被记为 other。"""
    cache_key = 'fake-semantic-v1'

    def __init__(self, events: list):
        self.events = events

    def predict(self, image, valid_mask=None):
        self.events.append('semantic')
        mask = np.zeros(image.shape[:2], np.uint8)
        confidence = np.full(image.shape[:2], .3, np.float32)
        for index, color in enumerate((ROAD_RGB, BUILDING_RGB, PLANT_RGB)):
            hit = (image == color).all(axis=2)
            mask[hit], confidence[hit] = index, .9
        if valid_mask is not None:
            mask[~valid_mask], confidence[~valid_mask] = 255, 0
        return {'mask': mask, 'confidence': confidence, 'metadata': {}}

    def close(self):
        self.events.append('semantic_close')


class NoKeySemantic(FakeSemantic):
    cache_key = None


class BlockFeatureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        width, height = 900, 700
        rgb = np.zeros((height, width, 3), np.uint8)
        rgb[height//2-6:height//2+6] = ROAD_RGB
        for x in range(0, width, 100):
            rgb[height//2-60:height//2-20, x:x+80] = BUILDING_RGB
        rgb[height//2+15:height//2+45, :width//2] = PLANT_RGB
        self.image = self.root/'map.png'
        Image.fromarray(rgb).save(self.image)
        # 与 RGB 同尺寸、无地理信息的 DSM（按像素对齐）：路面 100 m，建筑 +8 m，植被 +2 m，其余 +0.2 m。
        dsm = np.full((height, width), GROUND_M+.2, np.float32)
        dsm[(rgb == ROAD_RGB).all(axis=2)] = GROUND_M
        dsm[(rgb == BUILDING_RGB).all(axis=2)] = GROUND_M+8
        dsm[(rgb == PLANT_RGB).all(axis=2)] = GROUND_M+2
        self.dsm = write_raster(self.root/'dsm.tif', dsm)
        self.config = GlobalPreprocessConfig(segmenter='dummy', output_dir=str(self.root/'out'),
            cache_dir=str(self.root/'cache'), feature_cache_dir=str(self.root/'feature_cache'),
            min_component_length_m=0, min_spur_length_m=0, parallel_merge_dist_m=0, min_edge_score=0,
            block_features_enabled=True)
        self.events = []

    def run_map(self, config=None, dsm=True):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            state = run_pipeline(str(self.image), config or self.config, dsm_path=str(self.dsm) if dsm else None,
                                 semantic_segmenter=FakeSemantic(self.events))
        return state, load_manifest(state['manifest_path'])

    def test_pipeline_with_dsm_judges_height_only(self):
        state, manifest = self.run_map()
        self.assertGreater(len(manifest['blocks']), 1)
        self.assertEqual(manifest['image']['dsm']['alignment'], 'pixel')
        self.assertEqual(self.events[-1], 'semantic_close')
        expected = {'class_pct', 'height_band_pct', 'height_stats', 'degradation'}   # 只输出指标和判定，不做可靠性标记
        for block in manifest['blocks']:
            features = block['features']
            self.assertEqual(set(features), expected)
            self.assertAlmostEqual(sum(features['class_pct'].values()), 100, places=3)
            self.assertAlmostEqual(sum(features['height_band_pct'].values()), 100, places=3)
            self.assertEqual(list(features['height_band_pct']), ['0-0.5m', '0.5-2m', '>2m'])
            stats = features['height_stats']
            # 0 点 = 块中心（道路上的布块点）路面 DSM
            self.assertEqual(stats['base_ref'], 'center_road')
            self.assertAlmostEqual(stats['base_dsm_m'], GROUND_M, places=3)
            self.assertEqual(stats['below_base_pct'], 0.)
            degradation = features['degradation']
            # 有 DSM：不判语义，只判高度
            self.assertIsNone(degradation['semantic_degraded'])
            self.assertEqual(degradation['height_degraded'],
                             features['height_band_pct']['0-0.5m'] > 90 or features['height_stats']['below_base_pct'] > 70)
            self.assertEqual(degradation['degraded'], degradation['height_degraded'])
            x0, y0, x1, y1 = block['valid_global_bbox']
            for name in ('rgb', 'semantic_color', 'semantic_overlay', 'dsm_raw', 'height', 'height_bands'):
                with Image.open(resolve_path(block['feature_vis'][name])) as image:
                    self.assertEqual(image.size, (x1-x0, y1-y0))
        # 合成图每块有超过 10% 的 +8 m 建筑，且没有低于路面的像元。
        self.assertFalse(any(b['features']['degradation']['height_degraded'] for b in manifest['blocks']))
        from agent.map_preprocess.feature_contact_sheet import main as sheet
        self.assertEqual(sheet(['--manifest', state['manifest_path'], '--count', '3', '--per-sheet', '2']), 0)
        task = resolve_path(state['output_dir'])
        pages = list((task/'visualization'/'08_contact_sheets').glob('seed0_*.png'))
        self.assertEqual(len(pages), 2)
        with Image.open(pages[0]) as image:
            self.assertEqual(image.width, 5*(400+10)+10)
        self.assertTrue((task/'visualization'/'05_blocks_degraded.png').is_file())
        self.assertEqual(list((task/'visualization'/'06_degraded_block_sheets').glob('degraded_*.png')), [])
        from agent.map_preprocess.export_feature_table import main as export
        from agent.map_preprocess.feature_definitions import METRICS, flatten
        self.assertEqual(export(['--manifest', state['manifest_path']]), 0)
        table = task/'state'/'block_features_table'
        self.assertTrue((table/'features.csv').is_file() and (table/'features.md').is_file())
        row = flatten(manifest['blocks'][0])
        self.assertEqual(set(row), {'block_id', 'edge_id', *(m['column'] for m in METRICS)})
        self.assertTrue(all(m['name_zh'] and m['name_en'] and m['definition_zh'] and m['definition_en'] for m in METRICS))
        summary = manifest['extensions']['block_features']
        self.assertEqual((summary['status'], summary['height_source']), ('completed', 'dsm'))
        self.assertTrue(resolve_path(manifest['files']['block_features']).is_file())
        self.assertEqual(state['block_features']['block_count'], len(manifest['blocks']))

    def test_rgb_only_judges_semantic_only(self):
        state, manifest = self.run_map(dsm=False)
        self.assertNotIn('dsm', manifest['image'])
        for block in manifest['blocks']:
            features = block['features']
            self.assertIsNone(features['height_band_pct'])
            self.assertIsNone(features['height_stats'])
            degradation = features['degradation']
            self.assertIsNone(degradation['height_degraded'])
            self.assertEqual(degradation['degraded'], bool(degradation['semantic_degraded']))
            self.assertEqual(set(block['feature_vis']), {'rgb', 'semantic_color', 'semantic_overlay'})
        summary = manifest['extensions']['block_features']
        self.assertIsNone(summary['height_source'])
        self.assertEqual(summary['height_degraded_blocks'], [])
        self.assertEqual(len(summary['degraded_sheets']), -(-len(summary['degraded_blocks'])//5))
        for page in summary['degraded_sheets']:
            with Image.open(resolve_path(page)) as image:
                self.assertEqual(image.width, 3*(400+10)+10)
        self.assertNotIn('height_unreliable_blocks', summary)
        from agent.map_preprocess.feature_contact_sheet import main as sheet
        self.assertEqual(sheet(['--manifest', state['manifest_path'], '--count', '2']), 0)
        from agent.map_preprocess.export_feature_table import main as export
        self.assertEqual(export(['--manifest', state['manifest_path']]), 0)
        # 之后补 DSM：单独重跑 Node 5 即可加上高程，语义命中缓存
        self.events.clear()
        summary = run_block_features(state['manifest_path'], self.config, dsm_path=self.dsm,
                                     semantic_segmenter=FakeSemantic(self.events))
        self.assertEqual((self.events, summary['height_source']), ([], 'dsm'))
        again = load_manifest(state['manifest_path'])
        self.assertTrue(all(b['features']['height_band_pct'] for b in again['blocks']))
        self.assertIn('height', again['blocks'][0]['feature_vis'])

    def test_missing_road_reference_is_reported_without_review(self):
        from agent.map_preprocess.nodes.vlm_review import review_types

        with Image.open(self.image) as image:
            rgb = np.array(image)
        rgb[(rgb == ROAD_RGB).all(axis=2)] = BUILDING_RGB
        Image.fromarray(rgb).save(self.image)
        state, manifest = self.run_map(replace(self.config, feature_vis_enabled=False))
        self.assertTrue(manifest['blocks'])
        self.assertEqual(state['block_features']['missing_road_reference_block_ids'],
                         sorted(b['block_id'] for b in manifest['blocks']))
        for block in manifest['blocks']:
            self.assertIsNone(block['features']['height_stats'])
            self.assertEqual(review_types(block), [])
            self.assertIn('road_reference_missing', state['block_features']['unavailable_review_data'][block['block_id']])

    def test_cache_skips_model_on_rerun(self):
        state, manifest = self.run_map()
        first = {b['block_id']: b['features'] for b in manifest['blocks']}
        self.events.clear()
        summary = run_block_features(state['manifest_path'], self.config, semantic_segmenter=FakeSemantic(self.events))
        self.assertEqual(self.events, [])
        self.assertEqual((summary['semantic_source'], summary['semantic_cache_hit']), ('whole_image', True))
        again = load_manifest(state['manifest_path'])
        self.assertEqual({b['block_id']: b['features'] for b in again['blocks']}, first)

    def test_tool_without_cache_key_is_not_cached(self):
        state, manifest = self.run_map(replace(self.config, block_features_enabled=False, feature_vis_enabled=False))
        self.assertEqual(manifest['extensions']['block_features'], {'status': 'skipped'})
        for _ in range(2):
            self.events.clear()
            run_block_features(state['manifest_path'], self.config, semantic_segmenter=NoKeySemantic(self.events))
            # 没有缓存身份时不缓存；每次整图推理一次（窗口大于图像），不再逐块推理
            self.assertEqual(self.events.count('semantic'), 1)

    def test_block_semantic_is_crop_of_whole_image(self):
        from agent.map_preprocess.nodes.block_features import semantic_crop
        labels = np.arange(100*120, dtype=np.int64).reshape(100, 120) % 251
        labels = labels.astype(np.uint8)
        confidence = np.linspace(0, 1, 100*120, dtype=np.float32).reshape(100, 120)
        same = semantic_crop(labels, confidence, (120, 100), [10, 20, 50, 70])
        self.assertTrue((same['mask'] == labels[20:70, 10:50]).all())
        self.assertTrue((same['confidence'] == confidence[20:70, 10:50]).all())
        # 工作网格为原图一半：最近邻放大回块的原图像素尺寸
        half = semantic_crop(labels[::2, ::2], confidence[::2, ::2], (120, 100), [20, 40, 100, 90])
        self.assertEqual(half['mask'].shape, (50, 80))
        self.assertEqual(half['mask'][0, 0], labels[::2, ::2][20, 10])

    def test_block_beyond_image_is_cropped_not_padded(self):
        from agent.map_preprocess.nodes.block_features import _read_block
        _, manifest = self.run_map(replace(self.config, block_features_enabled=False))
        block = dict(manifest['blocks'][0], crop_bbox=[-30, -20, 70, 80], valid_global_bbox=[0, 0, 70, 80])
        rgb, valid = _read_block(manifest, block)
        self.assertEqual(rgb.shape, (80, 70, 3))
        self.assertTrue(valid.all())

    def test_dsm_must_align_with_rgb(self):
        small = write_raster(self.root/'small.tif', np.zeros((10, 10), np.float32))
        with self.assertRaisesRegex(ValueError, '尺寸不同'):
            inspect_dsm(small, self.image, 1.)
        geo = write_raster(self.root/'geo.tif', np.zeros((10, 10), np.float32), 'EPSG:32633',
                           __import__('affine').Affine(1, 0, 500000, 0, -1, 5000000))
        with self.assertRaisesRegex(ValueError, '同时有地理信息'):
            inspect_dsm(geo, self.image, 1.)

    def test_config_validation(self):
        for kwargs in ({'feature_min_class_confidence': 2}, {'feature_height_flat_min_pct': 100}, {'feature_height_below_max_pct': 0},
                       {'feature_semantic_max_road_pct': 0}, {'block_features_enabled': 1}, {'feature_vis_max_side': 8},
                       {'feature_dsm_height_bands_m': (3., .5)}, {'feature_dsm_height_bands_m': ()}, {'feature_dsm_height_bands_m': (0., 3.)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.config, **kwargs)
        # 可靠性标记的配置已删除
        for name in ('feature_max_road_dsm_iqr_m', 'feature_max_dsm_nodata_pct', 'feature_max_other_pct', 'feature_min_mean_confidence', 'feature_min_class_pct'):
            self.assertFalse(hasattr(self.config, name))


class GeoDsmTests(unittest.TestCase):
    """RGB 与 DSM 分辨率、CRS 不同时，按地理坐标把 DSM 重投影到块网格。"""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        from affine import Affine
        # RGB：UTM 33N，0.1 m/px，400×300 px（40 m × 30 m）
        self.rgb_transform = Affine(.1, 0, 500000, 0, -.1, 5000030)
        self.rgb = write_raster(self.root/'rgb.tif', np.zeros((300, 400, 3), np.uint8), 'EPSG:32633', self.rgb_transform)
        # DSM：同 CRS，0.5 m/px，范围更大；高程 = 东向坐标（米）的线性函数，便于核对位置。
        x = 499990+.5*(np.arange(120)+.5)
        dsm = np.repeat((x-499990)[None], 100, 0).astype(np.float32)
        dsm[:10, :10] = -9999
        self.dsm = write_raster(self.root/'dsm.tif', dsm, 'EPSG:32633', Affine(.5, 0, 499990, 0, -.5, 5000040), nodata=-9999)

    def test_geo_alignment_and_block_read(self):
        info = inspect_dsm(self.dsm, self.rgb, .1)
        self.assertEqual((info['alignment'], info['block_grid_factor']), ('geo', 5))
        self.assertAlmostEqual(info['m_per_px'], .5, places=4)
        self.assertAlmostEqual(info['overlap_ratio'], 1.)
        # 块 [100, 50, 200, 150]（RGB 像素）→ 东向 500010~500020 m，块网格 20×20（每格 0.5 m）
        block = read_block_dsm(info, [100, 50, 200, 150])
        self.assertEqual(block.shape, (20, 20))
        expected = (500010+.5*(np.arange(20)+.5))-499990
        np.testing.assert_allclose(block[10], expected, atol=1e-3)

    def test_different_crs_is_reprojected(self):
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        with rasterio.open(self.dsm) as source:
            transform, width, height = calculate_default_transform(source.crs, 'EPSG:4326', source.width,
                                                                   source.height, *source.bounds)
            data = np.full((height, width), -9999, np.float32)
            reproject(rasterio.band(source, 1), data, dst_transform=transform, dst_crs='EPSG:4326',
                      resampling=Resampling.bilinear, dst_nodata=-9999)
        wgs = write_raster(self.root/'dsm_wgs84.tif', data, 'EPSG:4326', transform, nodata=-9999)
        info = inspect_dsm(wgs, self.rgb, .1)
        self.assertEqual(info['alignment'], 'geo')
        block = read_block_dsm(info, [100, 50, 200, 150])
        expected = (500010+.1*(np.arange(100)[::info['block_grid_factor']]+info['block_grid_factor']/2))-499990
        np.testing.assert_allclose(np.nanmedian(block, axis=0), expected[:block.shape[1]], atol=.6)

    def test_no_overlap_fails(self):
        from affine import Affine
        far = write_raster(self.root/'far.tif', np.zeros((10, 10), np.float32), 'EPSG:32633',
                           Affine(.5, 0, 600000, 0, -.5, 5000040))
        with self.assertRaisesRegex(ValueError, '没有重叠'):
            inspect_dsm(far, self.rgb, .1)


class MetricTests(unittest.TestCase):
    settings = GlobalPreprocessConfig(segmenter='dummy')

    def test_height_bands_in_metres(self):
        # 负值与 NaN 不计入（dsm_height_features 已把超出容差的负值置为 NaN）
        height = np.array([[-1., .2, 1.], [2.9, 3., np.nan]], np.float32)
        self.assertEqual(bf.height_band_pct(height, (1., 3.)), {'0-1m': 25., '1-3m': 50., '>3m': 25.})
        self.assertEqual(bf.band_labels((1., 3.)), ['0-1m', '1-3m', '>3m'])

    def test_block_center_road_is_zero(self):
        labels = np.full((40, 40), bf.PLANT, np.uint8)
        labels[:, :10] = bf.ROAD
        dsm = np.where(labels == bf.ROAD, -120., -100.).astype(np.float32)  # 高程基准为负也无妨
        dsm[30:, 20:] = -125.  # 比路面低：不算有效像元
        result = bf.dsm_height_features(dsm, labels, self.settings, 1., (20., 5.))
        stats = result['height_stats']
        self.assertEqual((stats['base_ref'], stats['base_dsm_m']), ('center_road', -120.))
        self.assertAlmostEqual(stats['below_base_pct'], 12.5)
        self.assertTrue(np.isnan(result['_relative'][35, 30]))
        self.assertEqual(result['height_band_pct'], {'0-0.5m': 25/.875, '0.5-2m': 0., '>2m': 100-25/.875})
        # 整体抬高 1000 m：结果不变
        shifted = bf.dsm_height_features(dsm+1000, labels, self.settings, 1., (20., 5.))
        self.assertEqual(shifted['height_band_pct'], result['height_band_pct'])
        # 路面噪声：略低于路面（-0.2 m）仍计为 0 m；严格模式（容差 0）则剔除
        noisy = dsm.copy()
        noisy[:, 5] -= .2
        self.assertEqual(bf.dsm_height_features(noisy, labels, self.settings, 1., (20., 5.))['height_stats']['below_base_pct'],
                         12.5)
        strict = replace(self.settings, feature_below_base_tolerance_m=0.)
        self.assertGreater(bf.dsm_height_features(noisy, labels, strict, 1., (20., 5.))['height_stats']['below_base_pct'], 12.5)
        # 中心不在路上：中心 5 m 方窗够不到道路，10 m 方窗能取得有效道路 DSM。
        off = bf.dsm_height_features(dsm, labels, self.settings, 1., (10., 13.))['height_stats']
        self.assertEqual((off['base_ref'], off['base_dsm_m'], off['base_offset_m']), ('expanded_road', -120., 4.))
        far = bf.dsm_height_features(dsm, labels, self.settings, 1., (10., 30.))
        self.assertIsNone(far['height_band_pct'])
        # 块内没有路面：不判高度
        none = bf.dsm_height_features(dsm, np.full_like(labels, bf.PLANT), self.settings, 1., (20., 5.))
        self.assertIsNone(none['height_band_pct'])
        self.assertIsNone(none['height_stats'])

    def test_center_on_building_uses_road_not_roof(self):
        # 拓扑线偏到路边建筑上：中心是 8 m 屋顶，真实路面在左侧。旧做法以屋顶为 0 点，路面全成“阶跃低于”。
        labels = np.full((60, 60), bf.PLANT, np.uint8)
        labels[:, :20] = bf.ROAD
        labels[20:40, 20:40] = bf.BUILDING
        dsm = np.full((60, 60), 30., np.float32)
        dsm[20:40, 20:40] = 38.
        result = bf.dsm_height_features(dsm, labels, self.settings, 1., (29.5, 22.5))
        stats = result['height_stats']
        self.assertEqual((stats['base_ref'], stats['base_dsm_m']), ('expanded_road', 30.))
        self.assertEqual(stats['step_below_pct'], 0.)
        self.assertAlmostEqual(float(np.median(result['_relative'][20:40, 20:40])), 8.)
        degradation = _degradation({'class_pct': bf.class_pct(labels), 'height_band_pct': result['height_band_pct'],
                                    'height_stats': stats}, has_dsm=True)
        self.assertNotIn('below_road', degradation['height_reasons'])

    def test_optional_detrend_removes_terrain_slope(self):
        # 64×64 网格、1 m/px、10% 坡面上有一栋 8 m 建筑：去坡后地面 ≈0，建筑 ≈8 m
        labels = np.full((64, 64), bf.PLANT, np.uint8)
        labels[:, 30:34] = bf.ROAD
        slope = np.repeat(np.arange(64, dtype=np.float32)[None]*.1, 64, 0)
        dsm = 50+slope
        dsm[10:30, 40:60] += 8
        labels[10:30, 40:60] = bf.BUILDING
        center = (32., 31.5)
        result = bf.dsm_height_features(dsm, labels, replace(self.settings, feature_dsm_detrend=True), 1., center)
        self.assertAlmostEqual(result['height_stats']['terrain_slope_pct'], 10., delta=.5)
        relative = result['_relative']
        self.assertLess(float(np.nanmax(relative[labels != bf.BUILDING])), .3)
        self.assertAlmostEqual(float(np.median(relative[10:30, 40:60])), 8., delta=.3)
        # 默认不去坡：路面左侧下坡低于路面，整片不计入
        flat = bf.dsm_height_features(dsm, labels, self.settings, 1., center)
        self.assertGreater(flat['height_stats']['below_base_pct'], 40)
        self.assertLess(result['height_stats']['below_base_pct'], 5)
        self.assertIsNone(flat['height_stats']['terrain_slope_pct'])

    def test_slope_below_is_ground_step_below_is_not(self):
        """缓坡下降（与路面连续）计为 0 m 地面；隔着 0.5 m 以上台阶的低处为阶跃低于（不合理的低）。"""
        n = 64
        labels = np.full((n, n), bf.PLANT, np.uint8)
        labels[:, 30:34] = bf.ROAD
        center = ((n-1)/2, (n-1)/2)
        # 1 m/px，路面 100 m，向东 10% 缓坡下降：东侧最低约 -3 m，但处处连续
        x = np.arange(n, dtype=np.float32)
        slope = np.tile(100-np.clip(x-32, 0, None)*.1, (n, 1)).astype(np.float32)
        result = bf.dsm_height_features(slope, labels, self.settings, 1., center)
        stats = result['height_stats']
        self.assertGreater(stats['slope_below_pct'], 30)
        self.assertEqual(stats['step_below_pct'], 0.)
        self.assertAlmostEqual(sum(result['height_band_pct'].values()), 100, places=3)
        # 同样的东侧低地，改成在 x=40 处直接下降 1 m（阶跃）：东侧整片为阶跃低于
        step = np.full((n, n), 100., np.float32)
        step[:, 40:] = 99.
        stats = bf.dsm_height_features(step, labels, self.settings, 1., center)['height_stats']
        self.assertGreater(stats['step_below_pct'], 30)
        self.assertEqual(stats['slope_below_pct'], 0.)
        # 0.3 m 小坎（< 0.5 m）不算台阶
        curb = np.full((n, n), 100., np.float32)
        curb[:, 40:] = 99.6
        stats = bf.dsm_height_features(curb, labels, self.settings, 1., center)['height_stats']
        self.assertEqual(stats['step_below_pct'], 0.)
        self.assertGreater(stats['slope_below_pct'], 30)
        # 退化规则只看阶跃低于
        from agent.map_preprocess.nodes.block_features import _degradation
        d = _degradation({'class_pct': None, 'height_band_pct': {'0-1m': 50., '1-3m': 50., '>3m': 0.},
                          'height_stats': {'below_base_pct': 90., 'step_below_pct': 10., 'slope_below_pct': 80.}})
        self.assertEqual(d['height_reasons'], [])

    def test_degradation_rules(self):
        from agent.map_preprocess.nodes.block_features import _degradation
        def run(cls, bands, below=0.):
            return _degradation({'class_pct': cls, 'height_band_pct': bands,
                                 'height_stats': None if bands is None else {'below_base_pct': below}})
        mixed = {'road': 30., 'building': 20., 'plant': 50., 'other': 0.}
        # 语义：道路 12.4% 不满足 < 10%；道路 8% 且植被 87.8% 满足
        self.assertFalse(run({'road': 12.4, 'building': 4.3, 'plant': 83.1, 'other': .2}, None)['semantic_degraded'])
        result = run({'road': 8., 'building': 4., 'plant': 87.8, 'other': .2}, None)
        self.assertEqual((result['semantic_degraded'], result['semantic_road_pct'], result['semantic_plant_pct']), (True, 8., 87.8))
        self.assertEqual((result['height_degraded'], result['degraded']), (None, True))  # 没有 DSM：只看语义
        # 高度：0-1m > 90% → flat
        result = run(mixed, {'0-1m': 92., '1-3m': 5., '>3m': 3.})
        self.assertEqual((result['height_degraded'], result['height_reasons']), (True, ['flat']))
        # >3m 占比大不算退化
        self.assertFalse(run(mixed, {'0-1m': 5., '1-3m': 5., '>3m': 90.})['height_degraded'])
        self.assertFalse(run(mixed, {'0-1m': 70., '1-3m': 20., '>3m': 10.})['height_degraded'])
        # 低于路面 > 70% → below_road
        result = run(mixed, {'0-1m': 60., '1-3m': 20., '>3m': 20.}, below=75.)
        self.assertEqual((result['height_degraded'], result['height_reasons']), (True, ['below_road']))
        self.assertEqual(run(mixed, {'0-1m': 95., '1-3m': 5., '>3m': 0.}, below=80.)['height_reasons'], ['flat', 'below_road'])
        # 恰好等于阈值不算
        self.assertFalse(run({'road': 10., 'building': 25., 'plant': 65., 'other': 0.},
                             {'0-1m': 90., '1-3m': 10., '>3m': 0.}, below=70.)['degraded'])
        # 全部像元低于路面：没有分段，但仍判 below_road
        result = _degradation({'class_pct': mixed, 'height_band_pct': None, 'height_stats': {'below_base_pct': 100.}})
        self.assertEqual(result['height_reasons'], ['below_road'])
        self.assertIsNone(_degradation({'class_pct': None, 'height_band_pct': None, 'height_stats': None})['semantic_degraded'])
        # 有 DSM：不判语义（即便语义满足退化条件），只看高度
        veg = {'road': 2., 'building': 0., 'plant': 98., 'other': 0.}
        result = _degradation({'class_pct': veg, 'height_band_pct': {'0-1m': 50., '1-3m': 30., '>3m': 20.},
                               'height_stats': {'below_base_pct': 0.}}, has_dsm=True)
        self.assertEqual((result['semantic_degraded'], result['semantic_plant_pct'], result['degraded']), (None, 98., False))
        result = _degradation({'class_pct': veg, 'height_band_pct': {'0-1m': 95., '1-3m': 5., '>3m': 0.},
                               'height_stats': {'below_base_pct': 0.}}, has_dsm=True)
        self.assertEqual((result['semantic_degraded'], result['height_degraded'], result['degraded']), (None, True, True))

    def test_other_from_low_confidence(self):
        mask = np.array([[0, 1], [2, 255]], np.uint8)
        confidence = np.array([[.9, .4], [.6, 0]], np.float32)
        labels = bf.semantic_labels(mask, confidence, .5)
        self.assertEqual(labels.tolist(), [[0, 3], [2, 255]])
        self.assertEqual(bf.class_pct(labels), {'road': 100/3, 'building': 0., 'plant': 100/3, 'other': 100/3})


if __name__ == '__main__':
    unittest.main()
