"""把 MaGRoadExtractor 接到整图预处理（Node 2）：输入整图路径，输出与 SAM-Road 适配器相同的 RawRoadGraph。"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from agent.paths import relative_path, resolve_path, write_json
from percection_tool.interfaces import RawRoadGraph
from .inference import MaGRoadExtractor, black_border_invalid, parse_checkpoint_spec

if TYPE_CHECKING:
    from agent.map_preprocess.global_config import GlobalPreprocessConfig


class MaGRoadGraphExtractor:
    """checkpoint 写法同 run_magroad：'wildroad' 或 'globalscale:ITSC=0.1,ROAD=0.15,TOPO=0.35'。"""

    def __init__(self, config: GlobalPreprocessConfig, resolution_m_per_px: float, output_dir: str, checkpoint: str):
        self.config = config
        self.resolution = resolution_m_per_px
        self.output_dir = resolve_path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint, self.label, self.overrides = parse_checkpoint_spec(checkpoint)
        self.model = MaGRoadExtractor(self.checkpoint, threshold_overrides=self.overrides)

    def _read(self, path: str | Path) -> tuple[np.ndarray, np.ndarray, float, dict | None, str]:
        from agent.map_preprocess.io import open_image
        with open_image(path, self.config.pillow_max_image_pixels) as source:
            bbox = [0, 0, source.width, source.height]
            if source.width*source.height <= self.config.max_tile_pixels:
                rgb, valid, resolution, decimated = source.read(bbox), source.valid_mask(bbox), self.resolution, None
            elif source.raster is None:
                raise ValueError('整图超过 max_tile_pixels 且无法用 rasterio 降采样读取，请核对内存后显式提高上限')
            else:
                # 超大原图直接按权重训练 GSD 降采样读取，与“载入原图再缩到工作网格”等价。
                from rasterio.enums import Resampling
                target = self.model.model_resolution
                size = (max(1, round(source.width*self.resolution/target)),
                        max(1, round(source.height*self.resolution/target)))
                rgb = source.raster.read([1, 2, 3], out_shape=(3, size[1], size[0]),
                                         resampling=Resampling.lanczos).transpose(1, 2, 0)
                valid = source.raster.dataset_mask(out_shape=(size[1], size[0]), resampling=Resampling.nearest) > 0
                resolution = target
                decimated = {'source_size': [source.width, source.height], 'read_size': list(size),
                             'scale_to_source_xy': [source.width/size[0], source.height/size[1]]}
        valid_source = 'dataset_mask'
        if valid.all():
            black = black_border_invalid(rgb, 15 if decimated is None else 3)
            if black is not None:
                valid, valid_source = ~black, 'black_border_fallback'
        return rgb, valid, resolution, decimated, valid_source

    def extract(self, image_path_or_array: str | Path | np.ndarray) -> RawRoadGraph:
        if isinstance(image_path_or_array, np.ndarray):
            rgb, valid, resolution, decimated, valid_source = image_path_or_array, None, self.resolution, None, 'none'
        else:
            rgb, valid, resolution, decimated, valid_source = self._read(image_path_or_array)
        try:
            output = self.model.predict_graph(rgb, resolution, valid)
        finally:
            self.model.close()
        nodes = [[float(x), float(y)] for x, y in output['nodes_xy']]
        if decimated is not None:
            sx, sy = decimated['scale_to_source_xy']
            nodes = [[x*sx, y*sy] for x, y in nodes]
        files, array_files = {}, {}
        for name in ('road_probability', 'keypoint_probability'):
            values = np.asarray(output.pop(name), dtype=np.float32)
            raw_path = self.output_dir/f'{name}.npy'
            np.save(raw_path, values, allow_pickle=False)
            path = self.output_dir/f'{name}.png'
            Image.fromarray((np.clip(values, 0, 1)*255).astype(np.uint8)).save(path)
            array_files[name] = relative_path(raw_path)
            files[name] = relative_path(path)
            files[name+'_raw'] = relative_path(raw_path)
        for name in ('working_rgb', 'working_valid'):
            output.pop(name, None)
        metadata = {**output['metadata'], 'tool': 'magroad', 'checkpoint': self.checkpoint, 'variant': self.label,
                    'threshold_overrides': self.overrides, 'valid_source': valid_source,
                    'probability_grid': 'metadata.working_size'}
        if decimated is not None:
            metadata['decimated_read'] = decimated
        native = self.output_dir/'magroad_native.json'
        write_json(native, {'nodes_xy': nodes, 'edges': output['edges'], 'edge_scores': output['edge_scores'],
                            'directed_predictions': output['directed_predictions'], 'metadata': metadata,
                            'array_files': array_files})
        files['magroad_native'] = relative_path(native)
        return {'vertices': [{'node_id': str(i), 'xy': xy} for i, xy in enumerate(nodes)],
                'edges': [{'u': str(a), 'v': str(b), 'polyline_xy': [nodes[a], nodes[b]], 'confidence': float(score)}
                          for (a, b), score in zip(output['edges'], output['edge_scores'])],
                'metadata': {**metadata, 'files': files}}
