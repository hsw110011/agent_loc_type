"""独立的轻量路径/JSON/窗口读取，不导入 agent.vlm.common 的 VLM 依赖。"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
from typing import Any, Iterator
import warnings

import numpy as np
from PIL import Image

from agent.paths import PROJECT_ROOT, fsync_path, resolve_path, relative_path, write_json, read_json


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = read_json(path)
    if manifest.get('format_version') == 2 and manifest.get('workflow') == 'global_blocks':
        if not all(key in manifest for key in ('image', 'tiles', 'nodes', 'edges', 'blocks', 'groups', 'files')):
            raise ValueError('不完整的 global_blocks manifest')
        return manifest
    raise ValueError('只支持 format_version=2 的 global_blocks manifest（旧的采样点 manifest 已不再支持）')


class ImageReader:
    """rasterio 窗口读取；Pillow 降级可能解码整图，受显式像元上限约束。"""
    def __init__(self, path: str | Path, pillow_max_image_pixels: int):
        self.path = resolve_path(path)
        self.raster = None
        self.pillow = None
        try:
            import rasterio
        except ImportError:
            warnings.warn('rasterio 缺失：使用 Pillow，无法读取 CRS，压缩图可能整幅解码', RuntimeWarning)
        else:
            from rasterio.errors import RasterioIOError
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', rasterio.errors.NotGeoreferencedWarning)
                    self.raster = rasterio.open(self.path)
            except RasterioIOError:
                pass  # Pillow 支持但 GDAL 不支持的普通图片仍可读取。
        if (self.raster is not None and self.raster.driver == 'JPEG' and self.raster.crs is None
                and all({flag.name for flag in flags} == {'all_valid'} for flags in self.raster.mask_flag_enums)):
            # 普通 JPEG 统一用 Pillow 解码（不依赖扩展名），避免 GDAL/libjpeg
            # 与展示/裁剪路径的 IDCT 差异。带地理信息的源仍保留 rasterio。
            self.raster.close()
            self.raster = None
        if self.raster is not None:
            self.width, self.height = self.raster.width, self.raster.height
            if self.raster.count < 3 or any(dtype != 'uint8' for dtype in self.raster.dtypes[:3]):
                self.close()
                raise ValueError('输入必须为至少三个 uint8 RGB 波段；不隐式拉伸数据')
        else:
            # 假设 fallback 上限足以限制内存；不关闭 Pillow 的全局大图安全检查。
            self.pillow = Image.open(self.path)
            self.width, self.height = self.pillow.size
            if self.width * self.height > pillow_max_image_pixels:
                self.close()
                raise ValueError('Pillow 降级读取超过 pillow_max_image_pixels；请使用 rasterio')

    def read(self, bbox: list[int] | tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = bbox
        if not 0 <= x0 < x1 <= self.width or not 0 <= y0 < y1 <= self.height:
            raise ValueError(f'读取窗口越界：{bbox}')
        if self.raster is not None:
            from rasterio.windows import Window
            return self.raster.read([1, 2, 3], window=Window(x0, y0, x1-x0, y1-y0)).transpose(1, 2, 0)
        return np.array(self.pillow.crop(bbox).convert('RGB'))

    def valid_mask(self, bbox: list[int]) -> np.ndarray:
        if self.raster is not None:
            from rasterio.windows import Window
            x0, y0, x1, y1 = bbox
            return self.raster.dataset_mask(window=Window(x0, y0, x1-x0, y1-y0)) > 0
        return np.ones((bbox[3]-bbox[1], bbox[2]-bbox[0]), dtype=bool)

    def close(self) -> None:
        for source in (self.raster, self.pillow):
            if source is not None:
                source.close()


@contextmanager
def open_image(path: str | Path, pillow_max_image_pixels: int) -> Iterator[ImageReader]:
    source = ImageReader(path, pillow_max_image_pixels)
    try:
        yield source
    finally:
        source.close()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve_path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
