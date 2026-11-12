"""Node 3 掩码拓扑缓存：缓存 road_topo.run_arrays 的结果（R/S 融合、骨架化、遮挡补全、面状区域等），命中时不再重算。

键 = MaGRoad 缓存键（含输入哈希、权重、分辨率）+ 整图语义缓存键（含模型身份）+ 图像尺寸/分辨率
     + 配置里 road_topo 一节 + road_topo.py 源码哈希 + CACHE_VERSION。
改 Node 4 及之后的参数不会让缓存失效；改 road_topo 参数或代码会自动失效。
_topology_from_masks 中 run_arrays 之前的输入准备（概率/语义缩放）若有改动，要递增 CACHE_VERSION。

缓存目录 <cache_dir>/road_topo/<时间>_<图名>_<键前 12 位>/：
  result.pkl（(路网, 每步统计, 删除的分支点列)）、meta.json、data/（state/road_topo 的数据）、figures/（01b_road_topo 图）。
先写临时目录、fsync 后整体改名，写入时断电不会留下半个缓存；读取失败时重新计算。
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import pickle
import shutil
from typing import Any
from uuid import uuid4
import warnings

import yaml

from ..global_config import GlobalPreprocessConfig
from ..io import fsync_path, read_json, resolve_path, write_json
from .. import output_layout as layout
from ..state import PreprocessState

CACHE_VERSION = 1
ROAD_TOPO_SOURCE = Path(__file__).resolve().parents[1]/'utils'/'road_topo.py'


def topo_settings(config: GlobalPreprocessConfig) -> Any:
    """配置文件里的 road_topo 一节（与 road_topo.load_config 读取的内容相同）。"""
    values = yaml.safe_load(resolve_path(config.road_topo_config).read_text(encoding='utf-8')) or {}
    return values.get('road_topo', values)


def cache_key(state: PreprocessState, config: GlobalPreprocessConfig, semantic_segmenter: Any, geotiff: bool) -> str | None:
    """不可缓存（关闭缓存、没有 MaGRoad 缓存、注入的语义模型没有身份）时返回 None。"""
    from .semantic_roads import semantic_cache_key
    road = state.get('road_graph_cache') or {}
    if not config.cache_enabled or not road.get('key') or road.get('enabled') is False:
        return None
    semantic, fingerprint, *_ = semantic_cache_key(state, config, semantic_segmenter)
    if fingerprint is None:
        return None
    payload = {'version': CACHE_VERSION, 'road_graph': road['key'], 'semantic': semantic,
               'size': [state['image']['width'], state['image']['height']], 'resolution': state['resolution_m_per_px'],
               'geotiff': geotiff, 'settings': topo_settings(config),
               'code': hashlib.sha256(ROAD_TOPO_SOURCE.read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def cache_root(config: GlobalPreprocessConfig) -> Path:
    return resolve_path(config.cache_dir)/'road_topo'


def load(config: GlobalPreprocessConfig, key: str, out_dir: Path, vis_dir: Path, logger: logging.Logger) -> tuple | None:
    """命中时把数据和图复制到本次任务目录，返回 (路网, 每步统计, 删除的分支点列)；未命中或缓存损坏返回 None。"""
    directory = layout.find_cache(cache_root(config), key)
    if directory is None:
        return None
    try:
        if read_json(directory/'meta.json').get('key') != key:
            return None
        with (directory/'result.pkl').open('rb') as handle:
            result = pickle.load(handle)   # 本项目自己写的缓存文件
        for source, target in ((directory/'data', out_dir), (directory/'figures', vis_dir)):
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
    except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError, AttributeError, ImportError) as exc:
        warnings.warn(f'道路拓扑缓存无效，将重新计算：{exc}', RuntimeWarning)
        return None
    logger.info('道路拓扑缓存命中：%s', directory)
    return result


def save(config: GlobalPreprocessConfig, key: str, image_path: str, result: tuple, out_dir: Path, vis_dir: Path,
         logger: logging.Logger) -> Path | None:
    root = cache_root(config)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root/f'.{uuid4().hex}'
    try:
        temporary.mkdir()
        with (temporary/'result.pkl').open('wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for source, name in ((out_dir, 'data'), (vis_dir, 'figures')):
            if source.is_dir():
                shutil.copytree(source, temporary/name)
        write_json(temporary/'meta.json', {'key': key, 'version': CACHE_VERSION, 'image_path': image_path})
        for path in temporary.rglob('*'):
            if path.is_file():
                fsync_path(path)   # 先落盘再改名：写入时断电也不会留下半个缓存
        target = layout.find_cache(root, key) or root/layout.cache_name(image_path, key)
        if target.exists():
            shutil.rmtree(target)
        temporary.replace(target)
    except OSError as exc:   # 写缓存失败不影响本次结果
        warnings.warn(f'道路拓扑缓存写入失败：{exc}', RuntimeWarning)
        shutil.rmtree(temporary, ignore_errors=True)
        return None
    logger.info('道路拓扑已缓存：%s', target)
    return target
