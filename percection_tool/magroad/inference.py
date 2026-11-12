"""官方 MaGRoad checkpoint 的纯 PyTorch 推理适配；不依赖训练/评测库。

流程与上游 inferencer.py 一致：滑窗 → 融合 keypoint/road 概率 → 三遍 NMS 取点 →
patch 内 kNN 候选对 → MaGTopoNet（使用该 patch 的 mask_logits 计算路径特征）→
跨 patch 平均边分数并按 TOPO_THRESHOLD 过滤。
与上游的差别仅在大图适配：输入先重采样到权重训练时的 GSD，滑窗步长取上游在其
测试图尺寸上的步长；patch 特征缓存在 CPU；输出坐标映射回原图像素网格。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import logging
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image

from agent.paths import resolve_path

WEIGHTS_DIR = 'percection_tool/magroad/weights'
THRESHOLD_KEYS = ('ITSC_THRESHOLD', 'ROAD_THRESHOLD', 'TOPO_THRESHOLD')
TRAINING_ONLY_PREFIXES =frozenset({'mask_criterion', 'mask_dice_criterion', 'topo_criterion'})


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    filename: str
    drive_id: str
    upstream_filename: str
    size_bytes: int
    sha256: str
    config: str
    # 权重训练数据的地面分辨率（论文 Tab.1 / 数据集说明）
    model_resolution_m_per_px: float
    # 上游测试图短边长度；与 INFER_PATCHES_PER_EDGE 一起确定上游滑窗步长
    reference_edge_px: int
    config_overrides: dict[str, Any] = field(default_factory=dict)
    note: str = ''


CHECKPOINTS: dict[str, CheckpointSpec] = {spec.name: spec for spec in (
    CheckpointSpec('wildroad', 'wildroad_magroad.ckpt', '1Bf20K-meo3HQEmdYswSbOcbUDViErR7R',
                   'wild_road_epoch=22-step=15226-val_loss=0.3988.ckpt', 1098271334,
                   '095bf4f1688d7172604ff855a22960173c869eeee20061ed61d92a07540ef8a6', 'wildroad.yaml', 0.3, 4096,
                   note='WildRoad：越野/野外道路，Google Earth 0.3 m/px，1024 patch'),
    CheckpointSpec('globalscale', 'globalscale_magroad.ckpt', '1JpeeNsT2l0v3yxF_5O_2IMg3ekhnFaxv',
                   'global_scale_epoch=1019-step=173401-val_loss=0.1139.ckpt', 1069588527,
                   'e834dbf0f37cfe7617405df8ca9c098be3d1d262a326d4dde8bf7d22f918e361', 'globalscale.yaml', 1.0, 2048,
                   note='Global-Scale：六大洲城乡道路，1 m/px，512 patch'),
    CheckpointSpec('spacenet', 'spacenet_magroad.ckpt', '1FXcOQl8xmjc068RpuV7oSIOpRRexXcXs',
                   'spacenet_epoch=16-step=11254-val_loss=0.1414.ckpt', 354873000,
                   '0431d96dbe516f0af39537fc281bde5bedbbeaade79d80f23a59b9ce8a1abe13', 'spacenet.yaml', 1.0, 400,
                   # 该权重由 cityspace 分支训练，其 MaGTopoNet 恒含几何特征（geo_proj）；
                   # spacenet.yaml 未写此键，main 分支会按缺省关闭。另外公开权重含
                   # topo_net.node_proj 且 edge_proj 输入为 768=2*256+128+128，说明训练时
                   # USE_POINT_FEATURES=True，与公开配置的 False 不符；按权重实际结构覆盖。
                   config_overrides={'USE_GEOMETRIC_FEATURES': True, 'USE_POINT_FEATURES': True},
                   note='SpaceNet：城市道路，1 m/px，256 patch（cityspace 分支权重）'),
)}


class UpstreamConfig(dict):
    """复现 addict.Dict 语义：缺省键读为空值（假），供生成的官方模型代码使用。"""

    def __getattr__(self, name: str) -> Any:
        if name.startswith('__'):
            raise AttributeError(name)
        return self.get(name)

    def get(self, key: str, default: Any = None) -> Any:  # noqa: D401 - 与 addict 一致
        return dict.get(self, key, default)


def load_config(spec: CheckpointSpec) -> UpstreamConfig:
    import yaml
    raw = yaml.safe_load((Path(__file__).parent/'vendor/configs'/spec.config).read_text(encoding='utf-8'))
    raw.update(spec.config_overrides)
    config = UpstreamConfig(raw)
    unsupported = {key: config[key] for key in ('NO_SAM', 'USE_SAM_DECODER', 'ENCODER_LORA', 'USE_FAST_NMS') if config.get(key)}
    # cityspace 分支配置无 MODEL_NAME，其 inferencer 无条件构建 MaGRoad。
    if unsupported or config.get('MODEL_NAME', 'MaGRoad') != 'MaGRoad':
        raise ValueError(f'不支持的 MaGRoad 推理配置：{unsupported or config.MODEL_NAME}')
    return config


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve_path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def nms_points(points: np.ndarray, scores: np.ndarray, radius: float) -> np.ndarray:
    """上游 nms_points_old 语义（分数降序、闭圆半径、分数 > 1 强制保留），网格索引代替 KDTree。"""
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int64)
    order = np.argsort(scores)[::-1]
    sorted_points, sorted_scores = points[order], scores[order]
    kept = np.ones(len(points), dtype=bool)
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (x, y) in enumerate(sorted_points):
        cells[(math.floor(x/radius), math.floor(y/radius))].append(index)
    buckets = {key: np.asarray(indices, dtype=np.int64) for key, indices in cells.items()}
    for index, point in enumerate(sorted_points):
        if not kept[index]:
            continue
        x, y = (math.floor(value/radius) for value in point)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                indices = buckets.get((x+dx, y+dy))
                if indices is None:
                    continue
                distances = np.square(sorted_points[indices]-point).sum(axis=1)
                neighbors = indices[distances <= radius*radius]
                kept[neighbors] = sorted_scores[neighbors] > 1.0
        kept[index] = True
    return sorted_points[kept]


def extract_points(keypoint_mask: np.ndarray, road_mask: np.ndarray, config: UpstreamConfig) -> np.ndarray:
    """上游 extract_graph_points(use_fast_nms=False)：路口点与道路点各自 NMS，再以路口优先合并。返回 (x, y)。"""
    sets = []
    for mask, threshold, radius in ((keypoint_mask, config.ITSC_THRESHOLD, config.ITSC_NMS_RADIUS),
                                     (road_mask, config.ROAD_THRESHOLD, config.ROAD_NMS_RADIUS)):
        y, x = np.nonzero(mask > threshold*255)
        sets.append(nms_points(np.column_stack((x, y)), mask[y, x], radius))
    points = np.concatenate(sets, axis=0)
    scores = np.r_[np.ones(len(sets[0])), np.zeros(len(sets[1]))]
    return nms_points(points, scores, config.ROAD_NMS_RADIUS)


def upstream_stride(spec: CheckpointSpec, config: UpstreamConfig) -> float:
    """上游 linspace 滑窗在其测试图上的步长（如 Global-Scale: (2048-512)/15≈102 px）。"""
    return (spec.reference_edge_px-config.PATCH_SIZE)/(config.INFER_PATCHES_PER_EDGE-1)


def patch_starts(length: int, size: int, stride: float) -> list[int]:
    if length < size:
        raise ValueError('内部输入应先填充至 PATCH_SIZE')
    count = max(1, math.ceil((length-size)/stride)+1)
    return sorted({round(value) for value in np.linspace(0, length-size, count)})


def neighbor_queries(points: np.ndarray, radius: float, max_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """等价于上游 KDTree.query(k=MAX+1, distance_upper_bound=R) 去掉自身后的候选对。"""
    count = len(points)
    sources = np.repeat(np.arange(count)[:, None], max_neighbors, axis=1)
    targets = sources.copy()
    valid = np.zeros(sources.shape, dtype=bool)
    for index, point in enumerate(points):
        distances = np.square(points-point).sum(axis=1).astype(np.float64)
        distances[index] = np.inf
        neighbors = np.flatnonzero(distances < radius*radius)
        neighbors = neighbors[np.argsort(distances[neighbors], kind='stable')][:max_neighbors]
        targets[index, :len(neighbors)] = neighbors
        valid[index, :len(neighbors)] = True
    return np.stack((sources, targets), axis=-1), valid


def parse_checkpoint_spec(text: str) -> tuple[str, str, dict[str, float]]:
    """'globalscale' 或 'globalscale:ROAD=0.2,TOPO=0.3' → (权重, 输出名, 阈值覆盖)。"""
    name, _, spec = str(text).strip().partition(':')
    if name not in CHECKPOINTS:
        raise ValueError(f'未知 MaGRoad 权重 {name!r}；可选 {sorted(CHECKPOINTS)}')
    overrides = {}
    for item in filter(None, (part.strip() for part in spec.split(','))):
        key, sep, value = item.partition('=')
        key = key.strip().upper().removesuffix('_THRESHOLD')
        if not sep or f'{key}_THRESHOLD' not in THRESHOLD_KEYS:
            raise ValueError(f'非法阈值写法 {item!r}；应为 ITSC/ROAD/TOPO=数值')
        overrides[key] = float(value)
        if not 0 < overrides[key] < 1:
            raise ValueError(f'阈值 {key}={value} 必须在 (0, 1)')
    label = name + ''.join(f'_{key.lower()}{value:g}' for key, value in overrides.items())
    return name, label, overrides


def black_border_invalid(rgb: np.ndarray, kernel_px: int = 15) -> np.ndarray | None:
    """无 nodata/alpha 的正射图外缘常为纯黑：开运算后的大片 (0,0,0) 视为无效；没有时返回 None。"""
    import cv2
    black = (rgb.max(axis=2) == 0).astype(np.uint8)
    if not black.any():
        return None
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, np.ones((kernel_px, kernel_px), np.uint8)) > 0
    return black if black.any() else None


class MaGRoadExtractor:
    """加载一个官方 MaGRoad checkpoint，输出原图像元边缘坐标中的节点/边。"""

    def __init__(self, checkpoint: str, *, weights_dir: str | Path = WEIGHTS_DIR, device: str = 'auto',
                 batch_size: int | None = None, model_resolution_m_per_px: float | None = None,
                 max_working_pixels: int = 64_000_000, verify_sha256: bool = True,
                 threshold_overrides: dict[str, float] | None = None):
        if checkpoint not in CHECKPOINTS:
            raise ValueError(f'未知 MaGRoad 权重 {checkpoint}；可选 {sorted(CHECKPOINTS)}')
        self.spec = CHECKPOINTS[checkpoint]
        self.config = load_config(self.spec)
        self.official_thresholds = {name: self.config[name] for name in THRESHOLD_KEYS}
        # 阈值只影响后处理（取点、连边），不改网络；键可写 ROAD 或 ROAD_THRESHOLD。
        for key, value in (threshold_overrides or {}).items():
            name = key.upper() if key.upper().endswith('_THRESHOLD') else f'{key.upper()}_THRESHOLD'
            if name not in THRESHOLD_KEYS or not 0 < float(value) < 1:
                raise ValueError(f'非法阈值覆盖 {key}={value}；可选 ITSC/ROAD/TOPO，取值 (0, 1)')
            self.config[name] = float(value)
        self.weights_path = resolve_path(weights_dir)/self.spec.filename
        self.requested_device = device
        self.batch_size = batch_size or self.config.INFER_BATCH_SIZE
        self.model_resolution = model_resolution_m_per_px or self.spec.model_resolution_m_per_px
        self.max_working_pixels = max_working_pixels
        self.verify_sha256 = verify_sha256
        self.model = None
        self.device = None
        self.load_report: dict[str, Any] = {}
        self.logger = logging.getLogger(f'percection_tool.magroad.{checkpoint}')

    def _load(self) -> None:
        import torch
        from .vendor.model import MaGRoad
        if not self.weights_path.is_file():
            raise FileNotFoundError(f'MaGRoad 权重不存在：{self.weights_path}；运行 python -m percection_tool.magroad.download')
        digest = None
        if self.verify_sha256:
            digest = file_sha256(self.weights_path)
            if digest != self.spec.sha256:
                raise ValueError(f'{self.spec.name} checkpoint SHA256 与记录不符：{digest}')
        requested = self.requested_device
        if requested == 'auto':
            requested = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if requested.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('指定 CUDA 不可用')
        self.device = torch.device(requested)
        model = MaGRoad(self.config)
        data = torch.load(self.weights_path, map_location='cpu', weights_only=True, mmap=True)
        state = dict(data['state_dict'])
        # 训练损失 buffer（如 BCEWithLogitsLoss 的 pos_weight）不属于推理网络；只剔除这一类，其余 strict。
        dropped = sorted(key for key in state if key.split('.', 1)[0] in TRAINING_ONLY_PREFIXES)
        for key in dropped:
            del state[key]
        incompatible = model.load_state_dict(state, strict=True)
        self.load_report = {'strict': True, 'missing_keys': list(incompatible.missing_keys),
                            'unexpected_keys': list(incompatible.unexpected_keys),
                            'dropped_training_only_keys': dropped,
                            'checkpoint_sha256': digest, 'epoch': data.get('epoch'),
                            'global_step': data.get('global_step')}
        del data
        self.model = model.to(self.device).eval()

    def close(self) -> None:
        self.model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def predict_graph(self, image: np.ndarray, resolution_m_per_px: float,
                      valid_mask: np.ndarray | None = None) -> dict[str, Any]:
        import torch
        started = time.monotonic()
        if self.model is None:
            self._load()
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('MaGRoad 输入必须为 uint8 RGB')
        if not math.isfinite(resolution_m_per_px) or resolution_m_per_px <= 0:
            raise ValueError('MaGRoad 需要显式有效的输入分辨率')
        config, size = self.config, self.config.PATCH_SIZE
        height, width = image.shape[:2]
        working_width = max(1, round(width*resolution_m_per_px/self.model_resolution))
        working_height = max(1, round(height*resolution_m_per_px/self.model_resolution))
        if working_width*working_height > self.max_working_pixels:
            raise ValueError('MaGRoad 工作网格超过 max_working_pixels；核对 GSD 或显式调大上限')
        if (working_width, working_height) == (width, height):
            rgb = image
        else:
            rgb = np.asarray(Image.fromarray(image).resize((working_width, working_height), Image.Resampling.LANCZOS))
        valid = np.ones((working_height, working_width), bool) if valid_mask is None else np.asarray(
            Image.fromarray(valid_mask.astype(np.uint8)*255).resize((working_width, working_height), Image.Resampling.NEAREST)) > 0
        padded = np.pad(rgb, ((0, max(0, size-working_height)), (0, max(0, size-working_width)), (0, 0)), mode='edge')
        ph, pw = padded.shape[:2]
        stride = upstream_stride(self.spec, config)
        boxes = [(x, y) for x in patch_starts(pw, size, stride) for y in patch_starts(ph, size, stride)]
        fused = np.zeros((ph, pw, 2), dtype=np.float32)
        counter = np.zeros((ph, pw), dtype=np.uint32)
        cache: list[tuple[Any, Any]] = []
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode():
            for offset in range(0, len(boxes), self.batch_size):
                subset = boxes[offset:offset+self.batch_size]
                batch = np.stack([padded[y:y+size, x:x+size] for x, y in subset])
                embeddings, logits, scores = self.model.infer_masks_and_img_features(
                    torch.as_tensor(batch, dtype=torch.float32, device=self.device))
                scores = scores.cpu().numpy()
                if not np.isfinite(scores).all():
                    raise RuntimeError('MaGRoad 概率图出现非有限值')
                embeddings, logits = embeddings.cpu(), logits.cpu()
                for index, (x, y) in enumerate(subset):
                    fused[y:y+size, x:x+size] += scores[index]
                    counter[y:y+size, x:x+size] += 1
                    cache.append((embeddings[index:index+1], logits[index:index+1]))
                self.logger.info('%s masks %d/%d', self.spec.name, min(offset+self.batch_size, len(boxes)), len(boxes))
            if not counter.all():
                raise RuntimeError('MaGRoad 滑窗未完整覆盖工作网格')
            fused /= counter[:, :, None]
            fused = fused[:working_height, :working_width]
            fused[~valid] = 0
            # 与上游一致：概率 ×255 截断为 uint8 后再阈值化。
            masks_uint8 = (fused*255).astype(np.uint8)
            points = extract_points(masks_uint8[:, :, 0], masks_uint8[:, :, 1], config)
            totals: dict[tuple[int, int], float] = defaultdict(float)
            counts: dict[tuple[int, int], int] = defaultdict(int)
            for index, (x, y) in enumerate(boxes):
                ids = np.flatnonzero((points[:, 0] >= x) & (points[:, 0] <= x+size) &
                                     (points[:, 1] >= y) & (points[:, 1] <= y+size))
                if not len(ids):
                    continue
                local = points[ids].astype(np.float32)-np.array([x, y], dtype=np.float32)
                pairs, pair_valid = neighbor_queries(local, config.NEIGHBOR_RADIUS, config.MAX_NEIGHBOR_QUERIES)
                embedding, logit = cache[index]
                topo = self.model.infer_toponet(
                    embedding.to(self.device), torch.as_tensor(local[None], device=self.device),
                    torch.as_tensor(pairs[None], dtype=torch.long, device=self.device),
                    torch.as_tensor(pair_valid[None], device=self.device), logit.to(self.device))
                topo = topo.squeeze(0).squeeze(-1).cpu().numpy()
                src, rank = np.nonzero(pair_valid)
                for a, b, score in zip(ids[src], ids[pairs[src, rank, 1]], topo[src, rank]):
                    if not math.isfinite(float(score)):
                        raise RuntimeError('MaGRoad 有效连接预测出现非有限值')
                    totals[(int(a), int(b))] += float(score)
                    counts[(int(a), int(b))] += 1
            peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None
        directed, edges = [], {}
        for edge in sorted(totals):
            score = totals[edge]/counts[edge]
            if score > config.TOPO_THRESHOLD:
                directed.append({'source': edge[0], 'target': edge[1], 'score': score})
                key = tuple(sorted(edge))
                edges[key] = max(score, edges.get(key, 0.0))
        sx, sy = width/working_width, height/working_height
        source_points = (points.astype(float)+.5)*[sx, sy]
        return {
            'nodes_xy': source_points.tolist(), 'edges': [list(edge) for edge in sorted(edges)],
            'edge_scores': [edges[edge] for edge in sorted(edges)], 'directed_predictions': directed,
            'working_rgb': rgb, 'working_valid': valid,
            'road_probability': fused[:, :, 1], 'keypoint_probability': fused[:, :, 0],
            'metadata': {
                'model': 'MaGRoad', 'checkpoint': self.spec.name, 'checkpoint_file': self.spec.filename,
                'upstream_checkpoint': self.spec.upstream_filename, 'checkpoint_load': self.load_report,
                'config': self.spec.config, 'config_overrides': self.spec.config_overrides,
                'device': str(self.device), 'dtype': 'float32',
                'source_size': [width, height], 'working_size': [working_width, working_height],
                'source_resolution_m_per_px': resolution_m_per_px,
                'model_resolution_m_per_px': self.model_resolution,
                'trained_resolution_m_per_px': self.spec.model_resolution_m_per_px,
                'scale_to_source_xy': [sx, sy], 'patch_size': size, 'patch_count': len(boxes),
                'patch_stride_px': stride, 'batch_size': self.batch_size,
                'node_count': len(points), 'edge_count': len(edges),
                'peak_cuda_allocated_bytes': peak,
                'inference_seconds_including_model_load': time.monotonic()-started,
                'thresholds': {name: config[name] for name in THRESHOLD_KEYS},
                'official_thresholds': self.official_thresholds,
                'radii_px': {name: config[name] for name in ('ITSC_NMS_RADIUS', 'ROAD_NMS_RADIUS', 'NEIGHBOR_RADIUS')},
            },
        }
