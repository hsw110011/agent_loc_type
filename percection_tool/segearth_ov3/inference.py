"""SegEarth-OV3 重叠滑窗语义分割，滑窗融合与 SegEarth-OV-3/segment_geotiff.py 相同。

只输出三类：mask 为 0 road / 1 building / 2 plant / 255 无效。
与脚本的差别：不查询 background、water，也不按分数阈值判背景，每个有效像元都归入分数最高的一类。
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

from agent.paths import resolve_path
from percection_tool.interfaces import SemanticResult

if TYPE_CHECKING:
    from agent.perception_config import SegEarthOV3Config

ROAD, BUILDING, PLANT, IGNORE = 0, 1, 2, 255
# 同一行内逗号分隔的词是同义提示，取分数最大值后再和其它行比较。
PROMPTS = ('bareland,barren', 'grass', 'road', 'car', 'tree,forest', 'cropland', 'building,roof,house')
PROMPT_LABELS = (PLANT, PLANT, ROAD, ROAD, PLANT, PLANT, BUILDING)
CLASSES = ('road', 'building', 'plant')
PALETTE_RGB = ((255, 255, 255), (0, 0, 255), (0, 255, 0))
VENDOR_DIR = Path(__file__).resolve().parent/'vendor'


def expand_prompts() -> tuple[list[str], list[int]]:
    """展开为逐词查询；返回 (查询词, 每个查询所属的 PROMPTS 行号)。"""
    pairs = [(word.strip(), row) for row, prompt in enumerate(PROMPTS) for word in prompt.split(',')]
    return [word for word, _ in pairs], [row for _, row in pairs]


def window_starts(length: int, size: int, stride: int) -> list[int]:
    if length <= 0 or not 0 < stride < size:
        raise ValueError('滑窗参数非法')
    last = max(0, length-size)
    return [*range(0, last, stride), last]


def blend_weights(size: int, stride: int) -> np.ndarray:
    """重叠带内线性衰减的二维权重。"""
    overlap = size-stride
    axis = np.minimum(np.arange(size)+1, size-np.arange(size)).clip(max=overlap).astype(np.float32)/overlap
    return np.outer(axis, axis)


def score_to_confidence(score: np.ndarray, threshold: float) -> np.ndarray:
    """分数线性映射到 [0.5, 1]：不超过 threshold 取 0.5，1 对应 1。

    下限 0.5 使流程的 feature_min_class_confidence（默认 0.5）不会把像元降为 other，输出始终只有三类。
    """
    return (.5+.5*np.clip((score-threshold)/(1-threshold), 0., 1.)).astype(np.float32)


def cache_identity(settings: SegEarthOV3Config) -> dict:
    source = json.loads((resolve_path(settings.weights_dir)/'SOURCE.json').read_text(encoding='utf-8'))
    return {'model': 'segearth_ov3', 'sha256': source['files']['sam3.pt']['sha256'], 'tile_size': settings.tile_size,
            'stride': settings.stride, 'score_threshold': settings.score_threshold,
            'prompts': list(PROMPTS), 'prompt_labels': list(PROMPT_LABELS)}


class SegEarthOV3Segmenter:
    def __init__(self, config: SegEarthOV3Config | None = None):
        if config is None:
            from agent.perception_config import SegEarthOV3Config
            config = SegEarthOV3Config()
        self.settings = config
        self.processor = self.device = self.checkpoint_sha256 = None
        self.query_words, self.query_rows = expand_prompts()

    def _load(self) -> None:
        import torch
        from .import_model import sha256
        directory = resolve_path(self.settings.weights_dir)
        source = json.loads((directory/'SOURCE.json').read_text(encoding='utf-8'))
        expected = source['files']['sam3.pt']['sha256']
        if sha256(directory/'sam3.pt') != expected:
            raise ValueError('SAM 3 权重 SHA256 不符')
        if not torch.cuda.is_available():
            raise RuntimeError('SegEarth-OV3 使用 CUDA bfloat16 autocast，需要可用 CUDA')
        device = torch.device('cuda:0' if self.settings.device == 'auto' else self.settings.device)
        if torch.cuda.mem_get_info(device)[0] < self.settings.min_free_memory_mib*1024**2:
            raise RuntimeError('SegEarth-OV3 可用显存低于配置下限')
        if str(VENDOR_DIR) not in sys.path:  # vendor/sam3 内部使用绝对导入 sam3.*
            sys.path.insert(0, str(VENDOR_DIR))
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        model = build_sam3_image_model(bpe_path=str(VENDOR_DIR/'sam3'/'assets'/'bpe_simple_vocab_16e6.txt.gz'),
                                       checkpoint_path=str(directory/'sam3.pt'), device=str(device), load_from_HF=False)
        model = model.to(device)  # 上游只在 device 恰为 "cuda" 时才移动模型，cuda:n 需在此显式移动
        self.processor = Sam3Processor(model, confidence_threshold=self.settings.score_threshold, device=device)
        self.device, self.checkpoint_sha256 = device, expected

    def _view_scores(self, tile: np.ndarray) -> np.ndarray:
        """单窗各查询词的分数 (查询数, H, W)；移植自 segearthov3_segmentor._inference_single_view。"""
        import torch
        import torch.nn.functional as F
        from PIL import Image
        height, width = tile.shape[:2]
        scores = torch.zeros((len(self.query_words), height, width), device=self.device)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(tile))
            for index, word in enumerate(self.query_words):
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(state=state, prompt=word)
                for instance in range(state['masks_logits'].shape[0]):
                    logits = state['masks_logits'][instance].squeeze()
                    if logits.shape != (height, width):
                        logits = F.interpolate(logits.view(1, 1, *logits.shape), size=(height, width),
                                               mode='bilinear', align_corners=False).squeeze()
                    scores[index] = torch.max(scores[index], logits*state['object_score'][instance])
                semantic = state['semantic_mask_logits']
                if semantic.shape != (height, width):
                    semantic = F.interpolate(semantic, size=(height, width), mode='bilinear', align_corners=False).squeeze()
                scores[index] = torch.max(scores[index], semantic)
                scores[index] = scores[index]*state['presence_score']
        return scores.float().cpu().numpy()

    def predict(self, image: np.ndarray, valid_mask: np.ndarray | None = None) -> SemanticResult:
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError('语义输入必须为非空 HWC uint8 RGB')
        height, width = image.shape[:2]
        if height*width > self.settings.max_input_pixels:
            raise ValueError('语义输入超过 max_input_pixels；请按需裁剪块')
        if valid_mask is not None and (valid_mask.shape != (height, width) or valid_mask.dtype != np.bool_):
            raise ValueError('valid_mask 必须为同尺寸 bool 数组')
        started = time.monotonic()
        if self.processor is None:
            self._load()
        import torch
        torch.cuda.reset_peak_memory_stats(self.device)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(self.settings.threads)
        size, stride = self.settings.tile_size, self.settings.stride
        weights = blend_weights(size, stride)
        # 纯黑像元和无效像元一样不提供证据，也不产生预测。
        valid = (np.ones((height, width), bool) if valid_mask is None else valid_mask) & image.any(axis=2)
        scores = np.zeros((len(PROMPTS), height, width), dtype=np.float32)
        totals = np.zeros((height, width), dtype=np.float32)
        tiles = 0
        try:
            with torch.inference_mode():
                for y in window_starts(height, size, stride):
                    for x in window_starts(width, size, stride):
                        rows, columns = min(size, height-y), min(size, width-x)
                        tile_valid = valid[y:y+rows, x:x+columns]
                        if not tile_valid.any():
                            continue
                        # 小图补零到窗口尺寸，使模型看到的地物比例与配置一致。
                        tile = np.pad(image[y:y+rows, x:x+columns], ((0, size-rows), (0, size-columns), (0, 0)))
                        view = self._view_scores(tile)
                        if view.shape != (len(self.query_words), size, size):
                            raise RuntimeError(f'语义输出形状错误：{view.shape}')
                        if not np.isfinite(view).all():
                            raise RuntimeError('SegEarth-OV3 输出非有限分数')
                        merged = np.zeros((len(PROMPTS), rows, columns), dtype=np.float32)
                        for query, row in enumerate(self.query_rows):
                            np.maximum(merged[row], view[query, :rows, :columns], out=merged[row])
                        spatial = weights[:rows, :columns]*tile_valid
                        scores[:, y:y+rows, x:x+columns] += merged*spatial
                        totals[y:y+rows, x:x+columns] += spatial
                        tiles += 1
        finally:
            torch.set_num_threads(previous_threads)
        covered = totals > 0
        scores[:, covered] /= totals[covered]
        best = scores.max(axis=0)
        mask = np.asarray(PROMPT_LABELS, dtype=np.uint8)[scores.argmax(axis=0)]
        threshold = self.settings.score_threshold
        confidence = score_to_confidence(best, threshold)
        mask[~covered], confidence[~covered] = IGNORE, 0.
        valid_pixels = int(covered.sum())
        return {'mask': mask, 'confidence': confidence, 'metadata': {
            'model': 'segearth_ov3_sam3', 'classes': list(CLASSES), 'palette_rgb': [list(c) for c in PALETTE_RGB],
            'ignore_index': IGNORE, 'shape': [height, width], 'device': str(self.device),
            'runtime_torch_version': torch.__version__, 'precision': 'bfloat16_autocast',
            'tile_size': size, 'stride': stride, 'tile_count': tiles, 'valid_pixels': valid_pixels,
            'prompts': list(PROMPTS), 'prompt_labels': list(PROMPT_LABELS), 'score_threshold': threshold,
            'confidence_mapping': 'score<=score_threshold->0.5, 1->1, piecewise linear',
            'class_counts': {name: int(np.count_nonzero(mask == index)) for index, name in enumerate(CLASSES)},
            'class_ratios': {name: float(np.count_nonzero(mask == index)/valid_pixels) if valid_pixels else 0.
                             for index, name in enumerate(CLASSES)},
            'checkpoint_sha256': self.checkpoint_sha256,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(self.device),
            'elapsed_seconds': time.monotonic()-started}}

    def close(self) -> None:
        self.processor = self.device = None
