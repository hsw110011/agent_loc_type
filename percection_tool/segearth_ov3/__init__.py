"""SegEarth-OV3（SAM 3 开放词汇）语义分割工具；只有调用 predict 时才加载模型。"""
from .inference import SegEarthOV3Segmenter

__all__ = ['SegEarthOV3Segmenter']
