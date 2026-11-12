"""整图沿路预处理流水线（global_blocks）；不导入或调用 VLM。"""
from .config import PreprocessConfig, load_config
from .global_config import GlobalPreprocessConfig
from .graph import build_graph, run_pipeline
from .io import load_manifest
from .utils.global_blocks import crop_block, crop_context, crop_blocks_for_group

__all__ = ['PreprocessConfig', 'GlobalPreprocessConfig', 'load_config', 'build_graph', 'run_pipeline',
           'load_manifest', 'crop_block', 'crop_context', 'crop_blocks_for_group']
