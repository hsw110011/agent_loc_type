"""MaGRoad 道路拓扑工具（WildRoad / Global-Scale / SpaceNet 官方权重）。"""
from .inference import CHECKPOINTS, MaGRoadExtractor, black_border_invalid, parse_checkpoint_spec

__all__ = ["CHECKPOINTS", "MaGRoadExtractor", "black_border_invalid", "parse_checkpoint_spec"]
