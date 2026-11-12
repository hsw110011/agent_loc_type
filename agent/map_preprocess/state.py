"""与现有 LangGraph 一致：TypedDict + 节点增量更新，无大图/模型对象。"""
from typing import Any, TypedDict


class PreprocessState(TypedDict, total=False):
    image_path: str
    dsm_path: str | None
    output_dir: str
    task_id: str
    pipeline_version: int               # 流程算法版本（global_config.PIPELINE_VERSION）
    config: dict[str, Any]
    image: dict[str, Any]
    tiles: list[dict[str, Any]]
    resolution_m_per_px: float
    resolution_source: str
    manifest_path: str
    semantic_features: dict[str, Any]
    elevation_features: dict[str, Any]
    vlm_candidate_detection: dict[str, Any]
    block_features: dict[str, Any]
    vlm_review: dict[str, Any]
    vlm_expansion: dict[str, Any]
    degradation_fusion: dict[str, Any]  # Node 8：融合摘要，完整区域在 state/degradation_regions.json
    localization_recommendation: dict[str, Any]  # Node 9：逐区域推荐摘要，完整结果在 state/localization_recommendations.json
    final_summary: dict[str, Any]       # manifest 同步时汇总：conclusion = degraded / no_degradation / incomplete

    scene_type: dict[str, Any]          # Node 1b（segmenter=magroad_scene）：结构化/非结构化及选用的 MaGRoad 权重
    raw_road_graph_path: str
    clean_road_graph_path: str
    clean_graph_overview_path: str
    road_graph_cache: dict[str, Any]
    input_sha256: str
