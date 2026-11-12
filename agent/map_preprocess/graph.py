"""整图预处理 LangGraph（global_blocks）：Node 1-5（segmenter=magroad_scene 时 Node 1 后加 1b VLM 场景判断）+ Node 6 退化块 VLM 复核（可选）+ Node 7 退化区域尺寸扩展（可选）+ Node 8 退化区域空间融合（可选）+ Node 9 辅助定位方式推荐（可选）+ 占位扩展节点 + manifest 同步。
Node 5 没有退化候选、或 Node 6 没有确认退化时，跳过之后的节点直接同步 manifest。

节点返回 state 增量；每个节点结束时的完整 state 保存到 state/node_states/NN_<node>.json。
"""
from __future__ import annotations

from functools import partial
import logging
from typing import Callable
from datetime import datetime
from uuid import uuid4

from .global_config import GlobalPreprocessConfig, NODE_ORDER, PIPELINE_VERSION
from .io import relative_path, resolve_path, write_json
from . import output_layout as layout
from .nodes.global_pipeline import (load_image_and_resolution, extract_road_graph, clean_road_graph,
                                    build_blocks_and_groups, refresh_global_manifest)
from percection_tool.interfaces import RoadGraphExtractor, SemanticPredictor
from .utils.branch_review import BranchReviewer
from .state import PreprocessState
from .run_logging import ConsoleHandler, NodeProgress


def semantic_features(state: PreprocessState) -> dict:
    return {'semantic_features': {'status': 'placeholder', 'implemented': False}}


def elevation_features(state: PreprocessState) -> dict:
    return {'elevation_features': {'status': 'placeholder', 'implemented': False}}


def vlm_candidate_detection(state: PreprocessState) -> dict:
    return {'vlm_candidate_detection': {'status': 'placeholder', 'implemented': False}}


def _run_node(state: PreprocessState, function: Callable, progress: NodeProgress, index: int, **kwargs) -> dict:
    logger = logging.getLogger(f'map_preprocess.{state["task_id"]}')
    task_dir = resolve_path(state['output_dir'])
    records = layout.state_dir(task_dir)
    began = progress.begin(logger, index, function.__name__)
    try:
        update = function(state, **kwargs)
        merged = {**state, **update}
        write_json(layout.node_state_path(task_dir, function.__name__), merged)
        write_json(records/'state.json', merged)
    except Exception as exc:
        write_json(records/'error.json', {'node': function.__name__, 'error': str(exc)})
        logger.exception('节点失败：%s', function.__name__)
        raise
    progress.finish(logger, index, function.__name__, began, records/'node_timing.json')
    return update


EARLY_EXIT_NODES = ('compute_block_features', 'vlm_review_blocks')


def _route_after(state: PreprocessState, *, node: str) -> str:
    """Node 5 / Node 6 之后的分支：没有后续可做（见 final_summary.early_exit）时 'finish'，否则 'continue'。"""
    from .utils.final_summary import early_exit
    info = early_exit(state)
    if info is None or info['after'] != node:
        return 'continue'
    logging.getLogger(f'map_preprocess.{state["task_id"]}').info('%s，跳过后续节点，直接汇总 manifest', info['label'])
    return 'finish'


def shared_vlm_client(settings: GlobalPreprocessConfig, vlm_client=None, semantic_segmenter=None) -> tuple[object | None, bool]:
    """整次运行共用一个本地 VLM 客户端，返回 (客户端, 是否由本次运行负责关闭)。

    服务按需启动：Node 1b 用完即关闭（给 MaGRoad 让出显存），Node 6 首次请求时启动，之后 Node 7、Node 9 复用同一服务，
    流程结束再关闭，不再每个节点各启动一次。Node 7 需要整图语义：只有启用缓存、使用流程自带语义模型时才共用
    （此时语义一定命中 Node 3 写下的缓存，不会在 VLM 占满显存时再加载分割模型）；否则各节点仍自行启动、关闭。
    """
    if vlm_client is not None:
        return vlm_client, False
    limit = NODE_ORDER.index(settings.run_until) if settings.run_until else len(NODE_ORDER)-1
    nodes = (('vlm_review_enabled', 'vlm_review_blocks'), ('vlm_expansion_enabled', 'vlm_expand_blocks'),
             ('vlm_localization_enabled', 'recommend_localization'))
    used = sum(getattr(settings, flag) and NODE_ORDER.index(node) <= limit for flag, node in nodes)
    if used < 2 or not settings.cache_enabled or semantic_segmenter is not None:
        return None, False
    from agent.vlm.client import VLMClient
    return VLMClient(), True


def build_graph(*, config: GlobalPreprocessConfig | None = None, extractor: RoadGraphExtractor | None = None,
                reviewer: BranchReviewer | None = None, semantic_segmenter: SemanticPredictor | None = None,
                vlm_client=None):
    from langgraph.graph import END, START, StateGraph
    settings = config or GlobalPreprocessConfig()
    if not isinstance(settings, GlobalPreprocessConfig):
        raise ValueError('只支持 GlobalPreprocessConfig（global_blocks 流程）')
    graph = StateGraph(PreprocessState)
    order: list[str] = []
    progress = NodeProgress()
    limit = NODE_ORDER.index(settings.run_until) if settings.run_until else len(NODE_ORDER)-1
    def add_node(name: str, function: Callable, **kwargs) -> None:
        if name in NODE_ORDER and NODE_ORDER.index(name) > limit:
            return
        graph.add_node(name, partial(_run_node, function=function, progress=progress, index=progress.add(function.__name__), **kwargs))
        order.append(name)
    stages: list[Callable] = [load_image_and_resolution, extract_road_graph, clean_road_graph,
                              build_blocks_and_groups]
    if settings.segmenter == 'magroad_scene' and extractor is None:
        # Node 1b：VLM 判断整图结构化/非结构化，Node 2 据此选 MaGRoad 权重；注入 extractor 时无需选择。
        from .nodes.scene_type import classify_scene_type
        stages.insert(1, classify_scene_type)
    for function in stages:
        kwargs = {'config': settings}
        if function.__name__ == 'classify_scene_type':
            kwargs['client'] = vlm_client
        if function is extract_road_graph:
            kwargs['extractor'] = extractor
        if function is clean_road_graph:
            kwargs['reviewer'] = reviewer
            kwargs['semantic_segmenter'] = semantic_segmenter
        add_node(function.__name__, function, **kwargs)
    if settings.block_features_enabled:
        from .nodes.block_features import compute_block_features
        add_node('compute_block_features', compute_block_features, config=settings, semantic_segmenter=semantic_segmenter)
    if settings.vlm_review_enabled:
        from .nodes.vlm_review import vlm_review_blocks
        add_node('vlm_review_blocks', vlm_review_blocks, config=settings, client=vlm_client, semantic_segmenter=semantic_segmenter)
    if settings.vlm_expansion_enabled:
        from .nodes.vlm_expansion import vlm_expand_blocks
        add_node('vlm_expand_blocks', vlm_expand_blocks, config=settings, client=vlm_client, semantic_segmenter=semantic_segmenter)
    if settings.degradation_fusion_enabled:
        from .nodes.degradation_fusion import fuse_degraded_patches
        add_node('fuse_degraded_patches', fuse_degraded_patches, config=settings)
    if settings.vlm_localization_enabled:
        from .nodes.localization_recommendation import recommend_localization
        add_node('recommend_localization', recommend_localization, config=settings, client=vlm_client)
    for function in (semantic_features, elevation_features, vlm_candidate_detection):
        if limit == len(NODE_ORDER)-1 and getattr(settings, function.__name__+'_enabled'):
            # 加后缀避免与同名 state channel 冲突；业务接口名称仍为 function.__name__。
            add_node(function.__name__+'_placeholder', function)
    if limit >= NODE_ORDER.index('build_blocks_and_groups'):   # manifest 由 Node 4 生成，停在它之前就没有 manifest
        add_node('refresh_global_manifest', refresh_global_manifest)
    # 顺序连线；Node 5 / Node 6 之后加条件边：没有退化候选或没有确认的退化时跳过后续节点，直接汇总 manifest
    finish = 'refresh_global_manifest' if 'refresh_global_manifest' in order else END
    graph.add_edge(START, order[0])
    for index, name in enumerate(order):
        following = order[index+1] if index+1 < len(order) else END
        if name in EARLY_EXIT_NODES and following != finish:
            graph.add_conditional_edges(name, partial(_route_after, node=name), {'continue': following, 'finish': finish})
        else:
            graph.add_edge(name, following)
    return graph.compile()


def run_pipeline(image_path: str, config: GlobalPreprocessConfig | None = None,
                 *, dsm_path: str | None = None, extractor: RoadGraphExtractor | None = None,
                 reviewer: BranchReviewer | None = None,
                 semantic_segmenter: SemanticPredictor | None = None, vlm_client=None, console: bool = False) -> PreprocessState:
    """运行地图预处理，显式 DSM 优先于与 DOM 同名配对的 DSM。"""
    settings = config or GlobalPreprocessConfig()
    source = resolve_path(image_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if dsm_path is None and source.stem.endswith('_dom') and source.suffix.lower() in ('.tif', '.tiff'):
        for suffix in dict.fromkeys((source.suffix, '.tif', '.tiff')):
            paired = source.with_name(f'{source.stem.removesuffix("_dom")}_dsm{suffix}')
            if paired.is_file():
                dsm_path = str(paired)
                break
    if dsm_path is not None and not resolve_path(dsm_path).is_file():
        raise FileNotFoundError(resolve_path(dsm_path))
    # 任务 ID = 本地时间（到秒）+ 6 位随机后缀，如 20261104_051230_3eb8a1：按名字排序即运行先后，
    # 同一秒内重复运行也不会冲突。
    task_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    # 按源图名归档；保留独立任务 ID，避免同名图片或重复运行覆盖已有结果。
    image_name = source.stem.strip('. ') or 'image'
    directory = resolve_path(settings.output_dir)/image_name/task_id
    directory.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(f'map_preprocess.{task_id}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # 状态/数据放 state/，图片放 visualization/。
    records = layout.state_dir(directory)
    handler = logging.FileHandler(records/'run.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    terminal = ConsoleHandler() if console else None
    if terminal is not None:
        logger.addHandler(terminal)
        logger.info('任务 %s，完整日志：%s', task_id, relative_path(records/'run.log'))
    state: PreprocessState = {'image_path': relative_path(source), 'output_dir': relative_path(directory),
        'dsm_path': relative_path(resolve_path(dsm_path)) if dsm_path is not None else None,
        'task_id': task_id, 'pipeline_version': PIPELINE_VERSION, 'config': settings.to_dict(),
        'semantic_features': {'status': 'skipped'}, 'elevation_features': {'status': 'skipped'},
        'vlm_candidate_detection': {'status': 'skipped'}}
    write_json(records/'config.json', settings.to_dict())
    client, owned = shared_vlm_client(settings, vlm_client, semantic_segmenter)
    try:
        return build_graph(config=settings, extractor=extractor, reviewer=reviewer,
                           semantic_segmenter=semantic_segmenter, vlm_client=client).invoke(state)
    finally:
        if owned:
            client.close()
        logger.removeHandler(handler)
        handler.close()
        if terminal is not None:
            logger.removeHandler(terminal)
