"""global_blocks 任务目录布局：状态/数据与可视化分开存放。

<task_id>/
  state/                         # 节点状态快照与它们引用的数据（JSON、日志、表格）
    config.json  run.log  state.json
    node_states/01_<node>.json …  # 每个节点结束时的完整 state，编号即执行顺序
    raw_road_graph.json  clean_road_graph.json  manifest.json  block_features.json
    block_features_table/
  visualization/                 # 全部图片，编号即建议查看顺序
    01_road_cleaning.png …       # 见 FIGURES
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .io import resolve_path

STATE_DIR = 'state'
VIS_DIR = 'visualization'
NODE_STATES_DIR = 'node_states'

FIGURES = {
    'scene_type_inputs': '00_scene_type_inputs',      # Node 1b：送入 VLM 的整图缩略图（目录，magroad_scene）
    'road_cleaning': '01_road_cleaning.png',          # Node 3：保留边彩色、删除边红色
    'road_topo_stages': '01b_road_topo',              # Node 3：掩码拓扑每个阶段的图、影像叠加图（目录）
    'road_topology': '02_road_topology.png',          # Node 4：最终道路拓扑（粗线、路口、边 ID）
    'blocks_initial': '03_blocks_initial.png',        # Node 4：去重前沿每条边独立布块
    'blocks_final': '04_blocks_final.png',            # Node 4：最终全图块
    'blocks_degraded': '05_blocks_degraded.png',      # Node 5：全图块上标出退化块
    'degraded_sheets': '06_degraded_block_sheets',    # Node 5：退化块图片 + 指标表拼图（目录）
    'block_images': '07_per_block_images',            # Node 5：每块 rgb/语义/高度原图（目录）
    'contact_sheets': '08_contact_sheets',            # 手动抽块拼图工具的默认输出（目录）
    'vlm_review': '09_vlm_review.png',                # Node 6：VLM 复核结果（保留/扩大/删除）
    'vlm_review_inputs': '10_vlm_review_inputs',      # Node 6：每个复核块送入 VLM 的 rgb/semantic/height 图（目录）
    'vlm_expansion': '11_vlm_expansion.png',          # Node 7：各支路最大确认尺寸
    'vlm_expansion_inputs': '12_vlm_expansion_inputs',  # Node 7：每条支路送入 VLM 的双图（目录）
    'degradation_fusion': '13_degradation_fusion.png',  # Node 8：融合后的退化区域、来源 Patch 与外接矩形
    'localization_recommendation': '14_localization_recommendation.png',  # Node 9：各区域推荐的辅助定位方式
    'localization_inputs': '15_localization_inputs',  # Node 9：每个区域送入 VLM 的 RGB/掩膜/语义图（目录）
    'branch_review': 'branch_review_input.png',       # Node 3 分支复核的输入图（启用时）
    'debug_blocks': 'debug_blocks',                   # debug 模式下的块原图（目录）
}


def cache_name(image_path: str | Path, key: str, suffix: str = '') -> str:
    """缓存的新名字：<创建时间>_<图名>_<哈希前 12 位><后缀>，按名字排序就是时间顺序。"""
    stem = Path(image_path).stem.strip('. ') or 'image'
    return f'{datetime.now():%Y%m%d_%H%M%S}_{stem}_{key[:12]}{suffix}'


def find_cache(root: str | Path, key: str, suffix: str = '') -> Path | None:
    """按哈希找已有缓存；旧的纯哈希命名（完整哈希目录、前 24 位文件）仍然命中。"""
    root = Path(root)
    for legacy in (root/f'{key}{suffix}', root/f'{key[:24]}{suffix}'):
        if legacy.exists():
            return legacy
    matches = sorted(root.glob(f'*_{key[:12]}{suffix}')) if root.is_dir() else []
    return matches[-1] if matches else None


def state_dir(task_dir: str | Path) -> Path:
    path = resolve_path(task_dir)/STATE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def vis_dir(task_dir: str | Path) -> Path:
    path = resolve_path(task_dir)/VIS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def figure(task_dir: str | Path, key: str) -> Path:
    return vis_dir(task_dir)/FIGURES[key]


def diagnostic_dirs(task_dir: str | Path, node: str, *, dry_run: bool) -> tuple[Path, Path]:
    mode = 'dry_run' if dry_run else 'subset'
    return (vis_dir(task_dir)/'diagnostics'/node/mode, state_dir(task_dir)/'diagnostics'/node/mode)


def task_dir_of_manifest(manifest_path: str | Path) -> Path:
    """manifest 在 state/ 下时返回其上一级；旧布局（manifest 在任务目录根）返回所在目录。"""
    parent = resolve_path(manifest_path).parent
    return parent.parent if parent.name == STATE_DIR else parent


def node_state_path(task_dir: str | Path, node: str) -> Path:
    """节点快照路径 state/node_states/NN_<node>.json；同名节点已存在时复用其编号。"""
    directory = state_dir(task_dir)/NODE_STATES_DIR
    directory.mkdir(exist_ok=True)
    existing = sorted(directory.glob('[0-9][0-9]_*.json'))
    for path in existing:
        if path.stem[3:] == node:
            return path
    return directory/f'{len(existing)+1:02d}_{node}.json'
