"""卫星图定位分析 Agent 的唯一 LangGraph 编排和 CLI 入口。"""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from rasterio.errors import RasterioError

# 支持直接执行 vlm/langgraph_main.py，也支持 python -m vlm.langgraph_main。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm.candidates import CandidateState, identify_candidates
from vlm.client import VLMClient, VLMError
from vlm.read_picture import read_picture
from vlm.parallel_vlm import assess_regions
from vlm.extract_roads import extract_roads
from vlm.visualization import draw_candidates
from vlm.common import save_state, relative_paths, resolve_path, write_json


def _run_node(state, function, **kwargs):
    update = relative_paths(function(state, **kwargs))
    merged = {**state, **update}
    save_state(merged, f'state_{function.__name__}.json')
    save_state(merged)
    return update


def build_graph(*, client: VLMClient | None = None, max_tokens: int = 4096,
                config_dir: str | Path | None = None, road_node=None, stop_after: str = 'roads'):
    """默认只跑 Node 1/2；full 显式开启 prompt 系统后续节点。"""
    from langgraph.graph import END, START, StateGraph
    if stop_after not in {'input', 'roads', 'candidates', 'visualization', 'full'}:
        raise ValueError('stop_after 必须为 input、roads、candidates、visualization 或 full')
    vlm = client or VLMClient()
    graph = StateGraph(CandidateState)
    graph.add_node('read_picture', partial(_run_node, function=read_picture, client=vlm))
    graph.add_edge(START, 'read_picture')
    if stop_after == 'input':
        graph.add_edge('read_picture', END)
        return graph.compile()
    graph.add_node('extract_roads', partial(_run_node, function=road_node or extract_roads))
    graph.add_edge('read_picture', 'extract_roads')
    if stop_after == 'roads':
        graph.add_edge('extract_roads', END)
        return graph.compile()
    def candidates_node(state):
        # 在 Node 3 才读取候选 prompt 系统，前两个节点独立运行。
        return _run_node(state, identify_candidates, client=vlm, max_tokens=max_tokens, config_dir=config_dir)
    graph.add_node('identify_candidates', candidates_node)
    graph.add_edge('extract_roads', 'identify_candidates')
    if stop_after == 'candidates':
        graph.add_edge('identify_candidates', END)
        return graph.compile()
    graph.add_node('draw_candidates', partial(_run_node, function=draw_candidates))
    graph.add_edge('identify_candidates', 'draw_candidates')
    if stop_after == 'visualization':
        graph.add_edge('draw_candidates', END)
        return graph.compile()
    graph.add_node('assess_regions', partial(_run_node, function=assess_regions, client=vlm))
    graph.add_conditional_edges('draw_candidates',
                                lambda state: 'assess' if state.get('run_parallel_vlm') else 'finish',
                                {'assess': 'assess_regions', 'finish': END})
    graph.add_edge('assess_regions', END)
    return graph.compile()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='卫星图定位分析 LangGraph Agent')
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--primary-mode', default='未指定', help='定位方式上下文；Node 3 不选择或依赖主定位模式')
    parser.add_argument('--geo-coordinates', type=json.loads, help='WGS84 坐标 JSON，包含 longitude/latitude；转换为 Gauss 和像素坐标')
    parser.add_argument('--gsd-m', type=float, help='普通 TIFF 的原始 m/pixel；GeoTIFF 使用真实 CRS/transform，输出固定 0.2')
    parser.add_argument('--acquisition-date')
    parser.add_argument('--road-mask', type=Path, help='可选，与处理后 0.2m TIFF 对齐的二值 Road Mask（非零为道路）')
    parser.add_argument('--stop-after', choices=['input', 'roads', 'candidates', 'visualization', 'full'], default='roads', help='默认运行前两个节点；visualization 执行 Node 1～4')
    parser.add_argument('--pathfinder-checkpoint', type=Path)
    parser.add_argument('--pathfinder-device', default='auto', help='auto 优先 GPU；显存不足时选择 CPU，也可指定 cpu/cuda:0')
    parser.add_argument('--pathfinder-threads', type=int, default=4)
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--tile-overlap', type=int, default=64)
    parser.add_argument('--max-output-pixels', type=int, default=64_000_000)
    parser.add_argument('--max-regions', type=int)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--task-id')
    parser.add_argument('--parallel-vlm', action='store_true', help='执行 Node 5；需要填写 loc_type_sys 配置')
    parser.add_argument('--parallel-workers', type=int)
    parser.add_argument('--parallel-max-tokens', type=int)
    parser.add_argument('--save', type=Path, help='额外保存状态摘要 JSON（不含图片 base64）')
    args = parser.parse_args(argv)
    if args.parallel_vlm and args.stop_after != 'full':
        parser.error('--parallel-vlm 需要 --stop-after full')
    if args.geo_coordinates is not None and not isinstance(args.geo_coordinates, dict):
        parser.error('--geo-coordinates 必须是 JSON 对象')
    for name in ('max_regions', 'max_tokens', 'parallel_workers', 'parallel_max_tokens'):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f'--{name.replace("_", "-")} 必须为正整数')
    state: CandidateState = {
        'image_path': str(args.image), 'primary_mode': args.primary_mode, 'path_base': 'project_root',
        'max_regions': args.max_regions, 'gsd_m': args.gsd_m,
        'acquisition_date': args.acquisition_date, 'geo_coordinates': args.geo_coordinates,
        'run_parallel_vlm': args.parallel_vlm, 'parallel_workers': args.parallel_workers,
        'parallel_max_tokens': args.parallel_max_tokens,
        'pathfinder_device': args.pathfinder_device, 'pathfinder_threads': args.pathfinder_threads,
        'tile_size': args.tile_size, 'tile_overlap': args.tile_overlap, 'max_output_pixels': args.max_output_pixels,
    }
    if args.pathfinder_checkpoint is not None: state['pathfinder_checkpoint'] = str(args.pathfinder_checkpoint)
    if args.road_mask is not None: state['road_mask_path'] = str(args.road_mask)
    if args.output_dir is not None: state['output_dir'] = str(args.output_dir)
    if args.task_id is not None: state['task_id'] = args.task_id
    try:
        # 显式开启精细评估时先检查其配置，避免候选推理后才发现空配置。
        if args.parallel_vlm:
            from vlm.parallel_vlm import load_assessment_config
            load_assessment_config()
        with VLMClient() as client:
            result = build_graph(client=client, max_tokens=args.max_tokens, stop_after=args.stop_after).invoke(relative_paths(state))
        bundle = relative_paths({key: value for key, value in result.items()
                                if key not in {'encoded_image', 'road_encoded_image'}})
        text = json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False)
        save_state(result)
        if args.save:
            write_json(resolve_path(args.save), bundle)
        print(text)
        return 1 if result.get('parallel_vlm', {}).get('errors') else 0
    except (VLMError, OSError, ValueError, RasterioError, RuntimeError) as exc:
        print(f'Agent 运行失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
