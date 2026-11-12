"""python -m agent.map_preprocess.main --image ... --config ..."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.map_preprocess.config import load_config
from agent.map_preprocess.graph import run_pipeline
from agent.map_preprocess.io import resolve_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='卫星地图沿路预处理（启用 vlm_review_enabled 时 Node 6 调用本地 VLM）')
    parser.add_argument('--image', type=Path, required=True, help='uint8 RGB 图片路径（道路拓扑 + 语义）')
    parser.add_argument('--dsm', type=Path, help='显式指定 DSM（米）；省略时查找与 *_dom.tif 同目录的 *_dsm.tif')
    parser.add_argument('--config', type=Path, default=Path('agent/map_preprocess/pipeline.yaml'), help='JSON/YAML 配置路径，默认 agent/map_preprocess/pipeline.yaml')
    args = parser.parse_args(argv)
    try:
        state = run_pipeline(str(args.image), load_config(args.config),
                             dsm_path=str(args.dsm) if args.dsm else None, console=True)
        if state.get('final_summary'):
            print(f"最终结论：{state['final_summary']['message']}")
        print(resolve_path(state.get('manifest_path') or state['output_dir']))  # 最后一行固定为 manifest 路径
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f'预处理失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
