"""对已有 global_blocks manifest（已跑 Node 6）单独运行 Node 7 退化区域尺寸扩展，结果单独记录，不改块和 Node 6 结论。

python -m agent.map_preprocess.run_vlm_expansion --manifest output/.../<task_id>/state/manifest.json \
    [--config agent/map_preprocess/pipeline.yaml] [--block-ids E0_B1 E3_B0] [--dry-run] [--resume]

--dry-run 或 --block-ids 将结果写入独立的诊断目录。--dry-run 只生成图片和 prompt。
--resume 仅复用同流程版本、同请求内容的成功结果。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.map_preprocess.config import load_config
from agent.map_preprocess.global_config import GlobalPreprocessConfig
from agent.map_preprocess.io import load_manifest, resolve_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='global_blocks 配置；省略时沿用 manifest 的流程开关，扩展推理参数取当前默认值')
    parser.add_argument('--block-ids', nargs='+', help='只处理指定块，结果写入诊断目录')
    parser.add_argument('--dry-run', action='store_true', help='只渲染输入与 prompt，不调用 VLM')
    parser.add_argument('--resume', action='store_true', help='复用同流程版本、同请求内容的成功结果')
    args = parser.parse_args(argv)
    try:
        if args.config:
            config = load_config(args.config)
            if not isinstance(config, GlobalPreprocessConfig):
                raise ValueError('需要 workflow: global_blocks 配置')
        else:
            preserved = {'vlm_expansion_enabled', 'vlm_expansion_reasoning_effort', 'vlm_expansion_thinking_budget',
                         'vlm_expansion_max_tokens'}
            recorded = {key: value for key, value in load_manifest(args.manifest)['config'].items()
                        if key in preserved or not key.startswith('vlm_expansion_')}
            config = GlobalPreprocessConfig.from_recorded(recorded)
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        from agent.map_preprocess.nodes.vlm_expansion import run_vlm_expansion
        summary = run_vlm_expansion(args.manifest, config, block_ids=args.block_ids, dry_run=args.dry_run or None, resume=args.resume)
        if summary.get('path'):
            print(resolve_path(summary['path']))
        print({k: summary.get(k) for k in ('status', 'job_count', 'jobs_by_type', 'skipped', 'max_size_counts',
                                           'stop_reason_counts', 'errors') if k in summary})
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f'Node 7 失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
