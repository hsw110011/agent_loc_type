"""对已有 global_blocks manifest（已跑 Node 5）单独运行 Node 6 退化块 VLM 复核，结果写回同一 manifest。

python -m agent.map_preprocess.run_vlm_review --manifest output/.../<task_id>/state/manifest.json \
    [--config agent/map_preprocess/pipeline.yaml] [--block-ids E0_B1 E3_B0] [--dry-run]

--dry-run 或 --block-ids 将结果写入独立的诊断目录。--dry-run 只生成图片和 prompt。
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
    parser.add_argument('--config', type=Path, help='global_blocks 配置；省略时沿用 manifest 记录的思考档位和预算')
    parser.add_argument('--block-ids', nargs='+', help='只评估指定块，结果写入诊断目录')
    parser.add_argument('--dry-run', action='store_true', help='只渲染输入与 prompt，不调用 VLM')
    args = parser.parse_args(argv)
    try:
        if args.config:
            config = load_config(args.config)
            if not isinstance(config, GlobalPreprocessConfig):
                raise ValueError('需要 workflow: global_blocks 配置')
        else:
            preserved = {'vlm_review_enabled', 'vlm_review_reasoning_effort', 'vlm_review_thinking_budget', 'vlm_review_max_tokens'}
            recorded = {k: v for k, v in load_manifest(args.manifest)['config'].items()
                        if k in preserved or not k.startswith('vlm_review_')}
            config = GlobalPreprocessConfig.from_recorded(recorded)
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        from agent.map_preprocess.nodes.vlm_review import run_vlm_review
        summary = run_vlm_review(args.manifest, config, block_ids=args.block_ids, dry_run=args.dry_run or None)
        print(resolve_path(summary['path']))
        print({k: summary.get(k) for k in ('status', 'job_count', 'jobs_by_type', 'removed_block_ids',
                                           'degraded_scene_ids', 'rejected_block_ids', 'errors') if k in summary})
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f'Node 6 失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
