"""对已有 global_blocks manifest（已跑 Node 8）单独运行 Node 9 辅助定位方式推荐，结果写入 state/localization_recommendations.json。

python -m agent.map_preprocess.run_localization_recommendation --manifest output/.../<task_id>/state/manifest.json \
    [--config agent/map_preprocess/pipeline.yaml] [--dry-run]

--dry-run 只生成输入图片和 prompt，不调用 VLM、不改 manifest。
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
    parser.add_argument('--config', type=Path, help='global_blocks 配置；省略时沿用 manifest 的流程开关，定位推理参数取当前默认值')
    parser.add_argument('--dry-run', action='store_true', help='只渲染输入与 prompt，不调用 VLM')
    args = parser.parse_args(argv)
    try:
        if args.config:
            config = load_config(args.config)
            if not isinstance(config, GlobalPreprocessConfig):
                raise ValueError('需要 workflow: global_blocks 配置')
        else:
            preserved = {'vlm_localization_enabled', 'vlm_localization_reasoning_effort',
                         'vlm_localization_thinking_budget', 'vlm_localization_max_tokens'}
            recorded = {key: value for key, value in load_manifest(args.manifest)['config'].items()
                        if key in preserved or not key.startswith('vlm_localization_')}
            config = GlobalPreprocessConfig.from_recorded(recorded)
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        from agent.map_preprocess.nodes.localization_recommendation import run_localization_recommendation
        summary = run_localization_recommendation(args.manifest, config, dry_run=args.dry_run or None)
        if summary.get('localization_recommendations'):
            print(resolve_path(summary['localization_recommendations']))
        print({k: summary.get(k) for k in ('status', 'region_count', 'recommended_count', 'failed_count',
                                           'method_counts', 'errors') if k in summary})
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f'Node 9 失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
