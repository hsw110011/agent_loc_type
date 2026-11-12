"""对已有 global_blocks manifest 单独运行 Node 5，结果写回同一 manifest。

python -m agent.map_preprocess.run_block_features --manifest output/.../<task_id>/state/manifest.json [--dsm dsm.tif] [--config pipeline.yaml]
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dsm', type=Path, help='可选：给定时重新检查并使用该 DSM；省略时沿用 manifest 记录的 DSM（没有则只判语义）')
    parser.add_argument('--config', type=Path, help='global_blocks 配置；省略时沿用 manifest 记录的 Node 1-4 配置，Node 5 参数取当前默认值')
    args = parser.parse_args(argv)
    try:
        if args.config:
            config = load_config(args.config)
            if not isinstance(config, GlobalPreprocessConfig):
                raise ValueError('需要 workflow: global_blocks 配置')
        else:
            # manifest 只沿用 Node 1-4 的配置；Node 5 参数（feature_*）一律取当前默认值，
            # 避免旧 manifest 里过期或已删除的指标参数覆盖新定义。
            config = GlobalPreprocessConfig.from_recorded(load_manifest(args.manifest)['config'], keep_features=False)
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        from agent.map_preprocess.nodes.block_features import run_block_features
        summary = run_block_features(args.manifest, config, dsm_path=args.dsm)
        print(resolve_path(summary['path']))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f'Node 5 失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
