"""整图调试总览：保留边、全部块及 ID、区域边界。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.map_preprocess.global_config import GlobalPreprocessConfig
from agent.map_preprocess.io import load_manifest, resolve_path
from agent.map_preprocess.utils.global_render import render_global


def render_overview(manifest_path: str | Path, output_path: str | Path) -> Path:
    """整图缩略图上画保留边、全部块及 ID、3×3 归属区域边界。"""
    manifest = load_manifest(manifest_path)
    config = GlobalPreprocessConfig.from_recorded(manifest['config'])
    render_global(manifest['image'], output_path, config, edges=manifest['edges'],
                  blocks=manifest['blocks'], tiles=manifest['tiles'])
    return resolve_path(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='从 manifest 绘制全局采样点/ID/Tile 边界总览图')
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        print(render_overview(args.manifest, args.output))
        return 0
    except (OSError, ValueError) as exc:
        print(f'总览图生成失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
