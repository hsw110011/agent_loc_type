"""导出 Node 5 指标表：中英文定义、分布统计、逐块数值（CSV + Markdown），用于估计阈值。

python -m agent.map_preprocess.export_feature_table --manifest output/.../manifest.json
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from agent.map_preprocess import output_layout as layout
from agent.map_preprocess.feature_definitions import METRICS, flatten
from agent.map_preprocess.io import load_manifest, relative_path, resolve_path

NUMERIC = [m for m in METRICS if m['unit'] in ('%', '-', 'm')]


def _cell(value) -> str:
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'yes' if value else 'NO'
    if isinstance(value, float):
        return f'{value:.3f}' if abs(value) <= 1 and value != 0 else f'{value:.1f}'
    return str(value)


def _markdown(header: list[str], rows: list[list[str]]) -> str:
    lines = ['| '+' | '.join(header)+' |', '|'+'---|'*len(header)]
    lines += ['| '+' | '.join(row)+' |' for row in rows]
    return '\n'.join(lines)


def definitions_table() -> str:
    rows = [[m['column'], m['name_zh'], m['name_en'], m['unit'], m['range'], m['definition_zh']] for m in METRICS]
    return _markdown(['column', '中文名', 'English', '单位', '范围', '定义'], rows)


def stats_table(rows: list[dict]) -> str:
    table = []
    for metric in NUMERIC:
        values = np.array([r[metric['column']] for r in rows if r[metric['column']] is not None], float)
        if not values.size:
            continue
        p = np.percentile(values, [0, 10, 25, 50, 75, 90, 100])
        table.append([metric['column'], metric['name_zh'], *(f'{x:.3f}' if metric['unit'] == '-' else f'{x:.1f}' for x in p)])
    return _markdown(['column', '中文名', 'min', 'p10', 'p25', 'median', 'p75', 'p90', 'max'], table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', help='默认任务目录 state/block_features_table/')
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    rows = [flatten(b) for b in manifest['blocks'] if b.get('features')]
    if not rows:
        print('manifest 尚无 Node 5 features', file=sys.stderr)
        return 1
    directory = (resolve_path(args.output_dir) if args.output_dir
                 else layout.state_dir(layout.task_dir_of_manifest(args.manifest))/'block_features_table')
    directory.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    # utf-8-sig 便于 Excel 直接打开中文表头
    with (directory/'features.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        names = {m['column']: m['name_zh'] for m in METRICS}
        writer.writerow(columns)
        writer.writerow([names.get(c, {'block_id': '块ID', 'edge_id': '道路边ID'}.get(c, c)) for c in columns])
        for row in rows:
            writer.writerow([row[c] for c in columns])
    blocks = _markdown(columns, [[_cell(row[c]) for c in columns] for row in rows])
    report = ('# Node 5 指标表\n\n## 指标定义\n\n'+definitions_table()+
              f'\n\n## 分布（{len(rows)} 块）\n\n'+stats_table(rows)+'\n\n## 逐块数值\n\n'+blocks+'\n')
    (directory/'features.md').write_text(report, encoding='utf-8')
    print(relative_path(directory/'features.csv'))
    print(relative_path(directory/'features.md'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
