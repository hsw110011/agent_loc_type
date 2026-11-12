"""把 SegEarth-OV-3 的 SAM 3 权重导入本工具目录，并记录来源与 SHA256。

python -m percection_tool.segearth_ov3.import_model [--source-dir ../Geoseg_hsw/SegEarth-OV-3]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = TOOL_DIR.parents[2]/'Geoseg_hsw'/'SegEarth-OV-3'
# 权重之外一并记录的上游文件：推理逻辑移植自后两者，分词表随 vendor/sam3 使用。
TRACKED = ('sam3/assets/bpe_simple_vocab_16e6.txt.gz', 'segearthov3_segmentor.py', 'segment_geotiff.py')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(16*1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def import_model(source_dir: Path, output_dir: Path) -> Path:
    checkpoint = source_dir/'weights'/'sam3'/'sam3.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected = sha256(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir/'sam3.pt'
    if target.exists():
        if sha256(target) != expected:
            raise ValueError(f'{target} 已存在且内容不同，拒绝覆盖；请另选 --output-dir')
    else:
        pending = target.with_name('.sam3.pt.partial')
        shutil.copyfile(checkpoint, pending)
        if sha256(pending) != expected:
            pending.unlink()
            raise RuntimeError('权重复制后 SHA256 不符')
        pending.replace(target)
    commit = subprocess.run(['git', '-C', str(source_dir), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    record = {'source_dir': str(source_dir.resolve()), 'repository': 'https://github.com/earth-insights/SegEarth-OV-3',
              'commit': commit.stdout.strip() or None, 'imported_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'files': {'sam3.pt': {'sha256': expected, 'bytes': target.stat().st_size}},
              'upstream_files': {name: {'sha256': sha256(source_dir/name)} for name in TRACKED}}
    (output_dir/'SOURCE.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return output_dir/'SOURCE.json'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=TOOL_DIR/'weights')
    args = parser.parse_args(argv)
    try:
        print(import_model(args.source_dir, args.output_dir))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'导入失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
