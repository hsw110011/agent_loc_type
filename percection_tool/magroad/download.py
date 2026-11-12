"""下载官方 MaGRoad checkpoint（README Model Zoo 的 Google Drive 链接），断点续传并校验。

上游未公布 SHA256：大小与 Drive 返回的 content-length 一致后计算 SHA256，
固定在 inference.CHECKPOINTS 中，后续下载/加载都必须匹配。
"""
from __future__ import annotations

import argparse
import http.client
from pathlib import Path
import sys
import urllib.request

from agent.paths import resolve_path, write_json
from .inference import CHECKPOINTS, WEIGHTS_DIR, CheckpointSpec, file_sha256

DRIVE_URL = 'https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t'


def download_checkpoint(spec: CheckpointSpec, weights_dir: str | Path = WEIGHTS_DIR) -> Path:
    path = resolve_path(weights_dir)/spec.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    url = DRIVE_URL.format(id=spec.drive_id)
    if path.is_file():
        if file_sha256(path) != spec.sha256:
            raise ValueError(f'已有 {path.name} 的 SHA256 不符；保留文件，请检查或删除后重下')
        return path
    temporary = path.with_suffix(path.suffix+'.partial')
    for attempt in range(6):
        begin = temporary.stat().st_size if temporary.exists() else 0
        if begin == spec.size_bytes:
            break
        print(f'{spec.name} 下载尝试 {attempt+1}/6，已有 {begin/1024**2:.1f} MiB', file=sys.stderr, flush=True)
        request = urllib.request.Request(url, headers={'Range': f'bytes={begin}-'} if begin else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                append = begin > 0 and response.status == 206
                with temporary.open('ab' if append else 'wb') as target:
                    for block in iter(lambda: response.read(1024*1024), b''):
                        target.write(block)
        except (OSError, http.client.HTTPException) as exc:
            print(f'连接中断，保留已下载内容：{exc}', file=sys.stderr, flush=True)
    if not temporary.exists() or temporary.stat().st_size != spec.size_bytes:
        raise ValueError(f'{spec.name} 下载不完整，保留 partial 供重试')
    digest = file_sha256(temporary)
    if digest != spec.sha256:
        raise ValueError(f'{spec.name} SHA256 {digest} 与记录不符，保留 partial 供检查')
    temporary.replace(path)
    write_json(path.with_suffix('.source.json'), {
        'repository': 'https://github.com/xiaofei-guan/MaGRoad', 'url': url,
        'upstream_filename': spec.upstream_filename, 'sha256': digest, 'size_bytes': spec.size_bytes})
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='下载并校验官方 MaGRoad 权重')
    parser.add_argument('--checkpoint', choices=[*CHECKPOINTS, 'all'], default='all')
    parser.add_argument('--weights-dir', default=WEIGHTS_DIR)
    args = parser.parse_args(argv)
    names = list(CHECKPOINTS) if args.checkpoint == 'all' else [args.checkpoint]
    try:
        for name in names:
            print(download_checkpoint(CHECKPOINTS[name], args.weights_dir))
        return 0
    except (OSError, ValueError) as exc:
        print(f'MaGRoad 下载失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
