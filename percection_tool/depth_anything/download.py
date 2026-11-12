"""固定官方 HF revision 下载 V2 Large；不下载/执行远程 Python 代码。"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import shutil
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from agent.paths import resolve_path, write_json

REPOSITORY = 'depth-anything/Depth-Anything-V2-Large-hf'
REVISION = '7581137eff8d4e94f6e796d3baea0e9fa79b22d2'
FILES = {
    'config.json': {'git_blob': 'ecf6e643113d1bb5ae566347665b8d78cea38617'},
    'preprocessor_config.json': {'git_blob': 'a7822a90a0e91a14d50d1996953bac8f389a4377'},
    'model.safetensors': {'sha256': '4e01e34ed5549b529b70b92d53226bc370f03041977b390d3dde45d47f516cf9',
                          'bytes': 1341322868},
    'README.md': {'git_blob': '019a5c3b2fee1e494e5cd9b20888339b3d8b7794'},
}
DEFAULT_DIRECTORY = 'percection_tool/depth_anything/weights/depth_anything_v2_large'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def verify(path: Path, expected: dict) -> bool:
    if not path.is_file():
        return False
    if 'sha256' in expected:
        return path.stat().st_size == expected['bytes'] and sha256(path) == expected['sha256']
    data = path.read_bytes()
    return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest() == expected['git_blob']


def _download_large(url: str, part: Path, size: int) -> None:
    """分段断点续传；按 HTTP Content-Range 校验，最终仍校验整个文件 SHA256。"""
    chunk_size = 1024*1024
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        raise ValueError(f'下载临时文件过大，请检查 {part}')
    pieces = part.with_name(part.name+'.chunks')
    pieces.mkdir(exist_ok=True)
    # 兼容较大分段留下的续传文件，将已有字节拆分缓存，避免重下。
    for old in part.parent.glob(part.name+'.*.chunk'):
        start = int(old.name[len(part.name)+1:-6])
        with old.open('rb') as handle:
            for block in iter(lambda:handle.read(chunk_size),b''):
                cached = pieces/str(start)
                if not cached.exists() or cached.stat().st_size < len(block):
                    cached.write_bytes(block)
                start += len(block)
        old.unlink()
    ranges = [(start,min(size,start+chunk_size)) for start in range(offset,size,chunk_size)]
    def fetch(interval):
        start,stop = interval
        chunk = pieces/str(start)
        for attempt in range(4):
            done = chunk.stat().st_size if chunk.exists() else 0
            if done == stop-start:
                return chunk
            if done > stop-start:
                raise ValueError(f'下载分段长度异常：{chunk}')
            request = urllib.request.Request(url+f'&part={start}&retry={time.time_ns()}',
                headers={'Range':f'bytes={start+done}-{stop-1}'})
            try:
                with urllib.request.urlopen(request,timeout=60) as response:
                    expected = f'bytes {start+done}-{stop-1}/{size}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('服务器未按请求返回正确的分段范围')
                    with chunk.open('ab') as handle:
                        for data in iter(lambda:response.read(1024*1024),b''):
                            handle.write(data)
            except (OSError,ValueError):
                if attempt == 3:
                    raise
        if chunk.stat().st_size != stop-start:
            raise ValueError(f'分段未下载完整，重跑命令可继续：{chunk}')
        return chunk
    with ThreadPoolExecutor(max_workers=16) as executor:
        for number,chunk in enumerate(executor.map(fetch,ranges),1):
            with part.open('ab') as output,chunk.open('rb') as source:
                shutil.copyfileobj(source,output,1024*1024)
            chunk.unlink()
            if number % 64 == 0 or number == len(ranges):
                print(f'模型下载 {part.stat().st_size}/{size} bytes',flush=True)
    if not any(pieces.iterdir()):
        pieces.rmdir()


def download(directory: str = DEFAULT_DIRECTORY) -> Path:
    target = resolve_path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = target/name
        if verify(path, expected):
            print(f'已校验 {name}', flush=True)
            continue
        part = target/(name+'.part')
        url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}?download=true&v={int(time.time())}'
        if 'bytes' in expected:
            _download_large(url,part,expected['bytes'])
        else:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request,timeout=60) as response, part.open('wb') as handle:
                shutil.copyfileobj(response,handle,1024*1024)
        if not verify(part, expected):
            raise ValueError(f'{name} 校验失败；删除对应 .part 后重试')
        part.replace(path)
        print(f'下载并校验 {name}', flush=True)
    write_json(target/'SOURCE.json', {'repository': REPOSITORY, 'revision': REVISION,
        'model_card': f'https://huggingface.co/{REPOSITORY}', 'license': 'cc-by-nc-4.0',
        'files': {name: {'sha256': sha256(target/name), 'bytes': (target/name).stat().st_size} for name in FILES}})
    return target


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    print(download(args.output_dir))
