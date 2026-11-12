"""首次推理启动本机 EXL3 Qwen3.8-27B + DFlash2 服务（exl3_server.py）；线程共享一份模型。"""
from __future__ import annotations

import atexit
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import tempfile
import time
import urllib.request
from urllib.parse import urlparse
from datetime import datetime
from zoneinfo import ZoneInfo


class VLMRuntime:
    def __init__(self, base_url: str, model: str):
        self.base_url, self.model = base_url, model
        self.max_num_seqs = int(os.getenv('VLM_MAX_NUM_SEQS', '5'))
        if self.max_num_seqs <= 0:
            raise ValueError('VLM_MAX_NUM_SEQS 必须为正整数')
        self.autostart = os.getenv('VLM_AUTOSTART', '1') == '1'
        self.start_timeout = float(os.getenv('VLM_START_TIMEOUT', '900'))
        # 服务已启动后的探活容忍时间（秒）：忙于推理时 /models 可能响应慢
        self.ready_timeout = float(os.getenv('VLM_READY_TIMEOUT', '60'))
        if self.start_timeout <= 0:
            raise ValueError('VLM_START_TIMEOUT 必须为正数')
        from agent.paths import PROJECT_ROOT
        self.log_dir = PROJECT_ROOT / 'output/vlm/log'
        self.process = None
        self.log_path = None
        self._lock_file = None
        self._mutex = threading.RLock()
        self._http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        atexit.register(self.close)

    def _server_command(self, port):
        """用 Agent 同一个 Python 启动 exl3_server.py 独立进程；Agent 进程本身不导入 ExLlamaV3。"""
        import importlib.util
        root = Path(__file__).resolve().parents[2]
        from .common import resolve_path
        model_root = resolve_path(os.getenv('VLM_MODEL_ROOT', str(root / 'qwen3827b-exl3-dflash2')))
        python = Path(os.getenv('VLM_PYTHON', sys.executable))
        target = resolve_path(os.getenv('VLM_MODEL_PATH', str(model_root / 'models/qwen38-27b-exl3')))
        default_draft = str(model_root / 'models/dflash2-exl3') if self.max_num_seqs == 1 else ''
        draft = os.getenv('VLM_DRAFT_PATH', default_draft)
        draft = '' if draft.strip().lower() in {'', 'none', 'off'} else str(resolve_path(draft))
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError(f'Python 不可用: {python}')
        if 'VLM_PYTHON' not in os.environ and importlib.util.find_spec('exllamav3') is None:
            raise ValueError('当前环境未安装 ExLlamaV3；按 agent/vlm/README.md 编译安装到项目 .venv')
        for name, path in (('目标模型', target), ('草稿模型', Path(draft) if draft else None)):
            if path is not None and not (path / 'config.json').is_file():
                raise ValueError(f'{name}配置不存在: {path}/config.json')
        # Five full Node 6 requests reserve roughly 5 × 14k tokens, rounded to cache pages.
        default_cache = '73728' if self.max_num_seqs >= 4 else '32768'
        command = [str(python), str(Path(__file__).resolve().parent / 'exl3_server.py'),
                   '--target', str(target), '--draft', draft,
                   '--served-model-name', self.model, '--host', '127.0.0.1', '--port', str(port),
                   '--cache-tokens', os.getenv('VLM_CACHE_TOKENS', default_cache),
                   '--cq', os.getenv('VLM_CACHE_BITS', '3'),
                   '--max-batch-size', str(self.max_num_seqs),
                   '--max-model-len', os.getenv('VLM_MAX_MODEL_LEN', '32768'),
                   '--max-images', os.getenv('VLM_MAX_IMAGES', '4'),
                   '--max-pixels', os.getenv('VLM_MAX_IMAGE_PIXELS', '1048576')]
        # 不继承 PYTHONPATH，避免模型项目源码目录里为其它 torch 编译的扩展遮蔽已安装的 ExLlamaV3。
        env = {k: v for k, v in os.environ.items() if k not in {'PYTHONPATH', 'PYTHONHOME'}}
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
        # RTX 3090 (sm_86) 调优默认值（与模型项目 README 一致），环境变量已设置时保持外部值。
        for key, value in (('EXL3_HGEMM_F16ACC', 'auto'), ('EXL3_INT8_GEMV', '2'),
                           ('EXL3_INT8_GEMV_MAX_K', '5'), ('EXL3_QC_STAGING', '1')):
            env.setdefault(key, value)
        return command, env

    def _ready(self, timeout: float = 2.):
        try:
            with self._http.open(self.base_url.rstrip('/') + '/models', timeout=timeout) as response:
                data = json.load(response)
            return self.model in [item.get('id') for item in data.get('data', [])]
        except (OSError, ValueError):
            return False

    def _wait_ready(self, seconds: float) -> bool:
        """已启动的服务忙于推理时 /models 可能短暂超时；进程仍在时重试到 seconds 秒，避免把繁忙误判为退出。"""
        deadline = time.monotonic() + seconds
        while self.process.poll() is None:
            if self._ready(timeout=5):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(1)
        return False

    def ensure_started(self):
        if not self.autostart:
            return  # 显式外部服务模式，由调用者负责部署与生命周期。
        with self._mutex:
            if self.process is not None:
                if self._wait_ready(self.ready_timeout):
                    return
                raise RuntimeError(f'VLM 服务已退出或不可用；查看 {self.log_path}')
            parsed = urlparse(self.base_url)
            if parsed.hostname not in {'127.0.0.1', 'localhost'} or parsed.path.rstrip('/') != '/v1':
                raise ValueError('自动启动仅支持 127.0.0.1/localhost 的 /v1 地址')
            port = parsed.port or 80
            self.log_dir.mkdir(parents=True, exist_ok=True)
            project_key = hashlib.sha256(str(Path(__file__).resolve().parent).encode()).hexdigest()[:12]
            self._lock_file = (Path(tempfile.gettempdir()) / f'hsw_vlm_{project_key}_{port}.lock').open('a')
            try:
                try:
                    fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(f'端口 {port} 已由另一个 VLM 任务管理；请共享同一个 VLMClient') from exc
                with socket.socket() as probe:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        probe.bind(('127.0.0.1', port))
                    except OSError as exc:
                        raise RuntimeError(f'端口 {port} 已占用，无法确认已有服务为本项目 EXL3 服务；请先关闭旧服务，或显式设置 VLM_AUTOSTART=0') from exc
                # 感知模型已离开作用域，释放其显存缓存，不在 agent 进程导入 VLM。
                gc.collect()
                torch = sys.modules.get('torch')
                if torch is not None and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
                self.log_path = self.log_dir / f'{timestamp}_vlm.log'
                command, env = self._server_command(port)
                if env.get('VLM_API_KEY'):
                    raise ValueError('自动启动服务暂不配置认证；请移除 VLM_API_KEY 或使用外部服务模式')
                with self.log_path.open('a') as log:
                    self.process = subprocess.Popen(
                        command,
                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + self.start_timeout
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError(f'EXL3 服务启动失败；查看 {self.log_path}')
                    if self._ready():
                        return
                    time.sleep(0.5)
                raise RuntimeError(f'EXL3 服务启动超时；查看 {self.log_path}')
            except BaseException:
                self.close()
                raise

    def close(self):
        """仅停止本实例启动的进程组；可重复调用。"""
        with self._mutex:
            if self.process is not None:
                process, self.process = self.process, None
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                except ProcessLookupError:
                    process.wait()
            if self._lock_file is not None:
                self._lock_file.close()
                self._lock_file = None
